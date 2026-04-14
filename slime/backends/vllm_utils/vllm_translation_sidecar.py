"""
vLLM Translation Sidecar
========================

Lightweight FastAPI process co-located with each vLLM engine.
Receives SGLang-format ``/generate`` requests from the SlimeRouter,
translates them to vLLM ``/v1/completions``, and translates responses back.

Also proxies lifecycle endpoints:
  /health, /health_generate, /abort_request, /flush_cache, /get_weight_version

See docs/en/vllm/ROUTER_DESIGN.md for the full specification.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sampling-param translation tables
# ---------------------------------------------------------------------------

# SGLang name → vLLM /v1/completions name  (only entries that differ)
_PARAM_RENAME = {
    "max_new_tokens": "max_tokens",
    "no_stop_trim": "include_stop_str_in_output",
    "sampling_seed": "seed",
}

# Parameters passed through unchanged
_PARAM_DIRECT = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "stop_token_ids",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    }
)

# finish_reason vLLM → SGLang-style {"type": ...}
_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    None: "abort",
}


# ---------------------------------------------------------------------------
# Request / response translation helpers
# ---------------------------------------------------------------------------


def translate_generate_request(
    body: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Translate an SGLang-format /generate request → vLLM /v1/completions payload."""

    sp: dict = body.get("sampling_params", {})

    vllm_payload: dict[str, Any] = {
        "model": model_name,
        # vLLM accepts list[int] as a pre-tokenized prompt
        "prompt": body.get("input_ids") or body.get("input_tokens", []),
        "stream": False,
    }

    # --- direct-copy params ---
    for key in _PARAM_DIRECT:
        if key in sp:
            vllm_payload[key] = sp[key]

    # --- renamed params ---
    for src, dst in _PARAM_RENAME.items():
        if src in sp:
            vllm_payload[dst] = sp[src]

    # --- logprob handling ---
    if body.get("return_logprob", False):
        vllm_payload["logprobs"] = 1
        # request token IDs alongside logprobs
        # NOTE: must be a top-level param; "extra_body" is an OpenAI SDK
        # client concept and is ignored by vLLM's raw HTTP API.
        vllm_payload["return_token_ids"] = True

    return vllm_payload


def translate_vllm_response(
    vllm_resp: dict[str, Any],
    weight_version: int,
) -> dict[str, Any]:
    """Translate a vLLM /v1/completions response → SGLang-format JSON."""

    choice: dict = vllm_resp.get("choices", [{}])[0]
    usage: dict = vllm_resp.get("usage", {})

    # --- output token IDs ---
    output_ids: list[int] = choice.get("token_ids", [])

    # --- logprobs: zip(logprob, token_id) ---
    output_token_logprobs: list[list[float | int]] = []
    logprobs_obj = choice.get("logprobs")
    if logprobs_obj and output_ids:
        raw_lp: list[float | None] = logprobs_obj.get("token_logprobs", [])
        output_token_logprobs = [
            [float(lp) if lp is not None else 0.0, tid]
            for lp, tid in zip(raw_lp, output_ids)
        ]

    # --- finish reason ---
    raw_reason = choice.get("finish_reason")
    mapped = _FINISH_REASON_MAP.get(raw_reason, raw_reason or "abort")
    finish_reason = {"type": mapped}

    meta_info: dict[str, Any] = {
        "finish_reason": finish_reason,
        "weight_version": weight_version,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", len(output_ids)),
        "cached_tokens": 0,
    }
    # Only include output_token_logprobs when we have valid paired data;
    # a None value causes RadixTreeMiddleware to silently fail when iterating.
    if output_token_logprobs:
        meta_info["output_token_logprobs"] = output_token_logprobs

    return {
        "text": choice.get("text", ""),
        "output_ids": output_ids,
        "meta_info": meta_info,
    }


# ---------------------------------------------------------------------------
# Sidecar application
# ---------------------------------------------------------------------------


class TranslationSidecar:
    """Manages state and provides the FastAPI app for the translation sidecar."""

    def __init__(
        self,
        vllm_base_url: str,
        model_name: str,
        *,
        timeout: float = 600.0,
        max_connections: int = 256,
        weight_transfer_backend: str = "nccl",
    ):
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.model_name = model_name
        self._weight_version: int = 0
        self._weight_transfer_backend = weight_transfer_backend
        self._active_connections: set[httpx.Response] = set()
        self._lock = asyncio.Lock()

        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout
        self._max_connections = max_connections

        self.app = self._build_app()

    # ---- lifecycle -----------------------------------------------------------

    async def startup(self):
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_connections,
                max_keepalive_connections=self._max_connections,
            ),
            timeout=httpx.Timeout(self._timeout),
        )

    async def shutdown(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ---- app factory ---------------------------------------------------------

    def _build_app(self) -> FastAPI:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            yield
            await self.shutdown()

        app = FastAPI(title="vLLM Translation Sidecar", lifespan=lifespan)

        app.post("/generate")(self.generate)
        app.get("/health")(self.health)
        app.get("/health_generate")(self.health_generate)
        app.post("/abort_request")(self.abort_request)
        app.get("/flush_cache")(self.flush_cache)
        app.get("/get_weight_version")(self.get_weight_version)
        app.post("/set_weight_version")(self.set_weight_version)
        app.post("/update_weights_from_tensor")(self.update_weights_from_tensor)

        return app

    # ---- endpoints -----------------------------------------------------------

    async def generate(self, request: Request):
        """Translate SGLang /generate → vLLM /v1/completions → SGLang response."""

        body = await request.json()
        vllm_payload = translate_generate_request(body, self.model_name)

        url = f"{self.vllm_base_url}/v1/completions"

        resp: httpx.Response | None = None
        try:
            async with self._lock:
                # We don't actually hold the lock during the request,
                # just use it to safely add to the tracking set.
                pass

            resp = await self._client.post(url, json=vllm_payload)
            self._active_connections.add(resp)

            resp.raise_for_status()
            vllm_data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "vLLM /v1/completions returned %s: %s",
                exc.response.status_code,
                exc.response.text[:2000],
            )
            # Return an abort-style response so the middleware retries
            return JSONResponse(
                content={
                    "text": "",
                    "output_ids": [],
                    "meta_info": {
                        "finish_reason": {"type": "abort"},
                        "weight_version": self._weight_version,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                    },
                },
                status_code=200,
            )
        except Exception as exc:
            logger.error("Error calling vLLM: %s", exc, exc_info=True)
            return JSONResponse(
                content={
                    "text": "",
                    "output_ids": [],
                    "meta_info": {
                        "finish_reason": {"type": "abort"},
                        "weight_version": self._weight_version,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                    },
                },
                status_code=200,
            )
        finally:
            if resp is not None:
                self._active_connections.discard(resp)
                await resp.aclose()

        translated = translate_vllm_response(vllm_data, self._weight_version)
        return JSONResponse(content=translated)

    async def health(self):
        """Proxy to vLLM's built-in /health endpoint."""
        try:
            resp = await self._client.get(f"{self.vllm_base_url}/health", timeout=5.0)
            return JSONResponse(content={"status": "ok"}, status_code=resp.status_code)
        except Exception:
            return JSONResponse(content={"status": "unhealthy"}, status_code=503)

    async def health_generate(self):
        """
        Startup readiness probe.

        Checks vLLM /health and optionally fires a dummy /v1/completions
        with max_tokens=1 to verify end-to-end readiness.
        """
        try:
            resp = await self._client.get(f"{self.vllm_base_url}/health", timeout=10.0)
            if resp.status_code != 200:
                return JSONResponse(content={"status": "not_ready"}, status_code=503)

            # Lightweight smoke test: single-token completion
            dummy_payload = {
                "model": self.model_name,
                "prompt": "hi",
                "max_tokens": 1,
                "stream": False,
            }
            resp2 = await self._client.post(
                f"{self.vllm_base_url}/v1/completions",
                json=dummy_payload,
                timeout=30.0,
            )
            if resp2.status_code == 200:
                return JSONResponse(content={"status": "ready"}, status_code=200)
            else:
                return JSONResponse(content={"status": "not_ready"}, status_code=503)
        except Exception as exc:
            logger.debug("health_generate check failed: %s", exc)
            return JSONResponse(content={"status": "not_ready"}, status_code=503)

    async def abort_request(self, request: Request):
        """
        Handle abort requests.

        vLLM uses HTTP disconnect for cancellation.  We close all active
        connections to the vLLM backend, which triggers vLLM's
        ``@with_cancellation`` decorator to abort in-flight requests.

        Alternatively, use pause/resume for a cleaner between-generation abort.
        """
        body = await request.json()

        if body.get("abort_all", False):
            # Strategy: pause + resume clears the pipeline
            try:
                await self._client.post(
                    f"{self.vllm_base_url}/pause",
                    params={"mode": "abort"},
                    timeout=30.0,
                )
                await self._client.post(
                    f"{self.vllm_base_url}/resume",
                    timeout=30.0,
                )
            except Exception as exc:
                logger.warning("pause/resume abort failed, falling back to connection close: %s", exc)
                # Fallback: close all tracked connections
                conns = list(self._active_connections)
                self._active_connections.clear()
                for conn in conns:
                    try:
                        await conn.aclose()
                    except Exception:
                        pass

        return JSONResponse(content={"status": "ok"})

    async def flush_cache(self):
        """
        Flush the KV cache.

        vLLM equivalent: sleep(level=1) + wake_up(tags=kv_cache).
        """
        try:
            await self._client.post(
                f"{self.vllm_base_url}/sleep",
                params={"level": "1", "mode": "abort"},
                timeout=30.0,
            )
            await self._client.post(
                f"{self.vllm_base_url}/wake_up",
                params={"tags": "kv_cache"},
                timeout=30.0,
            )
            return JSONResponse(content={"status": "ok"})
        except Exception as exc:
            logger.warning("flush_cache failed: %s", exc)
            return JSONResponse(content={"status": "ok"})

    async def get_weight_version(self):
        """Return the sidecar-tracked weight version counter."""
        return JSONResponse(content={"weight_version": self._weight_version})

    async def update_weights_from_tensor(self, request: Request):
        """Receive serialized CUDA IPC tensor handles, reconstruct weights, and
        load them into the vLLM backend.

        This mirrors the SGLang ``/update_weights_from_tensor`` endpoint so that
        the FSDP / Megatron ``UpdateWeightFromTensor`` codepath works with vLLM.

        **IPC path** (``weight_transfer_backend == "ipc"``):
        Deserializes GPU tensors → creates CUDA IPC handles via
        ``reduce_tensor`` → sends to vLLM ``POST /update_weights`` with
        ``ipc_handles_pickled``.  Zero-copy, no temp files, no CPU round-trip.

        **Safetensors fallback** (any other backend):
        Saves tensors to a temp ``.safetensors`` file and calls
        ``POST /load_weights``.
        """
        import torch

        from slime.utils.multiprocessing_serializer import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
            monkey_patch_torch_reductions,
        )

        body = await request.json()
        serialized_named_tensors: list[str] = body["serialized_named_tensors"]
        load_format: str | None = body.get("load_format")
        flush_cache: bool = body.get("flush_cache", False)
        weight_version: str | None = body.get("weight_version")

        # Ensure CUDA IPC handles use device UUIDs for correct cross-process mapping.
        monkey_patch_torch_reductions()

        # ── Deserialize all tensor payloads ──────────────────────────────
        all_named_tensors: list[tuple[str, torch.Tensor]] = []
        for serialized in serialized_named_tensors:
            data = MultiprocessingSerializer.deserialize(serialized)
            if load_format == "flattened_bucket":
                # When the training actor uses torch_memory_saver's LD_PRELOAD
                # (VMM), tensors arrive as CPU to avoid CUDA IPC
                # incompatibility.  Move to GPU before reconstructing.
                ft = data["flattened_tensor"]
                if not ft.is_cuda:
                    ft = ft.cuda()
                bucket = FlattenedTensorBucket(
                    flattened_tensor=ft,
                    metadata=data["metadata"],
                )
                all_named_tensors.extend(bucket.reconstruct_tensors())
            else:
                if isinstance(data, list):
                    all_named_tensors.extend(
                        (n, t.cuda() if not t.is_cuda else t) for n, t in data
                    )
                else:
                    n, t = data
                    all_named_tensors.append((n, t.cuda() if not t.is_cuda else t))

        if not all_named_tensors:
            return JSONResponse(content={"status": "ok", "loaded": 0})

        # ── Dispatch to IPC or safetensors path ──────────────────────────
        if self._weight_transfer_backend == "ipc":
            result = await self._load_weights_ipc(all_named_tensors)
        else:
            result = await self._load_weights_safetensors(all_named_tensors)

        if result is not None:
            return result  # error response

        # ── Optionally flush the KV cache ────────────────────────────────
        if flush_cache:
            try:
                await self._client.post(
                    f"{self.vllm_base_url}/sleep",
                    params={"level": "1", "mode": "abort"},
                    timeout=30.0,
                )
                await self._client.post(
                    f"{self.vllm_base_url}/wake_up",
                    params={"tags": "kv_cache"},
                    timeout=30.0,
                )
            except Exception as exc:
                logger.warning("flush_cache after weight update failed: %s", exc)

        # ── Bump weight version ──────────────────────────────────────────
        if weight_version is not None:
            self._weight_version = int(weight_version)
        else:
            self._weight_version += 1

        return JSONResponse(
            content={
                "status": "ok",
                "loaded": len(serialized_named_tensors),
                "weight_version": self._weight_version,
            }
        )

    # ------------------------------------------------------------------
    # IPC path – zero-copy GPU-to-GPU via vLLM's native IPC engine
    # ------------------------------------------------------------------

    async def _load_weights_ipc(
        self,
        named_tensors: list[tuple[str, "torch.Tensor"]],
    ) -> JSONResponse | None:
        """Send tensors to vLLM via CUDA IPC handles using vLLM's native
        ``IPCWeightTransferEngine``.  Returns *None* on success, or a
        ``JSONResponse`` on error."""
        import asyncio

        import torch

        from vllm.distributed.weight_transfer.ipc_engine import (
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )

        # IPC only supports single-GPU (TP=1).
        devices = {t.device for _, t in named_tensors}
        if len(devices) > 1:
            return JSONResponse(
                content={
                    "status": "error",
                    "detail": (
                        f"IPC weight transfer only supports TP=1, but tensors "
                        f"reside on {len(devices)} devices: {devices}. "
                        f"Use weight_transfer_backend='nccl' for TP > 1."
                    ),
                },
                status_code=400,
            )

        # trainer_send_weights uses torch.accelerator.current_device_index()
        # to obtain the GPU UUID — set the current device to match.
        tensor_device = named_tensors[0][1].device
        torch.cuda.set_device(tensor_device)

        trainer_args = IPCTrainerSendWeightsArgs(
            mode="http", url=self.vllm_base_url,
        )

        logger.info(
            "update_weights_from_tensor (IPC): sending %d params to vLLM at %s "
            "(device=%s)",
            len(named_tensors), self.vllm_base_url, tensor_device,
        )
        try:
            # trainer_send_weights is synchronous (uses requests internally),
            # so run in a thread to avoid blocking the async event loop.
            await asyncio.to_thread(
                IPCWeightTransferEngine.trainer_send_weights,
                iterator=iter(named_tensors),
                trainer_args=trainer_args,
            )
        except Exception as exc:
            logger.error("IPC weight update failed: %s", exc, exc_info=True)
            return JSONResponse(
                content={"status": "error", "detail": str(exc)},
                status_code=500,
            )
        return None

    # ------------------------------------------------------------------
    # Safetensors fallback – temp file via /load_weights
    # ------------------------------------------------------------------

    async def _load_weights_safetensors(
        self,
        named_tensors: list[tuple[str, "torch.Tensor"]],
    ) -> JSONResponse | None:
        """Save tensors to a temp safetensors file and load via vLLM
        ``POST /load_weights``.  Returns *None* on success, or a
        ``JSONResponse`` on error."""
        import os
        import tempfile

        try:
            from safetensors.torch import save_file as _st_save
        except ImportError:
            logger.error(
                "safetensors is required for the safetensors weight-load fallback. "
                "Install it with: pip install safetensors"
            )
            return JSONResponse(
                content={"status": "error", "detail": "safetensors not installed"},
                status_code=500,
            )

        tensor_dict = {name: t.contiguous().cpu() for name, t in named_tensors}

        # Prefer /dev/shm (RAM-backed tmpfs) to avoid real disk I/O.
        tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
        tmp_fd = tempfile.NamedTemporaryFile(
            prefix="slime_weights_", suffix=".safetensors",
            dir=tmp_dir, delete=False,
        )
        tmp_path = tmp_fd.name
        tmp_fd.close()

        try:
            _st_save(tensor_dict, tmp_path)

            resp = await self._client.post(
                f"{self.vllm_base_url}/load_weights",
                json={"files": [tmp_path]},
                timeout=300.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "vLLM /load_weights returned %s: %s",
                exc.response.status_code,
                exc.response.text[:2000],
            )
            return JSONResponse(
                content={"status": "error", "detail": str(exc)},
                status_code=502,
            )
        except Exception as exc:
            logger.error("Failed to load weights into vLLM: %s", exc, exc_info=True)
            return JSONResponse(
                content={"status": "error", "detail": str(exc)},
                status_code=500,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            del tensor_dict

        return None

    async def set_weight_version(self, request: Request):
        """Increment or set the weight version (called by VLLMEngine after weight update)."""
        body = await request.json()
        if "weight_version" in body:
            self._weight_version = int(body["weight_version"])
        else:
            self._weight_version += 1
        return JSONResponse(content={"weight_version": self._weight_version})


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def run_sidecar(
    vllm_host: str = "127.0.0.1",
    vllm_port: int = 8000,
    sidecar_host: str = "0.0.0.0",
    sidecar_port: int = 8100,
    model_name: str = "default",
    timeout: float = 600.0,
    max_connections: int = 256,
    log_level: str = "info",
    weight_transfer_backend: str = "nccl",
):
    """Launch the translation sidecar as a standalone uvicorn process."""

    vllm_base_url = f"http://{vllm_host}:{vllm_port}"
    sidecar = TranslationSidecar(
        vllm_base_url=vllm_base_url,
        model_name=model_name,
        timeout=timeout,
        max_connections=max_connections,
        weight_transfer_backend=weight_transfer_backend,
    )
    uvicorn.run(
        sidecar.app,
        host=sidecar_host,
        port=sidecar_port,
        log_level=log_level,
    )


def main():
    parser = argparse.ArgumentParser(description="vLLM Translation Sidecar")
    parser.add_argument("--vllm-host", type=str, default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--sidecar-host", type=str, default="0.0.0.0")
    parser.add_argument("--sidecar-port", type=int, default=8100)
    parser.add_argument("--model-name", type=str, default="default")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-connections", type=int, default=256)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--weight-transfer-backend", type=str, default="nccl",
                        choices=["nccl", "ipc"], help="vLLM weight transfer backend")
    args = parser.parse_args()

    run_sidecar(
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        sidecar_host=args.sidecar_host,
        sidecar_port=args.sidecar_port,
        model_name=args.model_name,
        timeout=args.timeout,
        max_connections=args.max_connections,
        log_level=args.log_level,
        weight_transfer_backend=args.weight_transfer_backend,
    )


if __name__ == "__main__":
    main()
