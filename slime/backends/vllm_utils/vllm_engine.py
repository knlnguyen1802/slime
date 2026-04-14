"""VLLMEngine: Ray actor that launches and manages a vLLM server + translation sidecar."""

import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
import time

import requests

from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info
from slime.utils.misc import get_free_port

logger = logging.getLogger(__name__)


class VLLMEngine(RayActor):
    """Ray actor that runs vLLM server with same interface as SGLangEngine for weight sync."""

    def __init__(self, args, rank: int, base_gpu_id: int | None = None, gpu_ids: list[int] | None = None, **kwargs):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id or 0
        self.gpu_ids = gpu_ids or [self.base_gpu_id]
        self.server_host = None
        self.server_port = None
        self.sidecar_port = None
        self.process = None
        self.sidecar_process = None
        self._log_file = None
        self._sidecar_log_file = None
        self._weight_version: int = 0
        self._weight_transfer_backend: str = kwargs.get(
            "weight_transfer_backend",
            getattr(args, "vllm_weight_transfer_backend", "nccl"),
        )

    @property
    def sidecar_url(self) -> str:
        """URL of the translation sidecar (registered with the router)."""
        return f"http://{self.server_host}:{self.sidecar_port}"

    @property
    def vllm_url(self) -> str:
        """URL of the raw vLLM server."""
        return f"http://{self.server_host}:{self.server_port}"

    def init(self, port=None, host=None, router_ip=None, router_port=None, **kwargs):
        self.server_host = host or get_host_info()[1]
        self.server_port = port or get_free_port(15000)
        self.sidecar_port = get_free_port(self.server_port + 100)
        self.router_ip = router_ip or getattr(self.args, "sglang_router_ip", None)
        self.router_port = router_port or getattr(self.args, "sglang_router_port", None)

        model = getattr(self.args, "vllm_model", None) or self.args.hf_checkpoint
        self._model_name = model
        tp = self.args.rollout_num_gpus_per_engine
        gpu_ids = self.gpu_ids[:tp]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            visible = [int(x) for x in cvd.split(",") if x.strip()]
            dev_str = ",".join(str(visible[gid]) if gid < len(visible) else str(gid) for gid in gpu_ids)
        else:
            dev_str = ",".join(str(g) for g in gpu_ids)

        seed = getattr(self.args, "seed", 1234) + self.rank
        cmd = [
            "vllm", "serve", model,
            "--tensor-parallel-size", str(tp),
            "--port", str(self.server_port),
            "--host", "0.0.0.0",
            "--weight-transfer-config", json.dumps({"backend": self._weight_transfer_backend}),
            "--seed", str(seed),
            "--trust-remote-code",
        ]
        gpu_mem_util = getattr(self.args, "vllm_gpu_memory_utilization", 0.4)
        cmd.extend(["--gpu-memory-utilization", str(gpu_mem_util)])
        if getattr(self.args, "offload_rollout", False):
            cmd.append("--enable-sleep-mode")
        if getattr(self.args, "vllm_enforce_eager", True):
            cmd.append("--enforce-eager")
        if getattr(self.args, "fp16", False):
            cmd.extend(["--dtype", "float16"])

        env = os.environ.copy()
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = dev_str
        env.setdefault("NCCL_DEBUG", "INFO")
        env.setdefault("NCCL_DEBUG_SUBSYS", "ALL")
        env["NCCL_P2P_DISABLE"] = "1"
        env.setdefault("NCCL_IB_DISABLE", "1")
        if self._weight_transfer_backend == "ipc":
            env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        self._log_file = tempfile.NamedTemporaryFile(
            prefix="vllm_engine_", suffix=".log", delete=False, mode="w"
        )
        logger.info("Launching vLLM: cmd=%s, CUDA_VISIBLE_DEVICES=%s, log=%s",
                     " ".join(cmd), dev_str, self._log_file.name)
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self._wait_healthy()

        # Launch the translation sidecar
        self._launch_sidecar()
        self._wait_sidecar_healthy()

        # Register the sidecar URL with the router
        if self.router_ip and self.router_port:
            self._register_with_router()

    def _wait_healthy(self, timeout=300):
        base = f"http://{self.server_host}:{self.server_port}"
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{base}/health", timeout=5)
                if r.status_code == 200:
                    logger.info("vLLM server healthy at %s:%s", self.server_host, self.server_port)
                    return
            except Exception:
                pass
            if self.process and self.process.poll() is not None:
                log_tail = self._read_log_tail()
                raise RuntimeError(f"vLLM process exited with code {self.process.returncode}.\n{log_tail}")
            time.sleep(2)
        log_tail = self._read_log_tail()
        raise TimeoutError(f"vLLM server failed to become healthy within {timeout}s.\n{log_tail}")

    def _launch_sidecar(self):
        """Launch the translation sidecar as a subprocess."""
        from slime.backends.vllm_utils.vllm_translation_sidecar import run_sidecar

        self._sidecar_log_file = tempfile.NamedTemporaryFile(
            prefix="vllm_sidecar_", suffix=".log", delete=False, mode="w"
        )

        def _target():
            run_sidecar(
                vllm_host="127.0.0.1",
                vllm_port=self.server_port,
                sidecar_host="0.0.0.0",
                sidecar_port=self.sidecar_port,
                model_name=self._model_name,
                log_level="info",
                weight_transfer_backend=self._weight_transfer_backend,
            )

        self.sidecar_process = multiprocessing.Process(target=_target, daemon=True)
        self.sidecar_process.start()
        logger.info(
            "Launched translation sidecar on port %s (vLLM → %s:%s), log=%s",
            self.sidecar_port,
            self.server_host,
            self.server_port,
            self._sidecar_log_file.name,
        )

    def _wait_sidecar_healthy(self, timeout: float = 60.0):
        """Block until the sidecar /health endpoint responds 200."""
        url = f"{self.sidecar_url}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    logger.info("Translation sidecar healthy at %s", self.sidecar_url)
                    return
            except Exception:
                pass
            if self.sidecar_process and not self.sidecar_process.is_alive():
                raise RuntimeError(
                    f"Sidecar process exited with code {self.sidecar_process.exitcode}"
                )
            time.sleep(1)
        raise TimeoutError(f"Sidecar failed to become healthy within {timeout}s")

    def _register_with_router(self):
        """Register the sidecar URL with the SlimeRouter."""
        router_url = f"http://{self.router_ip}:{self.router_port}"
        response = requests.post(
            f"{router_url}/add_worker",
            params={"url": self.sidecar_url},
        )
        response.raise_for_status()
        logger.info(
            "Registered sidecar %s with router at %s",
            self.sidecar_url,
            router_url,
        )

    def _bump_weight_version(self, version: int | None = None):
        """Notify the sidecar to increment (or set) its weight version counter."""
        url = f"{self.sidecar_url}/set_weight_version"
        payload = {"weight_version": version} if version is not None else {}
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            self._weight_version = r.json().get("weight_version", self._weight_version)
        except Exception as exc:
            logger.warning("Failed to bump sidecar weight version: %s", exc)

    def _read_log_tail(self, n=200):
        if not self._log_file:
            return ""
        try:
            self._log_file.flush()
            with open(self._log_file.name) as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return ""

    def _post(self, path: str, json_data: dict | None = None, params: dict | None = None):
        url = f"http://{self.server_host}:{self.server_port}{path}"
        kwargs = {"timeout": 120}
        if json_data is not None:
            kwargs["json"] = json_data
        if params is not None:
            kwargs["params"] = params
        r = requests.post(url, **kwargs)
        if not r.ok:
            body = r.text[:2000] if r.text else "(empty)"
            log_tail = self._read_log_tail(50)
            logger.error(
                "vLLM %s returned %s: %s\n--- vLLM log tail ---\n%s",
                path, r.status_code, body, log_tail,
            )
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return None

    def health_generate(self, timeout: float = 5.0) -> bool:
        try:
            r = requests.get(
                f"http://{self.server_host}:{self.server_port}/health",
                timeout=timeout,
            )
            r.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def pause_generation(self):
        self._post("/pause", params={"mode": "abort"})

    def flush_cache(self):
        pass

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name=None, backend=None):
        if self._weight_transfer_backend == "ipc":
            # IPC backend needs no initialisation; call with empty info for the no-op.
            logger.info("IPC weight transfer backend – skipping NCCL group init.")
            self._post("/init_weight_transfer_engine", json_data={"init_info": {}})
            return

        logger.info(
            "Initializing NCCL weight transfer: master=%s:%s, rank_offset=%d, "
            "world_size=%d, vllm_url=http://%s:%s, vllm_log=%s",
            master_address, master_port, rank_offset, world_size,
            self.server_host, self.server_port,
            self._log_file.name if self._log_file else "<none>",
        )
        self._post("/init_weight_transfer_engine", json_data={
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
            }
        })
        log_tail = self._read_log_tail(30)
        logger.info("vLLM log after init_weight_transfer_engine:\n%s", log_tail)

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name=None,
        flush_cache=False,
        weight_version=None,
        packed: bool = True,
    ):
        dtype_names = [str(d).replace("torch.", "") for d in dtypes]
        shape_lists = [list(s) for s in shapes]
        self._post("/update_weights", json_data={
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shape_lists,
                "packed": packed,
            }
        })
        # Notify the sidecar about the new weight version
        self._bump_weight_version(weight_version)

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """Update model weights from tensor data via CUDA IPC handles.

        Deserializes the ``FlattenedTensorBucket`` payloads (produced by
        :class:`MultiprocessingSerializer`), re-creates CUDA IPC handles via
        ``reduce_tensor``, and sends them to vLLM's native
        ``POST /update_weights`` endpoint with the ``ipc`` backend.

        When the weight-transfer backend is not ``ipc`` (e.g. ``nccl``), falls
        back to the translation-sidecar safetensors path.

        This mirrors the ``SGLangEngine.update_weights_from_tensor`` interface
        so that the FSDP / Megatron weight-sync code works with either backend.
        """
        if self._weight_transfer_backend == "ipc":
            return self._update_weights_ipc(
                serialized_named_tensors, load_format, flush_cache, weight_version,
            )
        return self._update_weights_via_sidecar(
            serialized_named_tensors, load_format, flush_cache, weight_version,
        )

    # ------------------------------------------------------------------
    # IPC path – zero-copy, GPU-to-GPU via vLLM's native IPC engine
    # ------------------------------------------------------------------

    def _update_weights_ipc(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None,
        flush_cache: bool,
        weight_version: str | None,
    ):
        import torch

        from slime.utils.multiprocessing_serializer import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
            monkey_patch_torch_reductions,
        )

        monkey_patch_torch_reductions()

        # ── Deserialize all tensor payloads ──────────────────────────────
        all_named_tensors: list[tuple[str, torch.Tensor]] = []
        for serialized in serialized_named_tensors:
            data = MultiprocessingSerializer.deserialize(serialized)
            if load_format == "flattened_bucket":
                bucket = FlattenedTensorBucket(
                    flattened_tensor=data["flattened_tensor"],
                    metadata=data["metadata"],
                )
                all_named_tensors.extend(bucket.reconstruct_tensors())
            else:
                if isinstance(data, list):
                    all_named_tensors.extend(data)
                else:
                    all_named_tensors.append(data)

        if not all_named_tensors:
            return

        # ── Validate: IPC only supports TP=1 (single GPU) ───────────────
        # trainer_send_weights assigns one GPU UUID to all tensors.
        # With TP > 1 the gathered serialized data spans multiple GPUs,
        # so the single-UUID assumption breaks.  Use NCCL for multi-GPU.
        devices = {t.device for _, t in all_named_tensors}
        if len(devices) > 1:
            raise RuntimeError(
                f"IPC weight transfer only supports TP=1 (single GPU), but "
                f"deserialized tensors reside on {len(devices)} devices: "
                f"{devices}.  Use weight_transfer_backend='nccl' for TP > 1."
            )

        # ── Set CUDA device to match the deserialized tensors ────────────
        # trainer_send_weights calls torch.accelerator.current_device_index()
        # internally to obtain the GPU UUID.  We must ensure the current
        # device matches the GPU the tensors actually live on.
        tensor_device = all_named_tensors[0][1].device
        torch.cuda.set_device(tensor_device)

        # ── Send weights via vLLM's native IPC engine ────────────────────
        # Uses IPCWeightTransferEngine.trainer_send_weights() which handles
        # CUDA IPC handle creation, serialization, and HTTP transport
        # internally — no manual reduce_tensor / pickle / base64 needed.
        from vllm.distributed.weight_transfer.ipc_engine import (
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )

        vllm_url = f"http://{self.server_host}:{self.server_port}"
        trainer_args = IPCTrainerSendWeightsArgs(mode="http", url=vllm_url)

        logger.info(
            "update_weights_from_tensor (IPC): sending %d params to vLLM at %s "
            "(device=%s)",
            len(all_named_tensors), vllm_url, tensor_device,
        )
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=iter(all_named_tensors),
            trainer_args=trainer_args,
        )

        self._bump_weight_version(weight_version)

        if flush_cache:
            self.flush_cache()

    # ------------------------------------------------------------------
    # Sidecar fallback – safetensors via translation sidecar
    # ------------------------------------------------------------------

    def _update_weights_via_sidecar(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None,
        flush_cache: bool,
        weight_version: str | None,
    ):
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version

        url = f"{self.sidecar_url}/update_weights_from_tensor"
        response = requests.post(url, json=payload, timeout=300)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            body = response.text[:2000] if response.text else "(empty)"
            logger.error(
                "sidecar /update_weights_from_tensor returned %s: %s",
                response.status_code, body,
            )
            raise
        return response.json()

    def continue_generation(self):
        self._post("/resume")

    def destroy_weights_update_group(self, group_name):
        pass

    def release_memory_occupation(self):
        try:
            self._post("/sleep", params={"level": "1", "mode": "abort"})
        except requests.RequestException as e:
            logger.warning("vLLM sleep failed (need --enable-sleep-mode?): %s", e)

    def resume_memory_occupation(self, tags: list[str] | None = None):
        try:
            params = {}
            if tags:
                params["tags"] = tags
            self._post("/wake_up", params=params)
        except requests.RequestException as e:
            logger.warning("vLLM wake_up failed: %s", e)

    def get_weight_version(self):
        if self.sidecar_port:
            try:
                r = requests.get(f"{self.sidecar_url}/get_weight_version", timeout=5)
                r.raise_for_status()
                return r.json().get("weight_version", self._weight_version)
            except Exception:
                pass
        return self._weight_version

    def check_weights(self, action: str):
        pass

    def post_process_weights(self, **kwargs):
        pass

    def shutdown(self):
        # Shutdown translation sidecar first
        if self.sidecar_process and self.sidecar_process.is_alive():
            self.sidecar_process.terminate()
            self.sidecar_process.join(timeout=10)
            if self.sidecar_process.is_alive():
                self.sidecar_process.kill()
            self.sidecar_process = None
        if self._sidecar_log_file:
            try:
                self._sidecar_log_file.close()
            except Exception:
                pass

        # Shutdown vLLM server
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
