"""VLLMEngine: Ray actor that launches and manages a vLLM server + rollout adapter."""

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
        self.adapter_port = None
        self.process = None
        self.adapter_process = None
        self._log_file = None
        self._adapter_log_file = None
        self._weight_version: int = 0

    @property
    def adapter_url(self) -> str:
        """URL of the vLLM rollout adapter (registered with the router)."""
        return f"http://{self.server_host}:{self.adapter_port}"

    @property
    def vllm_url(self) -> str:
        """URL of the raw vLLM server."""
        return f"http://{self.server_host}:{self.server_port}"

    def init(self, port=None, host=None, router_ip=None, router_port=None, **kwargs):
        self.server_host = host or get_host_info()[1]
        self.server_port = port or get_free_port(15000)
        self.adapter_port = get_free_port(self.server_port + 100)
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
            "--weight-transfer-config", '{"backend": "nccl"}',
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

        # Launch the rollout adapter
        self._launch_adapter()
        self._wait_adapter_healthy()

        # Register the adapter URL with the router
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

    def _launch_adapter(self):
        """Launch the vLLM rollout adapter as a subprocess."""
        from slime.backends.vllm_utils.vllm_rollout_adapter import run_vllm_rollout_adapter

        self._adapter_log_file = tempfile.NamedTemporaryFile(
            prefix="vllm_adapter_", suffix=".log", delete=False, mode="w"
        )

        def _target():
            run_vllm_rollout_adapter(
                vllm_host="127.0.0.1",
                vllm_port=self.server_port,
                adapter_host="0.0.0.0",
                adapter_port=self.adapter_port,
                model_name=self._model_name,
                log_level="info",
            )

        self.adapter_process = multiprocessing.Process(target=_target, daemon=True)
        self.adapter_process.start()
        logger.info(
            "Launched vLLM rollout adapter on port %s (vLLM → %s:%s), log=%s",
            self.adapter_port,
            self.server_host,
            self.server_port,
            self._adapter_log_file.name,
        )

    def _wait_adapter_healthy(self, timeout: float = 60.0):
        """Block until the adapter /health endpoint responds 200."""
        url = f"{self.adapter_url}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    logger.info("vLLM rollout adapter healthy at %s", self.adapter_url)
                    return
            except Exception:
                pass
            if self.adapter_process and not self.adapter_process.is_alive():
                raise RuntimeError(
                    f"Adapter process exited with code {self.adapter_process.exitcode}"
                )
            time.sleep(1)
        raise TimeoutError(f"Adapter failed to become healthy within {timeout}s")

    def _register_with_router(self):
        """Register the adapter URL with the SlimeRouter."""
        router_url = f"http://{self.router_ip}:{self.router_port}"
        response = requests.post(
            f"{router_url}/add_worker",
            params={"url": self.adapter_url},
        )
        response.raise_for_status()
        logger.info(
            "Registered adapter %s with router at %s",
            self.adapter_url,
            router_url,
        )

    def _bump_weight_version(self, version: int | None = None):
        """Notify the adapter to increment (or set) its weight version counter."""
        url = f"{self.adapter_url}/set_weight_version"
        payload = {"weight_version": version} if version is not None else {}
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            self._weight_version = r.json().get("weight_version", self._weight_version)
        except Exception as exc:
            logger.warning("Failed to bump adapter weight version: %s", exc)

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
        # Notify the adapter about the new weight version
        self._bump_weight_version(weight_version)

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
        if self.adapter_port:
            try:
                r = requests.get(f"{self.adapter_url}/get_weight_version", timeout=5)
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
        # Shutdown rollout adapter first
        if self.adapter_process and self.adapter_process.is_alive():
            self.adapter_process.terminate()
            self.adapter_process.join(timeout=10)
            if self.adapter_process.is_alive():
                self.adapter_process.kill()
            self.adapter_process = None
        if self._adapter_log_file:
            try:
                self._adapter_log_file.close()
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
