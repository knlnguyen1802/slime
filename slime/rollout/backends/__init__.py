from slime.rollout.backends.base_client import BackendCapabilities, RolloutBackendClient
from slime.rollout.backends.vllm_client import VLLMClient


def __getattr__(name):
    if name == "SGLangClient":
        from slime.rollout.backends.sglang_client import SGLangClient

        return SGLangClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackendCapabilities",
    "RolloutBackendClient",
    "SGLangClient",
    "VLLMClient",
]
