import logging

from slime.rollout.backends.base_client import BackendCapabilities, RolloutBackendClient
from slime.rollout.base_types import RolloutBackendRequest, RolloutBackendResponse
from slime.utils.http_utils import get, post

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "abort": "abort",
    "end_turn": "stop",
    "max_tokens": "length",
}


class VLLMClient(RolloutBackendClient):
    """Rollout backend client for vLLM.

    Supports two modes:

    1. **Direct mode** (default): sends requests directly to vLLM's
       ``/v1/completions`` endpoint and parses the native vLLM response.
    2. **Router mode** (``use_slime_router=True``): sends SGLang-format
       requests to ``/generate`` through the SlimeRouter, which forwards
       to the vLLM rollout adapter. The adapter handles the vLLM translation
       and returns SGLang-format responses.
    """

    def __init__(self, args):
        self.args = args
        self._max_retries = getattr(args, "vllm_max_retries", 3)
        self._use_router = getattr(args, "use_slime_router", False)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_abort=self._use_router,  # abort is supported through the adapter
            supports_routed_experts=False,
            supports_prompt_logprobs=False,
        )

    async def generate(
        self,
        request: RolloutBackendRequest,
        base_url: str,
        headers: dict | None = None,
    ) -> RolloutBackendResponse:
        if self._use_router:
            return await self._generate_via_router(request, base_url, headers)
        return await self._generate_direct(request, base_url, headers)

    # ------------------------------------------------------------------
    # Router mode: SGLang-format /generate → adapter → vLLM
    # ------------------------------------------------------------------

    async def _generate_via_router(
        self,
        request: RolloutBackendRequest,
        base_url: str,
        headers: dict | None = None,
    ) -> RolloutBackendResponse:
        """Send SGLang-format request through the SlimeRouter → adapter pipeline."""

        payload = {
            "input_ids": request.input_ids,
            "sampling_params": request.sampling_params,
            "return_logprob": request.return_logprob,
            "stream": False,
        }
        if request.image_data:
            payload["image_data"] = request.image_data

        url = f"{base_url.rstrip('/')}/generate"
        output = await post(url, payload, headers=headers)

        # Parse SGLang-format response (produced by vLLM rollout adapter)
        meta = output.get("meta_info", {})
        logprobs_data = meta.get("output_token_logprobs", [])

        # output_token_logprobs is list of [logprob, token_id]
        output_token_ids = [item[1] for item in logprobs_data] if logprobs_data else []
        output_token_logprobs = [item[0] for item in logprobs_data] if logprobs_data else []

        # Fall back to output_ids if logprobs not available
        if not output_token_ids:
            output_token_ids = output.get("output_ids") or []

        finish_reason = meta.get("finish_reason", {}).get("type", "stop")

        return RolloutBackendResponse(
            text=output.get("text", ""),
            output_token_ids=output_token_ids,
            output_token_logprobs=output_token_logprobs,
            finish_reason=finish_reason,
            prompt_tokens=meta.get("prompt_tokens", len(request.input_ids)),
            completion_tokens=len(output_token_ids),
            backend_raw=output,
            routed_experts=None,
        )

    # ------------------------------------------------------------------
    # Direct mode: vLLM /v1/completions (no router)
    # ------------------------------------------------------------------

    async def _generate_direct(
        self,
        request: RolloutBackendRequest,
        base_url: str,
        headers: dict | None = None,
    ) -> RolloutBackendResponse:
        sp = request.sampling_params
        payload = {
            "prompt": request.input_ids,
            "max_tokens": sp.get("max_new_tokens", 1024),
            "temperature": sp.get("temperature", 1.0),
            "top_p": sp.get("top_p", 1.0),
            "top_k": sp.get("top_k", -1),
            "stop": sp.get("stop"),
            "stop_token_ids": sp.get("stop_token_ids"),
            "skip_special_tokens": sp.get("skip_special_tokens", True),
            "spaces_between_special_tokens": sp.get("spaces_between_special_tokens", False),
            "logprobs": 1 if request.return_logprob else None,
            "include_stop_str_in_output": sp.get("no_stop_trim", False),
            "return_token_ids": True,
            "seed": sp.get("sampling_seed"),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        url = f"{base_url.rstrip('/')}/v1/completions"
        output = await post(url, payload, headers=headers)

        choice = output.get("choices", [{}])[0]
        text = choice.get("text", "")
        finish = choice.get("finish_reason", "stop")
        finish_reason = _FINISH_REASON_MAP.get(finish, "stop")

        # vLLM /v1/completions response format:
        #   choice.token_ids: list[int]  (output token IDs)
        #   choice.logprobs.token_logprobs: list[float|None]
        #   choice.logprobs.tokens: list[str]  (token text, not IDs)
        output_token_ids = choice.get("token_ids") or []
        logprobs_obj = choice.get("logprobs") or {}
        raw_logprobs = logprobs_obj.get("token_logprobs") or []
        output_token_logprobs = [float(lp) if lp is not None else 0.0 for lp in raw_logprobs]

        # If token_ids not in response, fall back to tokenizer
        if not output_token_ids and text:
            logger.warning("vLLM response missing token_ids, falling back to tokenizer")
            from slime.utils.processing_utils import load_tokenizer
            tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
            output_token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Ensure logprobs list matches token count
        if len(output_token_logprobs) < len(output_token_ids):
            output_token_logprobs.extend([0.0] * (len(output_token_ids) - len(output_token_logprobs)))

        usage = output.get("usage", {})
        return RolloutBackendResponse(
            text=text,
            output_token_ids=output_token_ids,
            output_token_logprobs=output_token_logprobs,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", len(request.input_ids)),
            completion_tokens=usage.get("completion_tokens", len(output_token_ids)),
            backend_raw=output,
            routed_experts=None,
        )

    # ------------------------------------------------------------------
    # Abort (router mode only)
    # ------------------------------------------------------------------

    async def abort(self) -> list[str]:
        """Return worker URLs for abort. Only works in router mode."""
        if not self._use_router:
            return []
        base = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        r = await get(f"{base}/list_workers")
        return r.get("urls", [])
