# the file to manage all sglang deps in the megatron actor
# When sglang is installed we prefer its implementations; otherwise we
# fall back to API-compatible reimplementations in
# slime.backends.megatron_utils.weight_sync_utils.

# ── FP8 quantisation helpers (sglang-only, no local fallback) ───────
try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None

# ── monkey_patch_torch_reductions ───────────────────────────────────
try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    try:
        from sglang.srt.patch_torch import monkey_patch_torch_reductions
    except ImportError:
        from .weight_sync_utils import monkey_patch_torch_reductions

# ── MultiprocessingSerializer ───────────────────────────────────────
try:
    from sglang.srt.utils import MultiprocessingSerializer
except ImportError:
    from .weight_sync_utils import MultiprocessingSerializer

# ── FlattenedTensorBucket ───────────────────────────────────────────
try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    try:
        from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]
    except ImportError:
        from .weight_sync_utils import FlattenedTensorBucket

__all__ = [
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
]
