"""
Local reimplementations of sglang weight-sync utilities.

Used as fallback when sglang is not installed (e.g. vLLM-only mode).
The three classes/functions here are API-compatible with their sglang
counterparts so that the rest of the megatron weight-update code can
work unchanged.

Origin (sglang):
  - FlattenedTensorBucket  → sglang.srt.weight_sync.tensor_bucket
  - MultiprocessingSerializer / SafeUnpickler → sglang.srt.utils.common
  - monkey_patch_torch_reductions → sglang.srt.utils.patch_torch
"""

from __future__ import annotations

import base64
import io
import logging
import pickle
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Callable, Union

import torch
from torch.multiprocessing import reductions

logger = logging.getLogger(__name__)

# Tri-state flag for CUDA IPC availability:
#   None  – not yet probed
#   True  – CUDA IPC works (cudaMalloc / expandable_segments:False)
#   False – CUDA IPC unavailable (expandable_segments:True or similar)
_cuda_ipc_available: bool | None = None


def check_cuda_ipc_available() -> bool:
    """Proactively check whether CUDA IPC handles can be created.

    ``expandable_segments:True`` (the common cause) makes the CUDA caching
    allocator use ``cuMemCreate`` / ``cuMemMap`` instead of ``cudaMalloc``.
    Memory obtained that way is fundamentally incompatible with
    ``cudaIpcGetMemHandle``, which ``storage._share_cuda_()`` relies on.

    This function allocates a tiny 1-byte probe tensor, tries to create an
    IPC handle, and caches the result for the process lifetime.
    """
    global _cuda_ipc_available
    if _cuda_ipc_available is not None:
        return _cuda_ipc_available

    if not torch.cuda.is_available():
        _cuda_ipc_available = False
        return False

    try:
        probe = torch.zeros(1, device="cuda")
        probe.untyped_storage()._share_cuda_()
        _cuda_ipc_available = True
    except Exception:
        _cuda_ipc_available = False
        logger.warning(
            "CUDA IPC is unavailable (commonly caused by "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). "
            "Weight sync will transfer tensors via CPU instead of "
            "zero-copy GPU IPC."
        )
    return _cuda_ipc_available


# ── FlattenedTensorBucket ───────────────────────────────────────────


@dataclass
class FlattenedTensorMetadata:
    """Metadata for a tensor in a flattened bucket."""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """
    A bucket that flattens multiple tensors into a single uint8 tensor
    for efficient serialisation, while preserving all metadata needed
    for reconstruction.

    API-compatible with ``sglang.srt.weight_sync.tensor_bucket.FlattenedTensorBucket``.
    """

    # Checked by callers to decide whether to group tensors by dtype.
    supports_multi_dtypes = True

    def __init__(
        self,
        named_tensors: list[tuple[str, torch.Tensor]] | None = None,
        flattened_tensor: torch.Tensor | None = None,
        metadata: list[FlattenedTensorMetadata] | None = None,
    ):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")

            self.metadata: list[FlattenedTensorMetadata] = [None] * len(named_tensors)
            current_idx = 0
            flat_parts: list[torch.Tensor] = [None] * len(named_tensors)

            for i, (name, tensor) in enumerate(named_tensors):
                flat = tensor.flatten().view(torch.uint8)
                numel = flat.numel()
                flat_parts[i] = flat
                self.metadata[i] = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                current_idx += numel

            self.flattened_tensor: torch.Tensor = torch.cat(flat_parts, dim=0)
        else:
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "Must provide either named_tensors or both flattened_tensor and metadata"
                )
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        """Return the single flat uint8 tensor."""
        return self.flattened_tensor

    def get_metadata(self) -> list[FlattenedTensorMetadata]:
        """Return per-tensor metadata list."""
        return self.metadata

    def reconstruct_tensors(self) -> list[tuple[str, torch.Tensor]]:
        """Reconstruct the original named tensors from the flat representation."""
        reconstructed = [None] * len(self.metadata)
        for i, meta in enumerate(self.metadata):
            tensor = (
                self.flattened_tensor[meta.start_idx : meta.end_idx]
                .view(meta.dtype)
                .reshape(meta.shape)
            )
            reconstructed[i] = (meta.name, tensor)
        return reconstructed


# ── SafeUnpickler / MultiprocessingSerializer ───────────────────────


class SafeUnpickler(pickle.Unpickler):
    """
    Unpickler with an allow-list to prevent arbitrary code execution.

    API-compatible with the ``SafeUnpickler`` in ``sglang.srt.utils.common``.
    """

    ALLOWED_MODULE_PREFIXES = {
        # Python builtins
        "builtins.",
        "collections.",
        "copyreg.",
        "functools.",
        "itertools.",
        "operator.",
        "types.",
        "weakref.",
        # PyTorch
        "torch.",
        "torch._tensor.",
        "torch.storage.",
        "torch.nn.parameter.",
        "torch.autograd.function.",
        # torch.distributed
        "torch.distributed.",
        "torch.distributed._shard.",
        "torch.distributed._composable.",
        "torch._C._distributed_c10d.",
        "torch._C._distributed_fsdp.",
        "torch.distributed.optim.",
        # multiprocessing
        "multiprocessing.resource_sharer.",
        "multiprocessing.reduction.",
        "pickletools.",
        # HuggingFace / PEFT
        "peft.",
        "transformers.",
        "huggingface_hub.",
        # slime local reimplementation
        "slime.backends.megatron_utils.weight_sync_utils.",
        # sglang (if installed alongside)
        "sglang.srt.weight_sync.tensor_bucket.",
        "sglang.srt.model_executor.model_runner.",
        "sglang.srt.layers.",
        "sglang.srt.utils.",
        # NPU
        "torch_npu.",
    }

    DENY_CLASSES = {
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("os", "system"),
        ("subprocess", "Popen"),
        ("subprocess", "run"),
        ("codecs", "decode"),
        ("types", "CodeType"),
        ("types", "FunctionType"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self.DENY_CLASSES:
            raise RuntimeError(
                f"Blocked unsafe class loading ({module}.{name}), "
                f"to prevent exploitation of CVE-2025-10164"
            )
        if any((module + ".").startswith(prefix) for prefix in self.ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)
        raise RuntimeError(
            f"Blocked unsafe class loading ({module}.{name}), "
            f"to prevent exploitation of CVE-2025-10164"
        )


class MultiprocessingSerializer:
    """
    Serialize / deserialize Python objects via ``ForkingPickler`` so that
    CUDA tensors are transferred through shared memory (IPC handles).

    API-compatible with ``sglang.srt.utils.common.MultiprocessingSerializer``.

    Uses stdlib ``base64`` instead of ``pybase64`` to avoid adding a dependency.

    When CUDA IPC is unavailable (e.g. ``expandable_segments:True`` makes the
    allocator use ``cuMemCreate``/``cuMemMap`` instead of ``cudaMalloc``),
    tensors are proactively moved to CPU before serialization so that the
    transfer falls back to embedded-data pickle instead of crashing.
    """

    @staticmethod
    def serialize(obj, output_str: bool = False):
        buf = io.BytesIO()
        if check_cuda_ipc_available():
            # Normal path: ForkingPickler creates CUDA IPC handles for GPU
            # tensors, giving zero-copy weight sharing with the engine.
            ForkingPickler(buf).dump(obj)
        else:
            # Fallback: move CUDA tensors to CPU and use plain pickle which
            # embeds the tensor bytes directly (no IPC handles, no shared-
            # memory file descriptors that could become stale).
            obj = _cuda_tensors_to_cpu(obj)
            pickle.Pickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()
        if output_str:
            output = base64.b64encode(output).decode("utf-8")
        return output

    @staticmethod
    def deserialize(data):
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)
        return SafeUnpickler(io.BytesIO(data)).load()


def _cuda_tensors_to_cpu(obj):
    """Recursively move CUDA tensors inside *obj* to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.is_cuda else obj
    if isinstance(obj, dict):
        return {k: _cuda_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_cuda_tensors_to_cpu(v) for v in obj]
        return type(obj)(converted)
    return obj


# ── monkey_patch_torch_reductions ───────────────────────────────────

_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device_maybe_uuid: Union[int, str]) -> int:
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid
    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise RuntimeError("Invalid device_uuid=" + device_maybe_uuid)
    raise RuntimeError(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier: Callable):
    return (*t[:index], modifier(t[index]), *t[index + 1 :])


def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid)
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    # Ensure _rebuild_cuda_tensor_original exists even in processes where
    # monkey_patch_torch_reductions() was never called (e.g. vLLM server
    # subprocess).  This mirrors the fixup already present in
    # update_weight_from_distributed.py for NcclBridge subprocesses.
    if not hasattr(reductions, "_rebuild_cuda_tensor_original"):
        _rebuild_cuda = getattr(reductions, "_rebuild_cuda_tensor", None) or getattr(
            reductions, "rebuild_cuda_tensor", None
        )
        if _rebuild_cuda is not None and _rebuild_cuda is not _rebuild_cuda_tensor_modified:
            reductions._rebuild_cuda_tensor_original = _rebuild_cuda
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def monkey_patch_torch_reductions():
    """
    Monkey-patch ``torch.multiprocessing.reductions`` so that CUDA tensors
    are identified by device UUID rather than ordinal index.

    This works around https://github.com/pytorch/pytorch/pull/149248.

    API-compatible with ``sglang.srt.utils.patch_torch.monkey_patch_torch_reductions``.
    """
    if hasattr(reductions, "_reduce_tensor_original"):
        return  # already patched
    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified
    reductions.init_reductions()
