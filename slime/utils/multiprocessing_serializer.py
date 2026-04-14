"""
Standalone weight-sync utilities for Slime.

Provides ``MultiprocessingSerializer``, ``FlattenedTensorBucket``, and
``monkey_patch_torch_reductions`` without requiring sglang to be installed.

These are API-compatible with their sglang counterparts so that the rest of
the weight-update code can work with *any* rollout backend (SGLang, vLLM, …).

Origin (sglang):
  - FlattenedTensorBucket  → sglang.srt.weight_sync.tensor_bucket
  - MultiprocessingSerializer / SafeUnpickler → sglang.srt.utils.common
  - monkey_patch_torch_reductions → sglang.srt.utils.patch_torch
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Callable, Union

import torch
from torch.multiprocessing import reductions

# Use pybase64 when available (faster SIMD implementation), fall back to stdlib.
try:
    import pybase64 as _b64mod

    def _b64encode(data: bytes) -> bytes:
        return _b64mod.b64encode(data)

    def _b64decode(data: str | bytes) -> bytes:
        return _b64mod.b64decode(data, validate=True)

except ImportError:
    import base64 as _b64mod  # type: ignore[no-redef]

    def _b64encode(data: bytes) -> bytes:  # type: ignore[misc]
        return _b64mod.b64encode(data)

    def _b64decode(data: str | bytes) -> bytes:  # type: ignore[misc]
        return _b64mod.b64decode(data, validate=True)


# ── FlattenedTensorBucket ───────────────────────────────────────────


@dataclass
class FlattenedTensorMetadata:
    """Metadata for a tensor stored inside a :class:`FlattenedTensorBucket`."""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """Flatten multiple tensors into a single ``uint8`` tensor for efficient
    serialisation, while preserving all metadata needed for reconstruction.

    API-compatible with
    ``sglang.srt.weight_sync.tensor_bucket.FlattenedTensorBucket``.
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
        """Return the single flat ``uint8`` tensor."""
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


# ── SafeUnpickler ───────────────────────────────────────────────────


class SafeUnpickler(pickle.Unpickler):
    """Unpickler with an allow-list to prevent arbitrary code execution.

    API-compatible with the ``SafeUnpickler`` in ``sglang.srt.utils.common``.
    """

    ALLOWED_MODULE_PREFIXES = {
        # Python builtins (specific safe classes only – see ALLOW_CLASSES)
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
        "slime.utils.multiprocessing_serializer.",
        "slime.backends.megatron_utils.weight_sync_utils.",
        # sglang (if installed alongside)
        "sglang.srt.weight_sync.tensor_bucket.",
        "sglang.srt.model_executor.model_runner.",
        "sglang.srt.layers.",
        "sglang.srt.utils.",
        # NPU
        "torch_npu.",
    }

    ALLOW_CLASSES = {
        ("builtins", "True"),
        ("builtins", "False"),
        ("builtins", "None"),
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "complex"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "bytearray"),
        ("builtins", "bool"),
        ("builtins", "slice"),
        ("builtins", "range"),
        ("builtins", "enumerate"),
        ("builtins", "map"),
        ("builtins", "zip"),
        ("builtins", "filter"),
        ("builtins", "reversed"),
        ("builtins", "sorted"),
    }

    DENY_CLASSES = {
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("builtins", "getattr"),
        ("builtins", "setattr"),
        ("builtins", "delattr"),
        ("builtins", "__import__"),
        ("builtins", "globals"),
        ("builtins", "locals"),
        ("builtins", "open"),
        ("builtins", "breakpoint"),
        ("builtins", "input"),
        ("builtins", "memoryview"),
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
        if module == "builtins":
            if (module, name) in self.ALLOW_CLASSES:
                return super().find_class(module, name)
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


# ── MultiprocessingSerializer ──────────────────────────────────────


class MultiprocessingSerializer:
    """Serialize and deserialize Python objects (incl. GPU tensors)
    using PyTorch's ``ForkingPickler``, which preserves CUDA shared handles.

    For CUDA tensors only a small IPC handle is serialized – the actual GPU
    memory is shared across processes on the same node via ``cudaIpcOpenMemHandle``.

    API-compatible with ``sglang.srt.utils.common.MultiprocessingSerializer``.
    """

    @staticmethod
    def serialize(obj, output_str: bool = False, inline_cpu: bool = False):
        """Serialize *obj* (e.g. a dict containing CUDA tensors).

        When *output_str* is ``True`` the result is a base64-encoded UTF-8
        string suitable for JSON transport.

        When *inline_cpu* is ``True``, regular ``pickle.Pickler`` is used
        instead of ``ForkingPickler``.  This stores CPU tensor data inline
        as raw bytes in the pickle stream, avoiding the fd-sharing path
        (``rebuild_storage_fd`` / ``resource_sharer``) that requires a
        matching ``authkey`` and therefore fails across different Ray actor
        processes.
        """
        buf = io.BytesIO()
        if inline_cpu:
            pickle.Pickler(buf).dump(obj)
        else:
            ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()
        if output_str:
            output = _b64encode(output).decode("utf-8")
        return output

    @staticmethod
    def deserialize(data):
        """Deserialize (rebuild) a previously serialized object.

        Automatically re-opens CUDA IPC handles when the object was a GPU
        tensor.
        """
        if isinstance(data, str):
            data = _b64decode(data)
        return SafeUnpickler(io.BytesIO(data)).load()


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
    # Only replace the device ordinal with a UUID for CUDA tensors.
    # CPU tensors (e.g. moved to CPU for torch_memory_saver VMM compat)
    # have a shorter args tuple with no device index at position 6.
    tensor = args[0] if args else None
    if tensor is not None and tensor.is_cuda:
        output_args = _modify_tuple(output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid)
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def monkey_patch_torch_reductions():
    """Monkey-patch ``torch.multiprocessing.reductions`` so that CUDA tensors
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
