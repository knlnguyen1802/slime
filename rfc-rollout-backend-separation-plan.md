# RFC: Rollout Separation Plan (EngineGroup Generalization + Executor Cleanup)

## 1. Summary

This RFC proposes backend separation with minimal churn in runtime orchestration.

Scope is four items:

1. Refactor [slime/ray/rollout.py](slime/ray/rollout.py) by generalizing `EngineGroup.start_engines()` and abstracting engine/server creation (no new runtime manager class hierarchy).
2. Refactor [slime/rollout/sglang_rollout.py](slime/rollout/sglang_rollout.py) in-place: extract the one SGLang-specific code path (RadixTree) into a strategy hook, rename SGLang-prefixed args to generic names. No class hierarchy, no new files.
3. Refactor [slime/utils/arguments.py](slime/utils/arguments.py) into shared args + backend arg groups/finalizers.
4. Decouple [slime/backends/fsdp_utils/update_weight_utils.py](slime/backends/fsdp_utils/update_weight_utils.py) from SGLang internals so FSDP weight sync works with both SGLang and vLLM engines.

## 2. Already Done (Reuse, Do Not Rewrite)

- Unified rollout contracts in [slime/rollout/base_types.py](slime/rollout/base_types.py).
- Backend client abstraction in [slime/rollout/backends/base_client.py](slime/rollout/backends/base_client.py).
- Backend adapters in [slime/rollout/backends/sglang_client.py](slime/rollout/backends/sglang_client.py) and [slime/rollout/backends/vllm_client.py](slime/rollout/backends/vllm_client.py).
- Managed vLLM engine actor in [slime/backends/vllm_utils/vllm_engine.py](slime/backends/vllm_utils/vllm_engine.py).
- vLLM translation sidecar in [slime/backends/vllm_utils/vllm_translation_sidecar.py](slime/backends/vllm_utils/vllm_translation_sidecar.py).

## 3. Problem

- [slime/ray/rollout.py](slime/ray/rollout.py) mixes shared and backend-specific engine creation paths.
- [slime/rollout/sglang_rollout.py](slime/rollout/sglang_rollout.py) is ~95% backend-agnostic but has one inlined SGLang-specific code path (RadixTree, ~14 lines) and uses SGLang-prefixed arg names for generic rollout concepts.
- [slime/utils/arguments.py](slime/utils/arguments.py) still has SGLang alias behavior in vLLM path.
- [slime/backends/fsdp_utils/update_weight_utils.py](slime/backends/fsdp_utils/update_weight_utils.py) hard-imports SGLang internals (`FlattenedTensorBucket`, `MultiprocessingSerializer`, `monkey_patch_torch_reductions`) and calls SGLang-specific engine RPC names (`update_weights_from_tensor`, `update_weights_from_distributed`), making FSDP weight sync unusable with vLLM engines.

## 4. Goals and Non-Goals

### Goals

- Keep runtime refactor minimal and localized to `EngineGroup` + creation abstraction.
- Isolate the one SGLang-specific executor code path behind a strategy hook; keep functions as functions.
- Rename SGLang-prefixed arg names to generic rollout names to eliminate naming coupling.
- Reduce backend leakage in argument finalization.
- Preserve current external behavior, call sites, and import paths.

### Non-Goals

- No rewrite of `SGLangEngine` or `VLLMEngine` internals.
- No algorithmic changes to GRPO/PPO.
- No mandatory feature parity for unsupported backend capabilities.

## 5. Design

### 5.1 Runtime: Generalize `EngineGroup` (No New Runtime Manager Classes)

Keep [slime/ray/rollout.py](slime/ray/rollout.py) as the orchestration entry file.

Refactor focus:

1. Generalize `EngineGroup.start_engines()` to call backend-aware creation hooks.
2. Abstract engine creation and rollout-server assembly helpers.
3. Keep existing startup function API (`start_rollout_servers`) and return shape.

Proposed helper abstraction points:

- `create_engine_actor_cls(args, worker_type)`
  - returns `ray.remote(SGLangEngine)` or `ray.remote(VLLMEngine)`.
- `create_engine_remote(args, actor_cls, scheduling_strategy, ...)`
  - encapsulates `.options(...).remote(...)` with backend-specific init kwargs.
- `build_rollout_server(...)`
  - standardizes `RolloutServer` construction from engine groups.

`EngineGroup` and `RolloutServer` remain the main shared dataclasses.

### 5.2 Executor: Isolate Backend Logic In-Place (No Class Hierarchy)

#### Current state analysis

[slime/rollout/sglang_rollout.py](slime/rollout/sglang_rollout.py) (577 lines) is **already ~95% backend-agnostic**:

| Function / Class | Lines | Backend-specific? | Notes |
|---|---|---|---|
| `_get_backend_client()` | 6 | Factory only | Delegates to existing `RolloutBackendClient` subclasses |
| `_apply_backend_response()` | 20 | No | Uses `RolloutBackendResponse` contract |
| `GenerateState` | 37 | **Naming only** | References `sglang_server_concurrency`, `sglang_dp_size`, `sglang_enable_deterministic_inference` — all are generic rollout concepts with SGLang-prefixed names |
| `generate()` | 58 | **14 lines** | RadixTree middleware path (L170-183) is 100% SGLang-specific; the else branch (L185-195) already uses `RolloutBackendClient` |
| `generate_and_rm()` | 60 | No | Shared orchestration (semaphore, custom func, reward) |
| `generate_and_rm_group()` | 37 | No | Group parallelism + deterministic seeds |
| `abort()` | 38 | No | Already uses `backend.abort()` |
| `generate_rollout_async()` | 71 | No | Main loop, filtering, metrics |
| `eval_rollout()` / `eval_rollout_single_dataset()` | 118 | **Naming only** | `sglang_enable_deterministic_inference` reference |
| `generate_rollout()` | 41 | No | Sync entry point |

**Conclusion**: the actual backend logic that needs isolation is **one code path** (~14 lines) inside `generate()`. Everything else is either already abstracted through `RolloutBackendClient` or is a naming-only coupling (SGLang-prefixed arg names for generic concepts).

#### Approach

1. **Extract the RadixTree path into a strategy hook** that `generate()` calls conditionally.
2. **Rename SGLang-prefixed args** to generic names (coordinated with Phase 3 args refactor).
3. **Keep functions as functions** — they compose well and callers (`train.py`, OPD, multi-agent) import them directly.

#### Concrete changes

**Step 1 — Extract RadixTree strategy from `generate()`**

Current `generate()` has an `if use_radix: ... else: backend.generate(...)` branch.
Refactor into:

```python
# slime/rollout/sglang_rollout.py — generate() simplified

async def generate(args, sample, sampling_params):
    ...
    input_ids = ...  # shared prompt encoding (unchanged)

    strategy = _get_generate_strategy(args)
    resp = await strategy(args, sample, input_ids, sampling_params)
    _apply_backend_response(sample, resp, args)
    return sample
```

Two strategies:

```python
# Still in sglang_rollout.py (no new file needed)

def _get_generate_strategy(args):
    """Return the generate coroutine to use."""
    if _is_radix_tree_enabled(args):
        return _generate_radix_tree      # SGLang-only path
    return _generate_via_backend_client   # Generic path (SGLang or vLLM)

def _is_radix_tree_enabled(args) -> bool:
    return (
        args.use_slime_router
        and "RadixTreeMiddleware" in getattr(args, "slime_router_middleware_paths", [])
    )

async def _generate_radix_tree(args, sample, input_ids, sampling_params) -> RolloutBackendResponse:
    """SGLang RadixTree middleware path — returns normalized response."""
    from slime.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree
    url = f"http://{args.rollout_router_ip}:{args.rollout_router_port}/generate"
    payload = { ... }  # existing payload construction
    output = await post(url, payload, headers=headers)
    sample = await postprocess_sample_with_radix_tree(args, sample, output)
    return _extract_response_from_sample(sample)  # normalize to RolloutBackendResponse

async def _generate_via_backend_client(args, sample, input_ids, sampling_params) -> RolloutBackendResponse:
    """Generic backend client path — works for SGLang and vLLM."""
    backend = _get_backend_client(args)
    base_url = f"http://{args.rollout_router_ip}:{args.rollout_router_port}"
    req = RolloutBackendRequest(...)
    return await backend.generate(req, base_url, headers=headers)
```

**Step 2 — Rename SGLang-prefixed args to generic names**

| Current name | New name | Reason |
|---|---|---|
| `sglang_server_concurrency` | `rollout_concurrency` | Controls request parallelism for any backend |
| `sglang_dp_size` | `rollout_dp_size` | Data-parallel sharding, not SGLang-specific |
| `sglang_router_ip` / `sglang_router_port` | `rollout_router_ip` / `rollout_router_port` | Router endpoint, backend-agnostic |
| `sglang_router_policy` | `rollout_router_policy` | Routing strategy |
| `sglang_enable_deterministic_inference` | `rollout_deterministic_inference` | Seed-based determinism |
| `vllm_base_url` | (remove) | Folded into `rollout_router_ip:port`, no special case |

Legacy aliases kept in [slime/utils/arguments.py](slime/utils/arguments.py) for one release cycle (coordinated with Phase 3).

**Step 3 — No new files**

The file stays as [slime/rollout/sglang_rollout.py](slime/rollout/sglang_rollout.py) during this phase.
Optionally rename to `slime/rollout/rollout.py` in Phase 4 cleanup, since the file is backend-agnostic after the refactor.

### 5.3 Arguments/Config Refactor

In [slime/utils/arguments.py](slime/utils/arguments.py), split into:

1. Shared rollout args.
2. SGLang backend args/validation.
3. vLLM backend args/validation.

Add backend finalizers:

- `finalize_sglang_args(args)`
- `finalize_vllm_args(args)`

Move SGLang alias fallback out of shared finalize flow.

### 5.4 Weight Sync: Decouple FSDP `update_weight_utils.py` from SGLang Internals

#### Current state analysis

[slime/backends/fsdp_utils/update_weight_utils.py](slime/backends/fsdp_utils/update_weight_utils.py) (287 lines) has two concrete classes:

| Class | Weight-push method | SGLang coupling |
|---|---|---|
| `UpdateWeightFromTensor` | IPC via Gloo gather → `engine.update_weights_from_tensor.remote()` | Imports `FlattenedTensorBucket`, `MultiprocessingSerializer`, `monkey_patch_torch_reductions` directly from `sglang.srt.*` |
| `UpdateWeightFromDistributed` | NCCL broadcast → `engine.update_weights_from_distributed.remote()` | Calls `engine.init_weights_update_group.remote()` — SGLang engine API |

The abstract base `UpdateWeight` itself is clean (only PyTorch + Ray).

The Megatron side already solved this: [slime/backends/megatron_utils/sglang.py](slime/backends/megatron_utils/sglang.py) centralizes all SGLang imports into one shim. The FSDP side duplicates these imports inline.

#### Coupling points

1. **SGLang utility imports (L13-26)** — `monkey_patch_torch_reductions`, `MultiprocessingSerializer`, `FlattenedTensorBucket` are imported directly from `sglang.srt.*` with try/except version fallbacks.
2. **`UpdateWeightFromTensor.update_bucket_weights()`** — uses `FlattenedTensorBucket` to flatten tensors, `MultiprocessingSerializer` to serialize, then calls `engine.update_weights_from_tensor.remote()`.
3. **`UpdateWeightFromDistributed.connect_rollout_engines()`** — calls `engine.init_weights_update_group.remote()` which is an SGLang engine method.
4. **`UpdateWeightFromDistributed.update_bucket_weights()`** — calls `engine.update_weights_from_distributed.remote()` which is an SGLang engine method.

All four points assume the engine actor exposes SGLang's RPC interface. vLLM engines expose different method names.

#### Approach

**Step 1 — Centralize SGLang imports via a shim (same pattern as Megatron)**

Reuse or mirror the existing [slime/backends/megatron_utils/sglang.py](slime/backends/megatron_utils/sglang.py) pattern:

```python
# slime/backends/fsdp_utils/sglang_compat.py (new, ~15 lines)
try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket
```

Then `update_weight_utils.py` imports from `sglang_compat` — one import line instead of five, and only loaded when SGLang is the active backend.

**Step 2 — Abstract engine RPC calls behind a protocol**

The engine actors (`SGLangEngine`, `VLLMEngine`) already expose weight-sync methods but with different names/signatures. Introduce a lightweight protocol or adapter:

```python
# In update_weight_utils.py or a small helper

def _call_engine_update_tensor(engine, backend: str, **kwargs):
    """Dispatch IPC weight update to the correct engine method."""
    if backend == "vllm":
        return engine.update_weights_from_tensor.remote(**kwargs)  # vLLM uses same name via NcclBridge
    return engine.update_weights_from_tensor.remote(**kwargs)  # SGLang native

def _call_engine_update_distributed(engine, backend: str, **kwargs):
    if backend == "vllm":
        return engine.update_weights_from_distributed.remote(**kwargs)
    return engine.update_weights_from_distributed.remote(**kwargs)

def _call_engine_init_weight_group(engine, backend: str, **kwargs):
    if backend == "vllm":
        return engine.init_weights_update_group.remote(**kwargs)
    return engine.init_weights_update_group.remote(**kwargs)
```

> Note: Currently `VLLMEngine` already mirrors these method names (it wraps them via `NcclBridge`), so the dispatch functions may initially be identical. The value is making the indirection explicit so future method-name divergence is handled in one place.

**Step 3 — Lazy-import SGLang utilities only when backend is SGLang**

Move the `FlattenedTensorBucket` / `MultiprocessingSerializer` imports inside `UpdateWeightFromTensor.update_bucket_weights()` behind a lazy import, so the module can be loaded in a vLLM-only environment without SGLang installed.

#### What changes

| File | Change |
|---|---|
| `slime/backends/fsdp_utils/sglang_compat.py` | New shim file (~15 lines) centralizing SGLang imports |
| `slime/backends/fsdp_utils/update_weight_utils.py` | Replace 5 inline SGLang imports with one `from .sglang_compat import ...`; add backend-aware engine dispatch helpers |

#### What does NOT change

- `UpdateWeight` abstract base class — already clean.
- `UpdateWeightFromDistributed` NCCL logic — the broadcast itself is pure PyTorch; only the engine RPC dispatch gets a thin wrapper.
- Megatron-side weight sync — already has its own shim, not touched.

