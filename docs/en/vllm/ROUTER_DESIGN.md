# RFC: Replace SGLang Backend with vLLM — Router Integration

---

## Summary

Replace the SGLang inference backend behind **SlimeRouter** with **vLLM** while keeping the existing router and middleware stack completely unchanged.
This RFC covers **only the router layer** — what APIs the vLLM backend must expose, how the existing SlimeRouter is reused, and what translation is needed between the two formats.

**Key design decision:** Reuse vLLM's built-in [OpenAI-compatible API server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) (`vllm serve`) 


---

## 1. Target Architecture

```
 Rollout Workers                    SlimeRouter (NO CHANGE)                vLLM Engines (NEW)
 ──────────────                    ────────────────────────                ──────────────────
                                   ┌──────────────────────┐
 POST /generate ──────────────────▶│ RadixTreeMiddleware   │
                                   │  • prefix cache       │
                                   │  • retry on abort     │
                                   │  • token/logprob cache│
                                   └──────────┬───────────┘
                                              │
                                   ┌──────────▼───────────┐
                                   │ SlimeRouter.proxy()   │         ┌─────────────────────┐
                                   │  • least-connections  │────────▶│ vLLM Rollout        │
                                   │    load balancer      │         │ Adapter (per engine) │
                                   │  • health check loop  │         │                     │
                                   └──────────────────────┘         │ POST /generate      │
                                                                     │   ↓ translate        │
                                                                     │ POST /v1/completions │
                                                                     │   ↓ translate back   │
                                                                     │ → SGLang-format JSON │
                                                                     └─────────┬───────────┘
                                                                               │
                                                                     ┌─────────▼───────────┐
                                                                     │ vLLM Server          │
                                                                     │ (vllm serve)         │
                                                                     │  • /v1/completions   │
                                                                     │  • /health           │
                                                                     │  • /sleep, /wake_up  │
                                                                     │  • /pause, /resume   │
                                                                     │  • /update_weights   │
                                                                     └─────────────────────┘
```

### What stays the same

| Component | Change | Reason |
|---|---|---|
| `SlimeRouter` ([router.py](slime/router/router.py)) | **None** | Engine-agnostic HTTP proxy; only reads JSON responses |
| `RadixTreeMiddleware` ([radix_tree_middleware.py](slime/router/middleware_hub/radix_tree_middleware.py)) | **None** | Operates on request/response JSON; has no engine-specific code |
| `StringRadixTrie` ([radix_tree.py](slime/router/middleware_hub/radix_tree.py)) | **None** | Pure data structure, no engine coupling |
| Middleware loading (`--slime-router-middleware-paths`) | **None** | Dynamic import via `load_function()` |

### What is new

| Component | Description |
|---|---|
| `vllm_rollout_adapter.py` | Lightweight FastAPI process co-located with each vLLM engine. Receives SGLang-format `/generate` requests, translates to vLLM's `/v1/completions`, translates responses back. Also proxies lifecycle endpoints (`/abort_request`, `/health_generate`, etc.). |
| `vllm_engine.py` | Ray actor that manages the vLLM server process lifecycle (via `vllm serve`), the rollout adapter, weight updates, and registration with the router. |

---

## 2. Reusing SlimeRouter — Zero Modification

The SlimeRouter communicates with backends through **five interaction points**. All are already engine-agnostic:

### 2.1 Worker Registration

**Flow:** Engine starts → engine calls `POST /add_worker?url=http://{host}:{port}` → router adds to pool.

```
Router state after registration:
  worker_request_counts["http://10.0.0.1:10090"] = 0
  worker_failure_counts["http://10.0.0.1:10090"] = 0
```

**vLLM action:** The `VLLMEngine` Ray actor calls this endpoint after verifying the vLLM server + rollout adapter are healthy. The registered URL points to the **adapter**, not the raw vLLM server. No router change needed.

### 2.2 Request Proxying

**Flow:** `POST /generate` → middleware pipeline → `SlimeRouter.proxy()` → `httpx` forwards to backend (adapter).

The router selects a backend via **least-connections** (`_use_url()`), forwards the raw request body as-is, and returns the response as-is. It never inspects or transforms the request/response payload.

**vLLM action:** The adapter receives the forwarded request, translates it to `/v1/completions`, calls the co-located vLLM server, translates the response back to SGLang format, and returns it.

### 2.3 Health Check

**Flow:** Background loop calls `GET {worker_url}/health` every N seconds.

- 200 → healthy, reset failure count
- Non-200 or timeout → increment failure count
- Failures ≥ threshold (default 3) → quarantine worker permanently

**vLLM action:** The adapter's `/health` proxies to vLLM's built-in `/health` endpoint (returns 200 when ready). Compatible out of the box.

### 2.4 Worker Listing

**Flow:** `GET /list_workers` → returns `{"urls": [...]}`

Used by the rollout to discover engines for direct abort calls. No engine involvement.

### 2.5 Retrieve from Text (Radix Tree)

**Flow:** `POST /retrieve_from_text` → router looks up the radix tree cache → returns tokens/logprobs.

Fully router-internal. Never reaches the engine.

---

## 3. API Contract — What the vLLM Rollout Adapter Must Expose

The vLLM rollout adapter sits between SlimeRouter and the vLLM server. It receives SGLang-format requests and returns SGLang-format responses.

### 3.1 `POST /generate` — Generation

This is the primary endpoint. The adapter translates between Slime's format and vLLM's `/v1/completions`.

#### Incoming Request (from router)

```json
{
  "input_ids": [128000, 2610, 553, 264, 11190, 18328, 13],
  "input_tokens": [128000, 2610, 553, 264, 11190, 18328, 13],
  "sampling_params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": -1,
    "max_new_tokens": 1024,
    "stop": ["<|endoftext|>"],
    "stop_token_ids": [128001],
    "skip_special_tokens": false,
    "no_stop_trim": true,
    "spaces_between_special_tokens": false
  },
  "return_logprob": true,
  "stream": false
}
```

#### Translated Request (to vLLM `/v1/completions`)

```json
{
  "model": "<model_name>",
  "prompt": [128000, 2610, 553, 264, 11190, 18328, 13],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": -1,
  "stop": ["<|endoftext|>"],
  "stop_token_ids": [128001],
  "skip_special_tokens": false,
  "include_stop_str_in_output": true,
  "spaces_between_special_tokens": false,
  "logprobs": 1,
  "stream": false,
  "extra_body": {
    "return_token_ids": true
  }
}
```

**Key translations:**
- `input_ids` → `prompt` (vLLM accepts `list[int]` as pre-tokenized prompt)
- `max_new_tokens` → `max_tokens`
- `no_stop_trim: true` → `include_stop_str_in_output: true`
- `return_logprob: true` → `logprobs: 1` + `extra_body.return_token_ids: true`

#### vLLM Response (from `/v1/completions`)

```json
{
  "id": "cmpl-abc123",
  "choices": [{
    "text": "I'll help you with that. The answer is 42.",
    "logprobs": {
      "token_logprobs": [-0.152, -0.089, -0.203],
      "tokens": ["I", "'ll", " help"]
    },
    "token_ids": [40, 3358, 1520],
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 3,
    "total_tokens": 10
  }
}
```

#### Translated Response (returned to router)

```json
{
  "text": "I'll help you with that. The answer is 42.",
  "output_ids": [40, 3358, 1520],
  "meta_info": {
    "output_token_logprobs": [
      [-0.152, 40],
      [-0.089, 3358],
      [-0.203, 1520]
    ],
    "finish_reason": {
      "type": "stop"
    },
    "weight_version": 3,
    "prompt_tokens": 7,
    "cached_tokens": 0
  }
}
```

##### Field-by-field contract

| Field | Type | Required | Consumer | Description |
|---|---|---|---|---|
| `text` | `str` | **Yes** | Rollout, Middleware | Generated text (output only, not including prompt) |
| `output_ids` | `list[int]` | **Yes** | Middleware | Generated token IDs. Middleware checks existence as a gate for caching. |
| `meta_info.output_token_logprobs` | `list[[float, int]]` | **Yes** (if `return_logprob`) | Rollout, Middleware | Each element is `[logprob, token_id]`. Used for RL policy ratio calculation. |
| `meta_info.finish_reason` | `{"type": str}` | **Yes** | Rollout, Middleware | Must be `{"type": "stop"}`, `{"type": "length"}`, or `{"type": "abort"}`. **Not** a plain string. |
| `meta_info.weight_version` | `int` | **Yes** | Middleware, Rollout | Current model weight version. Tracked by the adapter (incremented on each weight update). |
| `meta_info.prompt_tokens` | `int` | Nice-to-have | Rollout (stats) | From `usage.prompt_tokens`. |
| `meta_info.cached_tokens` | `int` | Nice-to-have | Rollout (stats) | vLLM doesn't expose this directly; default to `0`. |

### 3.2 `GET /health` — Health Check

```
GET /health
→ Adapter proxies to vLLM's GET /health
→ 200 OK        (engine ready)
→ 503 or timeout (engine not ready / overloaded)
```

vLLM already provides this endpoint. **Passthrough — no translation needed.**

### 3.3 `POST /abort_request` — Cancel Generation

```
POST /abort_request
Body: {"abort_all": true}
→ 200 OK
```

Called **directly** by the rollout to each engine (bypasses the router). The rollout discovers engine URLs via `GET /list_workers`, then sends abort to each.

**vLLM approach:** vLLM uses **HTTP connection close** for abort (via its `@with_cancellation` decorator). When a client disconnects, the in-flight request is automatically cancelled.

**Implementation options:**
1. **Track active connections.** The adapter maintains a set of active `httpx` connections to the vLLM server. On `POST /abort_request`, close all of them — triggering vLLM's cancellation.
2. **Use vLLM's `/pause` endpoint.** Call `POST /pause` to block new requests, then `POST /resume` after the RL training step completes. This is semantically closer to how Slime uses abort (clearing the decks between training generations).

> **Note:** vLLM has `POST /abort_requests` only in disaggregated mode. For standard mode, HTTP disconnect is the canonical abort mechanism.

### 3.4 `GET /health_generate` — Startup Readiness Probe

```
GET /health_generate
→ 200 OK        (model loaded, engine ready for generation)
```

Called by `VLLMEngine.init()` during startup to block until the engine is fully ready. The adapter implements this by calling vLLM's `GET /health` and optionally performing a dummy `/v1/completions` call with `max_tokens=1` to verify end-to-end readiness.

### 3.5 Sampling Params Translation

The request uses SGLang-format parameter names. The adapter translates to vLLM's `/v1/completions` format:

| SGLang field (in request) | vLLM `/v1/completions` field | Notes |
|---|---|---|
| `input_ids` | `prompt` | Direct — vLLM accepts `list[int]` as pre-tokenized prompt |
| `temperature` | `temperature` | Direct |
| `top_p` | `top_p` | Direct |
| `top_k` | `top_k` | Both use `-1` for disabled |
| `max_new_tokens` | `max_tokens` | **Name change** |
| `stop` | `stop` | Direct (list of strings) |
| `stop_token_ids` | `stop_token_ids` | Direct |
| `skip_special_tokens` | `skip_special_tokens` | Direct |
| `no_stop_trim` | `include_stop_str_in_output` | **Same semantics, different name** |
| `spaces_between_special_tokens` | `spaces_between_special_tokens` | Direct |
| `return_logprob` | `logprobs` (set to `1`) | Also add `extra_body.return_token_ids = true` |
| `sampling_seed` | `seed` | Optional |
| — | `model` | Must be set to the model name served by vLLM |

### 3.6 Response Translation Pseudocode

```python
def translate_vllm_response(vllm_resp: dict, weight_version: int) -> dict:
    """Translate vLLM /v1/completions response to SGLang format."""
    choice = vllm_resp["choices"][0]
    usage = vllm_resp.get("usage", {})

    # Build output_token_logprobs: zip logprobs with token IDs
    output_token_logprobs = None
    if choice.get("logprobs") and choice.get("token_ids"):
        output_token_logprobs = [
            [logprob, token_id]
            for logprob, token_id in zip(
                choice["logprobs"]["token_logprobs"],
                choice["token_ids"]
            )
        ]

    # Translate finish_reason: plain string → {"type": str}
    raw_reason = choice.get("finish_reason")
    finish_reason = {"type": raw_reason if raw_reason else "abort"}

    return {
        "text": choice["text"],
        "output_ids": choice.get("token_ids", []),
        "meta_info": {
            "output_token_logprobs": output_token_logprobs,
            "finish_reason": finish_reason,
            "weight_version": weight_version,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": 0,
        }
    }
```

### 3.7 `finish_reason` Translation Table

| vLLM returns | Translate to | Notes |
|---|---|---|
| `"stop"` | `{"type": "stop"}` | Normal completion |
| `"length"` | `{"type": "length"}` | Hit `max_tokens` |
| `None` (aborted/incomplete) | `{"type": "abort"}` | Triggers middleware retry logic (sleep 30s, up to 5 retries) |

---

## 4. Server Launch Configuration

The `VLLMEngine` Ray actor should launch vLLM as follows:

```bash
# Environment
export VLLM_SERVER_DEV_MODE=1

# Launch vLLM server
vllm serve <model_path> \
    --host 0.0.0.0 \
    --port <engine_port> \
    --tensor-parallel-size <tp_size> \
    --enable-sleep-mode \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests
```

The rollout adapter runs on a separate port (`<adapter_port>`) and is the URL registered with the router via `POST /add_worker?url=http://{host}:{adapter_port}`.

```
                Router
                  │
                  ▼
    ┌─────────────────────────┐
    │ vLLM Rollout Adapter    │  ◄── registered with router
    │ port: adapter_port      │
    │                         │
    │ /generate ──translate──▶│──┐
    │ /health ──passthrough──▶│  │
    │ /abort_request          │  │
    │ /health_generate        │  │
    └─────────────────────────┘  │
                                 │
    ┌─────────────────────────┐  │
    │ vLLM Server             │◄─┘
    │ port: engine_port       │
    │                         │
    │ /v1/completions         │
    │ /health                 │
    │ /sleep, /wake_up        │
    │ /pause, /resume         │
    │ /update_weights         │
    │ /init_weight_transfer   │
    └─────────────────────────┘
```

---

## 5. Abort Strategy — Detailed Design

vLLM's abort mechanism differs fundamentally from SGLang's:

| Aspect | SGLang | vLLM |
|---|---|---|
| Abort granularity | Per-request via `POST /abort_request` with `rid` | Per-connection via HTTP disconnect |
| Bulk abort | `{"abort_all": true}` | No built-in equivalent |
| Mechanism | Engine tracks `request_id`, explicit `abort()` | `@with_cancellation` decorator; request cancelled when client disconnects |
| Between-generation abort | Abort + restart | `POST /pause` → training → `POST /resume` |

### Recommended implementation

For the Slime RL use case, the rollout calls `abort_all` between generation rounds (to clear the engine before the next batch). The best vLLM equivalent is:

```python
# In the rollout adapter
@app.post("/abort_request")
async def abort_request(request: Request):
    body = await request.json()
    if body.get("abort_all"):
        # Option 1: Close all tracked httpx connections → triggers vLLM cancellation
        for conn in active_connections:
            await conn.aclose()
        active_connections.clear()

        # Option 2: Use pause/resume (cleaner)
        await httpx.post(f"{vllm_url}/pause")
        await httpx.post(f"{vllm_url}/resume")

    return {"status": "ok"}
```

---

## 6. Endpoints Summary — Gap Analysis

### Engine-side endpoints (vLLM built-in vs. needs implementation)

| Endpoint | SGLang | vLLM Built-in | Action |
|---|---|---|---|
| `POST /v1/completions` | — | ✅ | **Reuse** — target for translation |
| `GET /health` | ✅ | ✅ | **Reuse** as-is (passthrough) |
| `POST /pause` | — | ✅ (dev mode) | **Reuse** for abort/weight-update |
| `POST /resume` | — | ✅ (dev mode) | **Reuse** for abort/weight-update |
| `POST /sleep` | — | ✅ (dev mode) | **Reuse** for weight updates |
| `POST /wake_up` | — | ✅ (dev mode) | **Reuse** for weight updates |
| `POST /collective_rpc` | — | ✅ (dev mode) | **Reuse** for weight reload |
| `GET /is_sleeping` | — | ✅ (dev mode) | **Reuse** for state checks |
| `POST /init_weight_transfer_engine` | — | ✅ (dev mode) | **Reuse** for NCCL setup |
| `POST /update_weights` | — | ✅ (dev mode) | **Reuse** for NCCL weight apply |
| `GET /get_world_size` | — | ✅ (dev mode) | **Reuse** for TP world size |

### Rollout adapter endpoints (to implement)

| Endpoint | Description | Complexity |
|---|---|---|
| `POST /generate` | Translate SGLang → `/v1/completions` → SGLang | **Medium** — main logic |
| `GET /health` | Proxy to vLLM `/health` | **Trivial** |
| `GET /health_generate` | Health + optional dummy completion | **Low** |
| `POST /abort_request` | Close connections or pause/resume | **Low** |
| `GET /flush_cache` | `POST /sleep?level=1` + `POST /wake_up?tags=kv_cache` | **Low** |
| `GET /get_weight_version` | Return adapter-tracked version counter | **Trivial** |

### Router endpoints (no change needed)

| Endpoint | Action |
|---|---|
| `POST /add_worker` | No change |
| `GET /list_workers` | No change |
| `POST /retrieve_from_text` | No change |
| Catch-all proxy | No change |

---



