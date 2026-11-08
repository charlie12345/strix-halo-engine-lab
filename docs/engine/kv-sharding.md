# KV storage and GLM token sharding

Long-context requests need memory for every token whose attention state must remain available. The key/value (KV) state grows with the request, alongside model weights, workspace and other request state. How much each request can hold therefore depends on the engine's cache layout and the memory admission at startup.

## How the engines store attention state

The comparison below describes one request on a two-node pair of 128 GB APUs connected over USB4, at 8K context unless the row states otherwise. The per-token values are cache-layout accounting for this pair, not a maximum-context claim.

| Engine | State held per token per node | Layout |
|---|---:|---|
| GLM | 17,737 B | Multi-head latent attention (MLA) state for the 11 sparse-attention (DSA) layers is replicated on both nodes in the default exact layout. |
| MiniMax | 76,800 B | Attention state is already split by attention heads across the two nodes. |
| DeepSeek V4.1 | About 894 B | Compressed state is replicated on both nodes. |
| Qwen hybrid | Varies by layer and format | State is split by layer only during two-node prefill; rank 0 holds the complete state afterward. |

The GLM option in this guide changes ownership of GLM's DSA latent state by token. It does not change the KV layout of MiniMax, DeepSeek V4.1 or Qwen.

## What GLM token sharding does

With token sharding, each node retains the latent state for its own token blocks. Both nodes exchange score keys so global top-k selection stays the same as in the replicated path. Each node computes its attention partial over the tokens it owns; the partials are merged in fixed rank order in FP32.

The changed reduction order means sharded execution is a separate FAST numerical profile, not bitwise exact relative to the default replicated exact profile. Its identity is:

`glm-dsa-kvshard-rank01-fp32-v3-declaredexp-owned-tree256-fma-rankmerge-FAST`

On the two-node 128 GB APU pair over USB4, for one request at 8K context, the measured generated-token vectors matched the replicated exact run in 128/128 tested positions. At 64K context on the same pair and one request, they matched in 32/32 tested positions. These token matches do not make the logits bitwise equal: logit differences are at reorder scale and grow with depth. Compare maximum absolute logit difference when qualifying a run.

## When to use it and how it is selected

The default `STRIX_GLM_KV_SHARD=auto` keeps replicated state when its startup admission passes. It selects sharding when the replicated layout does not pass admission and the sharded layout does. Startup reports the effective layout and the admission reason. Use `auto` when you want the largest admitted context without manually selecting the layout.

Set `STRIX_GLM_KV_SHARD=1` to force sharding, or `STRIX_GLM_KV_SHARD=0` to force the replicated layout. Set the same value on both ranks. Forced sharding fails startup if its required resources do not fit; forced replication fails startup when that layout does not fit.

The qualified configuration today is one request (`slots 1`), with `STRIX_GLM_PF_PIPELINE=1`, chunked prefill and no MTP or other speculation. Sharding is not selected for a speed gain; it trades lower per-node cache use for additional exchange and reduction work. The identity above applies to sharded FAST execution. The default replicated exact mode remains the bitwise default.

## Measured capacity and speed

All figures below were measured for one request on a two-node pair of 128 GB APUs over USB4. “Exact” means the bitwise default replicated layout; “FAST” means the named sharded profile above. Context is prompt context, and decode speed is milliseconds per generated token.

| Measurement | Exact replicated | FAST token-sharded |
|---|---:|---:|
| Maximum admitted context | 1.45M tokens, with bounded prefill workspace | About 2.85M tokens; 2.85M ran and 2.9M was refused at startup |
| Decode at 8K context | 39.7 ms/token | 41.5 ms/token |
| Decode at 2.0M–2.85M context | — | 43.6–45.7 ms/token |
| Prefill at 8K context | 17.1 s | 18.2 s |
| Prefill at 64K context | 145.1 s | 155.9 s |

These are measured ceilings and timings for this pair and configuration, not guarantees for other hardware or workloads. Use sharding when replicated state cannot admit the context you need, and expect slower decode where both layouts fit.

## Verify the selected profile

Each rank prints the startup admission decision and effective layout. For an automatic decision, inspect both ranks' startup lines to see which layout passed and why. The effective identity line has this form:

```text
rank R kv-shard effective=1 identity=glm-dsa-kvshard-rank01-fp32-v3-declaredexp-owned-tree256-fma-rankmerge-FAST
```

For replicated execution, the effective identity is `replicated`. The execution receipt also includes the numerical identity. Keep the startup output and receipt with the run's prompt/context and settings.

To compare layouts, run the same tokenized prompt, model assets, request settings and context twice on the same pair: once with `STRIX_GLM_KV_SHARD=0` and once with `STRIX_GLM_KV_SHARD=1` (or `auto` when admission selects sharding). Compare the complete generated token vectors and the maximum absolute difference between corresponding logits. A matching token vector is useful evidence for that prompt and length; it is not a bitwise-logit equivalence claim.

For per-layer diagnosis, set `STRIX_GLM_KVSHARD_DUMP` to an output directory before starting both ranks. The optional dump records selected token IDs, pool scores and attention output rows for the sharded DSA path. Use it to locate where selection or attention values first diverge; it is diagnostic output, not a replacement for complete-run comparison.

## Costs and limits

- Sharding reduces replicated DSA latent-cache storage by retaining only each rank's token blocks. It adds score-key exchange, per-rank partial attention and an ordered FP32 merge.
- The FAST profile changes summation order. Its logits differ from the replicated exact profile even when generated tokens match in the tested cases.
- The measured 8K decode and prefill timings are slower with sharding on this pair. Prefer the replicated exact layout when it admits the request and bitwise identity with the default is required.
- The current qualified scope is one request, pipeline prefill, and no MTP or speculation. Do not infer multi-request, speculative, other-hardware or beyond-64K prefill performance from the measurements above.
- Automatic selection is an admission decision, not a speed heuristic. If neither layout passes startup admission, the request is refused rather than silently run beyond the admitted capacity.
