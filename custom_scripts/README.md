# custom_scripts

Launch scripts for the large-model inference runs that do not fit the
`examples/` single-node pattern. Each script pairs with the config of the same
name, starts a Mooncake master, a Redis server and a `vllm serve` producer, then
runs `torchspec.train_entry` against that endpoint.

| Script | Config | Target model |
|---|---|---|
| `launch_ornith_1p5_397b.sh` | `vllm_ornith_1p5_397b.yaml` | `ornith-ai/Ornith-1.5-397B` |
| `launch_nemotron_3_super_120b.sh` | `vllm_nemotron_3_super_120b.yaml` | `aisingapore/Nemotron-SEA-LION-v5-120B-A12B` |

This file holds the reasoning behind flag choices that are not self-evident from
the scripts. Measurements are recorded with the job IDs that produced them;
treat them as data, not as theory to be re-derived.

## Aux hidden-state layer IDs

Applies to both scripts. The same note lives in
`torchspec/inference/engine/vllm_engine.py`.

Layer IDs use **post-layer semantics**: "capture the residual stream after layer
N runs". vLLM's capture hook fires at the INPUT of each listed layer (= the
output of the previous layer), so the script shifts by +1 to align with sglang's
convention.

vLLM's `_maybe_add_hidden_state` is called with `layer_idx + 1` *after* each
layer runs, so valid capture indices are `[0, num_hidden_layers]`. We keep ids
up to and including `num_hidden_layers`.

The final entry is the model's post-last-layer / pre-`norm` slot, reserved by
vLLM. It captures `last_hidden_states` for target logit computation; because it
is pre-norm, training applies the model's final norm itself on top of it.

Concretely, for Ornith-1.5-397B (60 layers):

```
vllm_ornith_1p5_397b.yaml   aux_hidden_states_layers:         [1, 9, 17, 25, 33, 41, 49, 57]
launch_ornith_1p5_397b.sh   eagle_aux_hidden_state_layer_ids: [2, 10, 18, 26, 34, 42, 50, 58, 60]
```

and for Nemotron-SEA-LION-v5-120B-A12B (88 layers):

```
vllm_nemotron_3_super_120b.yaml   aux_hidden_states_layers:         [1, 12, 22, 32, 43, 54, 64, 74, 85]
launch_nemotron_3_super_120b.sh   eagle_aux_hidden_state_layer_ids: [2, 13, 23, 33, 44, 55, 65, 75, 86, 88]
```

In both cases the script's list is the config's list with +1 added to every
entry, then `num_hidden_layers` appended as the final `last_hidden_states` slot.

Two invariants the config comments also state: `len(aux_hidden_states_layers)`
must equal `mooncake.num_aux_layers`, and the launch script must pass that list
shifted by +1 **plus** `num_hidden_layers` — so the script's list is always one
entry longer than the config's.

## Ornith-1.5-397B

### KV-cache geometry — measured, do not re-theorise

Three runs, same 24.7 GiB pool, same model, varying only the two flags below.
"KV needed" is vLLM's own `_check_enough_kv_cache_memory` figure for one
16,385-token request (`kv_cache_utils.py:879`):

| job | block size | prefix caching | KV needed/req | outcome |
|---|---|---|---|---|
| 210241 | 528 | ON (`align`) | ~21 GiB | ran, concurrency 1.18x |
| 211227 | 272 | OFF (`none`) | 64.01 GiB | refused to start |
| 211242 | 528 | OFF (`none`) | 124.04 GiB | refused to start |

Two things that comparison establishes, and nothing in the script should
contradict:

1. **`--no-enable-prefix-caching` is NOT safe on this model**, despite the 0.0%
   hit rate in job 210241. Prefix caching is what selects `mamba_cache_mode`
   `align`; without it the mode is `none`, and vLLM's accounting for `none` on
   this GDN hybrid demands ~6x more per request. Prefix caching here is
   load-bearing paging configuration, not a cache. Leave it ON.

2. **Available KV cache memory does NOT move with either flag** (24.79 vs 24.71
   GiB). The pool is `(total * gpu_memory_utilization) - weights - activation
   peak`; with 92.79 GiB of weights it is pinned near 24.7 GiB. Page-size tuning
   only redistributes that fixed pool. The only levers on concurrency are
   shrinking the weights (fp8), raising `--gpu-memory-utilization`, or lowering
   `--max-model-len`.

**Untested and worth one run:** keep prefix caching ON (`align`) *and* add
`--mamba-ssm-cache-dtype bfloat16`. In both refused runs the requirement scaled
exactly with page size (124.04/64.01 = 1.938, 528/272 = 1.941), i.e. a fixed
page count times `page_size_bytes`. If `align` mode scales the same way, halving
the mamba page should take ~21 GiB/req to ~11 GiB and concurrency from 1.18x to
roughly 2.2x. That is a prediction, not a measurement — verify against the
`Maximum concurrency` line before believing it.

### torch.compile / CUDA graph staging

`--enforce-eager` used to sit in the vLLM args. It disabled BOTH torch.compile
and CUDA graphs — vLLM's own warning in `logs/210241.log`:

```
Enforce eager set, disabling torch.compile and CUDAGraphs.
This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
```

Nothing in this repo documents a reason for it, and vLLM's
`CacheOnlyAttentionLayer` (the hidden-state capture layer) is written *for*
torch.compile: it threads a dummy tensor from `unified_kv_cache_update` into
`dummy_attention` purely so inductor cannot reorder or DCE the cache scatter.

One caveat that makes staging worthwhile: `extract_hidden_states.py` defines its
own plain-Python `unified_kv_cache_update`, NOT the registered
`torch.ops.vllm.unified_kv_cache_update` that vLLM adds to `splitting_ops`. So
on this path the scatter is traced inline rather than being a graph split point.
That is what stage 1 is testing.

Exactly one `COMPILE_ARGS` line in the script should be uncommented. Validate
hidden states against an eager run before advancing a stage.

| Stage | Setting | What it tests |
|---|---|---|
| 1 (active) | `-cc.mode=3 -cc.cudagraph_mode=NONE` | inductor only, no graph capture — separates "does compile preserve the hidden-state scatter" from "does capture work" |
| 2 | `-cc.mode=3 -cc.cudagraph_mode=PIECEWISE` | adds piecewise graphs; attention stays outside the graph |
| 3 | `-cc.mode=3 -cc.cudagraph_mode=FULL_AND_PIECEWISE` | vLLM's v1 default — full graphs for decode, piecewise for prefill |
| rollback | `--enforce-eager` | the previous behaviour |

`cudagraph_capture_sizes` is capped at 16 deliberately. Job 210241 never had
more than 8 requests Running across 4,179 sampled intervals, so the default
ladder (`[1,2,4] + range(8,256,8) + ...`) would capture ~30 shapes we never hit
and reserve GPU memory we cannot spare at 24.79 GiB of KV cache.

Watch the startup log for either of these — both mean the stage silently did not
take effect:

```
Piecewise compilation with empty splitting_ops ... Setting cudagraph_mode to NONE
Enforce eager set, disabling torch.compile and CUDAGraphs
```
