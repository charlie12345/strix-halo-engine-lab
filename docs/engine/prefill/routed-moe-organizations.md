# Routed MoE candidate organizations

The DA16 partial-batch tiers and 64-column register tile retain separate,
default-OFF controls. Their presence in one build does not qualify either
organization or their composition for production.

| Control | Request | Fallback |
|---|---|---|
| `STRIX_QPF2K_MOE_TIERS` | `1`: padded narrow tier; `2`: compact narrow tier | Unset or `0`: incumbent batches |
| `STRIX_QPF2K_MOE_DOWN_LDS_EPI` | `1`: coalesced down stores through LDS | Unset or `0`: incumbent stores; unaligned views retain scalar stores |
| `STRIX_QPF2K_MOE_TILE64` | `1`: wider register tile for eligible complete batches | Unset or `0`: incumbent register tile |

The tier organization takes precedence over the wider register tile. When no
tier split is admitted, an admitted LDS epilogue also takes precedence over the
wider tile. A declined tile request prints its reason once. This selection
preserves the separately gated transaction; it does not combine their proposed
gains. An unsupported LDS view can still admit the tile under its existing
guards. The wider tile retains its baseline and GSQ-OFF restrictions.

The listed-batch selector and private remainder instantiation tag occupy
separate template parameters. Incumbent entry points use neither; list kernels
use the selector and tag 2, and the wider tile's remainder uses tag 1. The
private tag is propagated through the projection body so remainder compilation
cannot silently reuse the incumbent helper instantiation.

Both fixture timing interfaces use the same three stream event boundaries:
before counter clearing, after all gate/up producers, and after all down
producers. Events are absent in ordinary execution. Every completed DA16 path,
including the wider tile and tier lists, records the down boundary when requested.

Build `strix-hip-moe-wmma-prefill-tests` with the existing HIP probe option.
The `tiers` and `tile64` byte suites test each organization separately. The
`tiers-coexist` mode repeats the tier byte suite with simultaneous wider-tile
requests and checks that admitted tier/LDS paths retain precedence. It compares
producer bytes, certificates and ordered merge/commit/finish output against the
incumbent transaction. It does not establish the speed of a jointly optimized
organization. The existing `tiersbench` and `tile64time` modes remain isolated
from the other candidate's environment controls.

Qualification still requires the compiler/resource gates, incumbent machine-code
comparison, complete native byte gates and interleaved complete-call timing.
Select a qualified organization by the actual batch regime; a geometric mean
cannot justify a regression in a particular regime. No timing improvement or
default change is claimed here.
