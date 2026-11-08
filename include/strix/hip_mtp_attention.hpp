#pragma once
#include <span>
#include "strix/hip_mtp_resources.hpp"
namespace strix {
// Complete metadata admission for BOTH stages, before any proposal enqueue.
// Uses the incumbent BF16 indexer/Q8 attention kernels and independent MTP KV
// and raw-index storage; predictor position is hidden-source position.
void preflight_hip_mtp_attention(const MtpNativeContext&);
// Teacher cache reconstruction only: exact incumbent index-key, K/V projection,
// key norm/RoPE and BF16 stores for c.position. No query, index selection,
// attention output, host synchronization or publication. Only that row changes.
void preflight_hip_mtp_cache_rows(const MtpNativeContext&);
// fast_cache is proposal-only; prompt priming keeps the incumbent Q8 producer.
void launch_hip_mtp_cache_rows(const MtpNativeContext&,void* stream,bool fast_cache=false);
// Tiled prompt priming batch: launch_hip_mtp_cache_rows (fast_cache off) for first.position .. +count-1 (count 1..8) of
// one owner, row t reading its hidden (BF16 [hidden]) from input[t] and writing its index key (F32 [128]), K and V
// (F32 [512] each) scratch; the KV/raw-index rows and the status word are the owner's. Row t's cache bytes equal the
// one-row launch at that position, bit for bit; the owner's attention workspace is not written.
struct HipMtpCacheRowsBatch {
    const std::uint16_t* input[8]{};
    float* index_key[8]{};
    float* key[8]{};
    float* value[8]{};
};
void launch_hip_mtp_cache_rows_batch(const MtpNativeContext& first,const HipMtpCacheRowsBatch& rows,std::uint32_t count,void* stream);
void launch_hip_mtp_indexer(const MtpNativeContext&,void* stream);
void launch_hip_mtp_sparse_attention(const MtpNativeContext&,void* stream);
// Draft-only: same stages with the Q/K/V and output Q8 projections on the
// wave-per-row draft GEMV (different summation order; drafts only).
void launch_hip_mtp_sparse_attention_draft(const MtpNativeContext&,void* stream);
// Multi-session draft batch (production branch: STRIX_QPERSIST_MTP_CACHE, _DRAFT_SCORE and _QSA_TAIL on): the
// indexer and the draft sparse attention of every context on `stream`, each row with its own position, caches,
// counts, workspace and status word; every row's results equal its own one-token launch, bit for bit.
[[nodiscard]] bool hip_mtp_attention_draft_rows_supported();
// The batch's BF16 projection (index query/key) over 1..8 rows: row t equals enqueue_hip_mtp_attention_bf16_linear on
// inputs[t] alone, bit for bit (raw entry for the fixture; the batched indexer uses the same kernels).
void enqueue_hip_mtp_attention_bf16_linear_rows(std::span<const std::uint16_t* const> inputs,const void* bf16,std::uint32_t width,
    std::uint32_t rows,std::span<float* const> outputs,void* stream);
void launch_hip_mtp_indexer_rows(std::span<const MtpNativeContext> contexts,void* stream);
void launch_hip_mtp_sparse_attention_draft_rows(std::span<const MtpNativeContext> contexts,void* stream);
// Raw exact Q8 projection used by the attention stages (benchmark reference).
void enqueue_hip_mtp_attention_q8_linear(const std::uint16_t* input,const void* q8,std::uint32_t width,
    std::uint32_t rows,float* output,void* stream);
void enqueue_hip_mtp_attention_bf16_linear(const std::uint16_t* input,const void* bf16,std::uint32_t width,
    std::uint32_t rows,float* output,void* stream);
} // namespace strix
