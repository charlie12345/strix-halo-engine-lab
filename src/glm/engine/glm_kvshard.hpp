#pragma once
#include "strix/glm/engine.hpp"
#include "glm/attn/glm_attn_fast.hpp"
#include "glm/engine/glm_kvshard_layout.hpp"
#include <memory>

namespace strix::glm::xfer { class XferExchange; }
namespace strix::glm::engine {
// KV sharding: owned TP2 decode and bounded prefill, no speculative/serving integration.
class KvShard {
public:
    KvShard(dist::Communicator&, unsigned rank);
    ~KvShard();
    KvShard(const KvShard&) = delete;
    KvShard& operator=(const KvShard&) = delete;
    void reset();
    void bind_exchange(std::shared_ptr<xfer::XferExchange>, bool shared_tp);
    void append(const DsaState&, const CallCtx&, const fast::DsaRows&, const float* latent,
                const float* key, const float* gate, const dev::IndexerDev&, std::uint32_t* hist);
    void prepare(const DsaState&, const CallCtx&, const fast::DsaRows&, const float* index_query,
                 const float* weights, std::uint32_t* keys, unsigned stride, std::uint32_t* hist,
                 const float* query, const std::uint16_t* kv_b, const fast::MlaWork&, bool score_keys);
    void attention(const DsaState&, const CallCtx&, const fast::DsaRows&, const std::uint32_t* tokens,
                   const std::uint32_t* count, const float* query, const std::uint16_t* kv_b,
                   const fast::MlaWork&, float* output);
    void prefill(PrefillPath&, const DsaLayerWeights&, DsaState&, const CallCtx&, const float*, float*);
    struct PrefillWork {
        const float *latent, *key, *gate, *index_query, *query;
        float *weights, *qt, *scores, *obar, *output;
        std::uint32_t *keys, *tokens, *counts;
        std::uint64_t path_bytes;
        unsigned attention_window; // installed scratch limit; zero uses the unchanged full-row launcher
    };
    void prefill_rows(const DsaLayerWeights&, const DsaState&, const CallCtx&,
                      const PrefillWork&, std::uint64_t margin);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::glm::engine
