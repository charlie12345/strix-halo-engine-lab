#pragma once
#include "hip_decode_options.hpp"
#include "strix/qwen_moe_prepared.hpp"
#include "strix/serve/json.hpp"

namespace strix::decode_cli {
struct MoeDirectBPolicy {
    bool requested=false,effective=false;
    std::string reason="off";
    std::uint64_t bytes=0;
};
// The prepared planes are a MoE weight organisation with no rotary site. Plain serial YaRN prefill (no MTP, HTTP, cache or
// session; the callers above already decline those) runs the same tiled known-prefix prefill as the native CLI, so only the
// MTP verifier windows (which must stay serial-equivalent) and the other YaRN-specific routes stay out.
inline bool yarn_plain_prefill(const Options& o) {
    return o.rotary.mode()==RotaryMode::yarn&&!o.mtp.enabled&&!o.mtp.supplied&&!o.http_serve&&!o.http_batch;
}
template<class Environment>
inline MoeDirectBPolicy moe_direct_b_policy(const Options& o,const Environment& env,const ModelManifest& manifest,bool hip_fp16) {
    MoeDirectBPolicy p;
    const auto it=env.find("STRIX_QPF2K_MOE_DIRECT_B");
    if(it==env.end()||it->second=="0")return p;
    p.requested=true;
    const auto decline=[&](const char* reason){p.reason=reason;return p;};
    if(it->second!="1")return decline("invalid-switch");
    const auto baseline=env.find("STRIX_BASELINE");
    if(o.baseline||(baseline!=env.end()&&baseline->second=="1"))return decline("baseline");
    if(!hip_fp16)return decline("HIP-FP16-required");
    const auto bf16=env.find("STRIX_QPF2K_MOE_ROUTE_BF16");
    if(bf16!=env.end()&&!bf16->second.empty()&&bf16->second.front()!='0')return decline("route-BF16");
    const auto pair=numerical_profile_pair(o);
    if(!numerical_profile_fast_prefill(pair.prefill)||!(pair.fast_components&kFastPrefillMoe)||
        !o.known_prefix_window||!o.known_prefix_full_rows||!o.known_prefix_expert_union||!o.disabled_prefill_features.empty())
        return decline("FP16-full-row-prefill-required");
    if(o.known_prefix_prefill_rows!=4096U&&o.known_prefix_prefill_rows!=8192U)return decline("prompt-shape");
    if(o.mtp.enabled||o.mtp.supplied||!o.mtp.cohort.empty()||!o.mtp.copy_wide_rows.empty()||
        o.http_serve||o.http_batch||!o.serving_prefix_files.empty()||o.serving_concurrency!=1U||
        o.serving_tensor_cohort_rows!=1U||o.serving_prefill_rows||o.prompt_lookup||
        o.prefix_cache_host_bytes||o.prefix_cache_roundtrip||!o.session_load.empty()||!o.session_save.empty()||
        o.session_tiled_suffix||o.session_live_control||o.session_management_requested||o.prefill_moe_route_bf16||
        (o.rotary.mode()!=RotaryMode::native&&!yarn_plain_prefill(o)))return decline("execution-route");
    const auto require=[&](const char* name,std::string_view value){const auto e=env.find(name);return e==env.end()||e->second==value;};
    if(!require("STRIX_INTPREFILL_MOE_TUNE","82")||!require("STRIX_QPF2K_MOE","0")||!require("STRIX_QPF2K_MOE_SMALL","0"))
        return decline("schedule");
    for(std::uint32_t i=0;i<manifest.tensors.size();++i)if(qwen_moe_prepared::admitted_spec(manifest,i)) {
        const auto bytes=manifest.tensors[i].byte_size;
        if(bytes>UINT64_MAX-p.bytes)return decline("size-overflow");
        p.bytes+=bytes;
    }
    if(!p.bytes)return decline("no-paired-Q4_K-tensors");
    p.effective=true;p.reason="candidate; pending-native-qualification";return p;
}
inline std::string moe_direct_b_json(bool requested,bool effective,std::string_view reason,std::uint64_t bytes) {
    auto j=serve::Json::object();
    j.set("requested",serve::Json(requested));j.set("effective",serve::Json(effective));
    j.set("reason",serve::Json(reason));j.set("prepared_bytes",serve::Json(static_cast<double>(bytes)));
    j.set("layout_version",serve::Json(static_cast<double>(qwen_moe_prepared::layout_version)));
    j.set("numerical_identity",serve::Json("unchanged"));return j.dump();
}
}
