#pragma once

#include "strix/mapped_model_file.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/uma_owner_inventory.hpp"

#include <memory>
#include <vector>

namespace strix {
// Owns only newly generated copies, never the original F32 or pack storage.
// Registration is retired before freeing its device backing.
class HipQwenRouterBf16 {
public:
    struct Copy {
        const void* original = nullptr;
        void* device = nullptr;
        std::uint64_t bytes = 0, lease = 0;
    };
    std::vector<Copy> copies;
    std::uint64_t bytes = 0, skipped = 0, reused = 0;
    HipQwenRouterBf16() = default;
    HipQwenRouterBf16(const HipQwenRouterBf16&) = delete;
    HipQwenRouterBf16& operator=(const HipQwenRouterBf16&) = delete;
    ~HipQwenRouterBf16();
    [[nodiscard]] UmaOwnerInventory owner_inventory(std::shared_ptr<const HipQwenRouterBf16> owner) const;
};
[[nodiscard]] std::shared_ptr<const HipQwenRouterBf16> load_hip_qwen_router_bf16(
    const MappedModelFile&, const ModelManifest&, const ModelRuntimeBinding&, bool enabled);
} // namespace strix
