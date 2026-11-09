// GLM instantiation and its existing non-template entry point.
#include "glm_xfer.hpp"
#include "../../xport/channel_impl.hip"
namespace strix::xport {
template struct ChannelOptions<glm::xfer::GlmTraits>;
template class Channel<glm::xfer::GlmTraits>;
template struct ChannelOptions<glm::xfer::GlmSeqparTraits>;
template class Channel<glm::xfer::GlmSeqparTraits>;
}
namespace strix::glm::xfer {
void argmax_exchange(Exchanger& x, const float* logits, unsigned rows, unsigned vn, unsigned ld,
                     std::uint32_t id0, std::uint64_t step, std::uint32_t* out_ids) {
    strix::xport::argmax_exchange<GlmTraits>(x, logits, rows, vn, ld, id0, step, out_ids);
}
}
