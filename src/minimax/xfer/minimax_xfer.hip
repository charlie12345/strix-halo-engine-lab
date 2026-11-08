// MiniMax instantiation and its existing non-template entry point.
#include "minimax_xfer.hpp"
#include "../../xport/channel_impl.hip"
namespace strix::xport {
template struct ChannelOptions<minimax::xfer::MinimaxTraits>;
template class Channel<minimax::xfer::MinimaxTraits>;
}
namespace strix::minimax::xfer {
void argmax_exchange(Exchanger& x, const float* logits, unsigned rows, unsigned vn, unsigned ld,
                     std::uint32_t id0, std::uint64_t step, std::uint32_t* out_ids) {
    strix::xport::argmax_exchange<MinimaxTraits>(x, logits, rows, vn, ld, id0, step, out_ids);
}
}
