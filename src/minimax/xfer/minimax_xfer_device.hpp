#pragma once
#include "minimax_xfer.hpp"
#include "../../xport/channel_device.hpp"
#if defined(__HIPCC__)
namespace strix::minimax::xfer {
using strix::xport::publish;
using strix::xport::wait_view;
__device__ inline void mmx_publish(const ProducerTarget& t, unsigned blocks) { publish(t, blocks); }
__device__ inline void mmx_wait(const RxView& v) { wait_view(v); }
}
#endif
