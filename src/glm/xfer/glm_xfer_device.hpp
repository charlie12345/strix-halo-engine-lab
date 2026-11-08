#pragma once
#include "glm_xfer.hpp"
#include "../../xport/channel_device.hpp"
#if defined(__HIPCC__)
namespace strix::glm::xfer {
using strix::xport::publish;
using strix::xport::wait_view;
__device__ inline void glmx_publish(const ProducerTarget& t, unsigned blocks) { publish(t, blocks); }
__device__ inline void glmx_wait(const RxView& v) { wait_view(v); }
}
#endif
