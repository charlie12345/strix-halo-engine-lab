#pragma once
// System-scope producer publication and consumer lifetime gates.
#include "channel.hpp"

#if defined(__HIPCC__)
namespace strix::xport {

__device__ __forceinline__ void publish(const ProducerTarget& t, unsigned blocks) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const auto old = __hip_atomic_fetch_add(t.counter, 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if (old + 1U == blocks) {
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            __hip_atomic_store(t.counter, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(t.flag, t.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

// Consumer side of recv_view(): ALL threads of EVERY block of the first kernel that reads the payload call this before
// reading it. One lane per block polls the RX flag with a system-scope acquire (s_sleep between polls); the barrier
// publishes the result to the block. A failed exchange releases the flag with ~0, so the kernel never hangs.
__device__ __forceinline__ void wait_view(const RxView& v) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const bool poller = v.gate == nullptr || (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
        if (poller) {
            while (__hip_atomic_load(v.flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < v.sequence) __builtin_amdgcn_s_sleep(2);
            if (v.gate != nullptr) __hip_atomic_store(v.gate, v.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {  // block 0 is dispatched first and never waits on another block: no deadlock
            while (__hip_atomic_load(v.gate, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < v.sequence) __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
}

}  // namespace strix::xport
#endif
