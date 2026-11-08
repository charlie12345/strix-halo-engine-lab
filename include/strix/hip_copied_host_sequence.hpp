#pragma once
#include "strix/copied_host_channel.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/hip_distributed_sequence.hpp"

namespace strix {
struct HipCopiedHostChannels {
    // Four independently opened real project transports. A single underlying
    // stream cannot be shared by payload/step/token/greedy progress.
    std::unique_ptr<dist::Transport> payload,step,token,greedy;
    std::uint32_t rank=0;
    std::uint64_t epoch=0,maximum_frame_bytes=65536,maximum_payload_bytes=32768;
    ModelContentIdentity agreed_manifest;
    std::string agreed_physical_route;
};
struct HipCopiedHostSequenceRoute {
    HipDistributedSequenceTransport sequence;
    std::shared_ptr<DistributedControlByteStream> greedy;
    std::string transport_identity;
    std::string compiled_source_commit;
};
// Constructed only from actually opened project host transports (e.g.
// usb4_host_auto or tcp_listen/connect). No callback/mock provider substitutes
// for these owned native transports; the manifest must already be reconciled.
// The returned backend is an owning alias of the complete channel/state owner,
// so its borrowed backing remains live even if actual_owner is released first.
[[nodiscard]] HipCopiedHostSequenceRoute make_hip_copied_host_sequence_route(HipCopiedHostChannels);
} // namespace strix
