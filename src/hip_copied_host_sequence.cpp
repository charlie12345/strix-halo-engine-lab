#include "strix/hip_copied_host_sequence.hpp"
namespace strix {
HipCopiedHostSequenceRoute make_hip_copied_host_sequence_route(HipCopiedHostChannels c) {
    using Channel=CopiedHostByteChannel<dist::Transport>;
    struct Owners {
        std::shared_ptr<Channel> payload,step,token,greedy;
        std::shared_ptr<AllocationGenerationSource> generations;
        std::unique_ptr<transport::CopiedStreamConnectionState> connection;
        std::unique_ptr<transport::CopiedStreamTransportBackend> backend;
        ~Owners() {
            // Synchronous host I/O retains no caller buffers after return.
            // GPU consumers must be independently drained by the sequence.
            if(backend) {backend->poison();if(backend->drain_retained()!=transport::CopiedStreamCleanupStatus::released) std::terminate();}
        }
    };
    if(!c.payload || !c.step || !c.token || !c.greedy || c.agreed_physical_route.empty())
        throw std::invalid_argument("host-copy model route requires four actual transports and an agreed physical route");
    auto o=std::make_shared<Owners>();
    const auto channel=[&](std::unique_ptr<dist::Transport> t,std::uint64_t id) {
        return std::make_shared<Channel>(std::move(t),c.rank,c.epoch,id,c.agreed_manifest,c.maximum_frame_bytes);
    };
    o->payload=channel(std::move(c.payload),1);o->step=channel(std::move(c.step),2);
    o->token=channel(std::move(c.token),3);o->greedy=channel(std::move(c.greedy),4);
    o->generations=std::make_shared<AllocationGenerationSource>();
    o->connection=std::make_unique<transport::CopiedStreamConnectionState>(transport::CopiedStreamConnection{
        c.agreed_physical_route,c.epoch,c.rank,1U-c.rank,0U});
    o->backend=std::make_unique<transport::CopiedStreamTransportBackend>(*o->payload,*o->connection,
        transport::CopiedStreamTransportPolicy{c.maximum_payload_bytes,c.maximum_frame_bytes,4096});
    HipCopiedHostSequenceRoute result;
    result.sequence.actual_owner=o;result.sequence.step_control=o->step;result.sequence.token_control=o->token;
    // Every escaped backend handle independently retains the complete owner.
    // Keeping a plain backend shared_ptr would leave its borrowed state/channel
    // dangling after a caller drops actual_owner. The alias creates no cycle:
    // Owners owns the backend uniquely; only external handles own Owners.
    result.sequence.copied=std::shared_ptr<transport::CopiedStreamTransportBackend>(o,o->backend.get());
    result.sequence.copied_generations=o->generations;
    result.greedy=o->greedy;result.transport_identity=o->payload->transport_identity();
#ifdef STRIX_QWEN_COPIED_ROUTE_SOURCE
    result.compiled_source_commit=STRIX_QWEN_COPIED_ROUTE_SOURCE;
#else
    result.compiled_source_commit="UNBOUND_NONNATIVE_OR_NONPROJECT_BUILD";
#endif
    return result;
}
} // namespace strix
