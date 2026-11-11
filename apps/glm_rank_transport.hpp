#pragma once
// TP2 transport and Communicator setup shared by strix-glm5-rank and strix-glm5-serve.
#include "strix/dist/communicator.hpp"
#include "strix/dist/open_transport.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix::glm::rank_transport {

// GPU-direct exchange opened by open_transport("usb4gpu"), consumed by make_communicator.
inline std::unique_ptr<strix::dist::DeviceExchange>& gpu_direct() {
    static std::unique_ptr<strix::dist::DeviceExchange> direct;
    return direct;
}

// The MiniMax rank app's transport setup (usb4-auto host path, usb4gpu GPU-direct when both ranks agree, tcp).
inline std::unique_ptr<strix::dist::Transport> open_transport(const std::string& spec, std::uint32_t rank, std::string& receipt) {
    strix::dist::OpenTransportOptions options;
    options.spellings = strix::dist::hyphen_transport;
    options.usb.rank = rank;
    options.usb.timeout = std::chrono::seconds(180);
    auto opened = strix::dist::open_transport(spec, options);
    gpu_direct() = std::move(opened.direct);
    receipt = std::move(opened.receipt);
    return std::move(opened.host);
}

// The engine's Communicator: one exchange carries up to max(8, exchange_rows) rows of 4096 floats.
inline std::unique_ptr<strix::dist::Communicator> make_communicator(std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport,
                                                                    unsigned exchange_rows) {
    strix::dist::Group group;
    group.rank = rank;
    group.world = 2;
    group.plan = 0x474c4d35ULL;  // "GLM5"
    const std::size_t max_bytes = std::size_t{std::max(8U, exchange_rows)} * 4096 * sizeof(float) + 4096;
    auto& direct = gpu_direct();
    return direct ? std::make_unique<strix::dist::Communicator>(group, std::move(transport), std::move(direct), max_bytes)
                  : std::make_unique<strix::dist::Communicator>(group, std::move(transport), max_bytes);
}

}  // namespace strix::glm::rank_transport
