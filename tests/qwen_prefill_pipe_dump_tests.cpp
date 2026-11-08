#include "strix/qwen_prefill_pair.hpp"
#include "strix/qwen_prefill_pipe_dump.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <chrono>

namespace {
void require(bool value) {
    if (!value) throw std::runtime_error("PP dump assertion failed");
}
}
int main() {
    try {
        using namespace strix::qwen_pp;
        // Direct execution needs the same setup as CTest. Own a separate
        // directory so concurrent invocations never share diagnostic files.
        const auto directory = std::filesystem::temp_directory_path() /
            ("strix-pp-dump-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
        } cleanup{directory};
#ifdef _WIN32
        require(_putenv_s("STRIX_QWEN_PF_PIPE_DUMP", directory.string().c_str()) == 0);
#else
        require(::setenv("STRIX_QWEN_PF_PIPE_DUMP", directory.string().c_str(), 1) == 0);
#endif
        require(dump_enabled());
        const std::array<std::byte, 1> letter{std::byte{0x61}};
        require(dump_hash(letter) == 0xaf63dc4c8601ec8cULL);
        const std::array<std::byte, 4> bytes{std::byte{0}, std::byte{0xff}, std::byte{0x80}, std::byte{0x61}};
        require(dump_hash(std::span<const std::byte>(bytes).subspan(2),
            dump_hash(std::span<const std::byte>(bytes).first(2))) == dump_hash(bytes));
        const auto name = dump_stem("split", 16384U, 16U) + "-save.seam.bin";
        dump_write(name, bytes);
        std::ifstream input(std::filesystem::path(std::getenv("STRIX_QWEN_PF_PIPE_DUMP")) / name, std::ios::binary);
        std::array<std::byte, 4> loaded{};
        input.read(reinterpret_cast<char*>(loaded.data()), static_cast<std::streamsize>(loaded.size()));
        require(input.gcount() == static_cast<std::streamsize>(loaded.size()) && loaded == bytes);
        require(input.peek() == std::char_traits<char>::eof());
        dump_text("metadata.txt", "wire=2\n");
        // A same-sized old payload has different semantics: reject it before consuming any row bytes.
        std::atomic<bool> aborted{false};
        std::size_t receives = 0;
        {
            Pipe consumer(1U, 1U, [](std::span<const std::byte>) {}, [&](std::span<std::byte> destination) {
                ++receives;
                auto legacy = PipeFrame{}.header;
                legacy[0] = 0x5050325345414dULL;
                legacy[3] = 16U; legacy[4] = 24U; legacy[41] = 1U;
                require(destination.size() == sizeof(legacy));
                std::memcpy(destination.data(), legacy.data(), sizeof(legacy));
            }, [&] { aborted.store(true); });
            bool rejected = false;
            try { (void)consumer.acquire(0U); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected);
        }
        require(aborted.load() && receives == 1U);
        std::cout << "PP dump hash, binary roundtrip and legacy wire rejection passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
