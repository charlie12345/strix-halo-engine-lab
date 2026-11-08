#pragma once
#include "output_split.hpp"
#include <charconv>
#include <map>
#include <string>
#include <vector>
namespace strix::qwen_split {
inline std::uint64_t integer(const std::string &text) {
    std::uint64_t n = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), n);
    require(result.ec == std::errc{} && result.ptr == text.data() + text.size() && !text.empty(),
            "invalid unsigned fixture integer");
    return n;
}
struct Options {
    std::map<std::string, std::string> args;
    bool enabled = false;
    explicit Options(int argc, char **argv) {
        const std::vector<std::string> unproved = {
            "--mtp",           "--draft",         "--mtp-sidecar", "--prefix-cache",
            "--session-cache", "--prefill-chunk", "--batch",       "--http-batch"};
        const std::vector<std::string> allowed = {
            "--rank",        "--world",         "--transport",
            "--model",       "--model-sha",     "--tensor",
            "--tensor-sha",  "--hidden",        "--hidden-sha",
            "--reference",   "--reference-sha", "--target-identity",
            "--inputs",      "--capacity",      "--position",
            "--epoch",       "--budget-bytes",  "--reserve-bytes",
            "--deadline-ms", "--scalar-leaf",   "--source-sha",
            "--out"};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--enable-output-row-split") {
                require(!enabled, "duplicate split opt-in");
                enabled = true;
                continue;
            }
            if (std::find(unproved.begin(), unproved.end(), key) != unproved.end()) {
                throw std::invalid_argument("UNPROVED_COMPOSITION_REFUSED: " + key);
            }
            require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(),
                    "unknown fixture option");
            require(i + 1 < argc && !args.contains(key), "missing/duplicate fixture argument");
            args.emplace(key, argv[++i]);
        }
        require(enabled, "OFF: explicit --enable-output-row-split is required");
    }
    std::string get(const std::string &name) const {
        const auto it = args.find(name);
        if (it == args.end()) {
            throw std::invalid_argument("missing required fixture binding: " + name);
        }
        return it->second;
    }
    std::uint64_t number(const std::string &name) const { return integer(get(name)); }
};
} // namespace strix::qwen_split
