#include "strix/device_profile.hpp"
#include "strix/hip_arch_caps.hpp"

#include <cstdlib>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void set_profile_env(const char* value) {
#ifdef _WIN32
    require(_putenv_s("STRIX_PRINT_DEVICE_PROFILE", value) == 0, "set environment");
#else
    require(setenv("STRIX_PRINT_DEVICE_PROFILE", value, 1) == 0, "set environment");
#endif
}
struct GroupedNumbers : std::numpunct<char> {
    char do_thousands_sep() const override { return ','; }
    std::string do_grouping() const override { return "\3"; }
};
} // namespace

int main() try {
    static_assert(!STRIX_HAS_RDNA35_WMMA && !STRIX_HAS_WMMA_BF16_W32 && !STRIX_HAS_DPP_ROW && !STRIX_HAS_SUDOT4);
    for (const auto arch : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153"}) {
        const auto caps = strix::device_isa_capabilities(arch);
        require(caps.wmma_bf16_w32 && caps.dpp_row && caps.sudot4, "family ISA flags");
        require(caps.rdna35_wmma == (std::string(arch).substr(0, 6) == "gfx115"), "packed WMMA ABI");
    }
    for (const auto arch : {"", "gfx115", "gfx11510", "gfx1200", "gfx900", "unknown"}) {
        const auto caps = strix::device_isa_capabilities(arch);
        require(!caps.rdna35_wmma && !caps.wmma_bf16_w32 && !caps.dpp_row && !caps.sudot4, "unknown ISA fails closed");
    }
    require(strix::device_isa_capabilities("gfx1152:xnack-").rdna35_wmma, "feature suffix");
    strix::DeviceProfile profile;
    profile.arch_name = "gfx1151";
    profile.cu_count = 40;
    profile.wave_size = 32;
    profile.lds_bytes = 65536;
    profile.total_memory_bytes = 192ULL << 30;
    profile.free_memory_bytes = 160ULL << 30;
    profile.isa = strix::device_isa_capabilities(profile.arch_name);
    std::locale::global(std::locale(std::locale::classic(), new GroupedNumbers));
    require(strix::format_device_profile(profile) ==
        "{\"schema\":\"strix-device-profile-v1\",\"arch_name\":\"gfx1151\",\"device_index\":0,\"cu_count\":40,\"wave_size\":32,"
        "\"lds_bytes\":65536,\"total_memory_bytes\":206158430208,\"free_memory_bytes\":171798691840,"
        "\"capability_source\":\"architecture-family-isa\",\"capabilities\":{\"rdna35_wmma\":true,\"wmma_bf16_w32\":true,"
        "\"dpp_row\":true,\"sudot4\":true},\"bandwidth\":null,\"qualification\":\"not-assessed\"}\n", "stable locale-independent JSON");
    profile.arch_name = "a\"\\\n\t";
    profile.arch_name.push_back(static_cast<char>(0x80));
    require(strix::format_device_profile(profile).find("a\\\"\\\\\\u000a\\u0009\\u0080") != std::string::npos, "JSON escaping");
    set_profile_env("0");
    const char* plain[] = {"app"};
    require(!strix::print_device_profile_if_requested(1, plain), "default performs no HIP work");
    set_profile_env("true");
    require(!strix::print_device_profile_if_requested(1, plain), "only literal 1 opts in");
    set_profile_env("1");
    const char* effective[] = {"app", "--print-effective-config"};
    require(!strix::print_device_profile_if_requested(2, effective), "effective config remains host-only");
    const char* both[] = {"app", "--print-device-profile", "--print-effective-config"};
    bool refused = false;
    try { (void)strix::print_device_profile_if_requested(3, both); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused, "explicit conflicting inspection refused before HIP");
    refused = false;
    try { (void)strix::print_device_profile_if_requested(1, plain); }
    catch (const std::runtime_error& error) { refused = std::string(error.what()).find("HIP support is not built") != std::string::npos; }
    require(refused, "environment opts into probe and errors propagate");
    set_profile_env("0");
    const char* explicit_probe[] = {"app", "--print-device-profile"};
    refused = false;
    try { (void)strix::print_device_profile_if_requested(2, explicit_probe); }
    catch (const std::runtime_error&) { refused = true; }
    require(refused, "standalone option opts into probe");
    std::cout << "PASS: ISA mapping, 192 GiB JSON, escaping, locale, opt-in and host-only inspection\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
