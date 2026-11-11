#pragma once

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

// The one place that reads switches, numbers and choices from the environment. Unset and empty always mean "use the
// default"; a number, byte count or choice that is set but not understood throws std::invalid_argument naming the
// variable instead of being silently read as some default (N=12x, MODE=Fast).
namespace strix {

// Most flags compare the entire value. Some legacy readers compare only byte 0;
// keep that choice explicit ("0tail" and "1tail" otherwise change meaning).
enum class EnvFlagMatch { exact, first_character };

namespace detail {
inline bool env_flag_matches(const char* value, const char* flag, EnvFlagMatch match) {
    return value && (match == EnvFlagMatch::first_character ? value[0] == flag[0] : std::strcmp(value, flag) == 0);
}
[[noreturn]] inline void env_reject(const char* name, const char* value, const std::string& why) {
    throw std::invalid_argument(std::string(name) + "=" + value + ": " + why);
}
}  // namespace detail

// Unset and empty are ON; only "0" (or a leading '0' when selected) disables.
inline bool env_flag_default_on(const char* name, EnvFlagMatch match = EnvFlagMatch::exact) {
    return !detail::env_flag_matches(std::getenv(name), "0", match);
}

// Unset and empty are OFF; only "1" (or a leading '1' when selected) enables.
inline bool env_flag_default_off(const char* name, EnvFlagMatch match = EnvFlagMatch::exact) {
    return detail::env_flag_matches(std::getenv(name), "1", match);
}

// A number in [lo, hi]; unset or empty gives `fallback` (not range checked: a default may be a sentinel). Integers
// are decimal (no octal or hex surprises), floating point anything strtold reads; the whole value must parse.
template <class T>
T env_number(const char* name, T fallback, T lo = std::numeric_limits<T>::lowest(), T hi = std::numeric_limits<T>::max()) {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "env_number: arithmetic type");
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    if (std::isspace(static_cast<unsigned char>(value[0]))) detail::env_reject(name, value, "leading space");
    const auto range = [&] { return "out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]"; };
    char* end = nullptr;
    errno = 0;
    if constexpr (std::is_floating_point_v<T>) {
        const long double v = std::strtold(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(v)) detail::env_reject(name, value, "not a number");
        if (v < static_cast<long double>(lo) || v > static_cast<long double>(hi)) detail::env_reject(name, value, range());
        return static_cast<T>(v);
    } else if constexpr (std::is_signed_v<T>) {
        const long long v = std::strtoll(value, &end, 10);
        if (end == value || *end != '\0' || errno == ERANGE) detail::env_reject(name, value, "not an integer");
        if (v < static_cast<long long>(lo) || v > static_cast<long long>(hi)) detail::env_reject(name, value, range());
        return static_cast<T>(v);
    } else {
        if (value[0] == '-') detail::env_reject(name, value, "negative");
        const unsigned long long v = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || errno == ERANGE) detail::env_reject(name, value, "not an integer");
        if (v < static_cast<unsigned long long>(lo) || v > static_cast<unsigned long long>(hi))
            detail::env_reject(name, value, range());
        return static_cast<T>(v);
    }
}

// A byte count: a decimal integer with an optional binary suffix K, M, G or T (also KB / KiB ...), in [lo, hi].
inline std::uint64_t env_bytes(const char* name, std::uint64_t fallback, std::uint64_t lo = 0,
                               std::uint64_t hi = std::numeric_limits<std::uint64_t>::max()) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    if (value[0] == '-' || std::isspace(static_cast<unsigned char>(value[0]))) detail::env_reject(name, value, "not a byte count");
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(value, &end, 10);
    if (end == value || errno == ERANGE) detail::env_reject(name, value, "not a byte count");
    unsigned shift = 0;
    switch (*end) {
    case '\0': break;
    case 'k': case 'K': shift = 10; ++end; break;
    case 'm': case 'M': shift = 20; ++end; break;
    case 'g': case 'G': shift = 30; ++end; break;
    case 't': case 'T': shift = 40; ++end; break;
    default: detail::env_reject(name, value, "not a byte count");
    }
    if (shift != 0U && (std::strcmp(end, "iB") == 0 || std::strcmp(end, "B") == 0)) end += std::strlen(end);
    if (*end != '\0') detail::env_reject(name, value, "not a byte count");
    if (shift != 0U && v > (std::numeric_limits<std::uint64_t>::max() >> shift)) detail::env_reject(name, value, "too large");
    const std::uint64_t bytes = static_cast<std::uint64_t>(v) << shift;
    if (bytes < lo || bytes > hi)
        detail::env_reject(name, value, "out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "] bytes");
    return bytes;
}

// The index of the value among `choices` (exact, case-sensitive); unset or empty gives `fallback`.
inline std::size_t env_choice(const char* name, std::size_t fallback, std::initializer_list<const char*> choices) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    std::size_t index = 0;
    std::string allowed;
    for (const char* choice : choices) {
        if (std::strcmp(value, choice) == 0) return index;
        allowed += (index++ == 0 ? "" : ", ") + std::string(choice);
    }
    detail::env_reject(name, value, "expected one of " + allowed);
}

}  // namespace strix
