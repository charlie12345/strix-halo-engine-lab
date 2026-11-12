#include "strix/device/declaration.hpp"

#include "strix/model_description.hpp"

#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace strix::device {
namespace {
using Value = model_description::Value;
using Table = Value::Table;

constexpr std::array<std::string_view, 7> roles_known{"pair", "decode", "prefill", "draft", "replica", "front", "experts"};
constexpr std::array<std::string_view, 3> roles_rated{"prefill", "front", "experts"};

[[noreturn]] void bad(const std::string& key, const std::string& what) { throw std::invalid_argument(key + ": " + what); }

std::string lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}
bool hex(char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }

// The key is taken by value: a reference parameter bound to a temporary makes GCC flag the returned reference.
const Table& table(const Value& v, std::string_view key) {
    if (const auto* t = std::get_if<Table>(&v.data)) return *t;
    bad(std::string(key), "expected a table");
}
std::string string(const Value& v, const std::string& key) {
    if (const auto* s = std::get_if<std::string>(&v.data)) return *s;
    bad(key, "expected a string");
}
bool boolean(const Value& v, const std::string& key) {
    if (const auto* b = std::get_if<bool>(&v.data)) return *b;
    bad(key, "expected true or false");
}
std::uint64_t integer(const Value& v, const std::string& key) {
    if (const auto* n = std::get_if<std::int64_t>(&v.data); n && *n >= 0) return static_cast<std::uint64_t>(*n);
    bad(key, "expected a nonnegative integer");
}
double rate(const Value& v, const std::string& key) {
    double out = -1.0;
    if (const auto* d = std::get_if<double>(&v.data)) out = *d;
    else if (const auto* n = std::get_if<std::int64_t>(&v.data)) out = static_cast<double>(*n);
    if (!std::isfinite(out) || out < 0.0) bad(key, "expected a nonnegative number");
    return out;
}
void only(const Table& t, const std::set<std::string_view>& allowed, const std::string& prefix) {
    for (const auto& entry : t)
        if (!allowed.contains(entry.first)) bad(prefix + entry.first, "unknown key");
}
const Value* find(const Table& t, std::string_view key) {
    const auto it = t.find(key);
    return it == t.end() ? nullptr : &it->second;
}

Attach parse_attach(const Value& v, const std::string& key) {
    const auto& t = table(v, key);
    only(t, {"kind", "pcie_gen", "pcie_width", "h2d_gbytes_ps", "gbps_measured", "rtt_us_measured"}, key + ".");
    Attach a;
    const auto* kind = find(t, "kind");
    if (!kind) bad(key + ".kind", "required");
    a.kind = string(*kind, key + ".kind");
    if (a.kind != "integrated" && a.kind != "pcie" && a.kind != "oculink" && a.kind != "remote")
        bad(key + ".kind", "expected integrated, pcie, oculink or remote");
    if (const auto* x = find(t, "pcie_gen")) a.pcie_gen = static_cast<std::uint32_t>(integer(*x, key + ".pcie_gen"));
    if (const auto* x = find(t, "pcie_width")) a.pcie_width = static_cast<std::uint32_t>(integer(*x, key + ".pcie_width"));
    if (a.pcie_gen > 7 || a.pcie_width > 32) bad(key, "pcie_gen or pcie_width out of range");
    if (const auto* x = find(t, "h2d_gbytes_ps")) a.h2d_gbytes_ps = rate(*x, key + ".h2d_gbytes_ps");
    if (const auto* x = find(t, "gbps_measured")) a.gbps_measured = rate(*x, key + ".gbps_measured");
    if (const auto* x = find(t, "rtt_us_measured")) a.rtt_us_measured = integer(*x, key + ".rtt_us_measured");
    return a;
}

Descriptor parse_device(const std::string& id, const Value& v) {
    const std::string key = "devices." + id;
    const auto& t = table(v, key);
    only(t, {"vendor", "arch", "uuid", "pci", "name", "memory_bytes", "attach", "roles", "excluded", "reason"}, key + ".");
    Descriptor d;
    d.id = id;
    const auto* vendor = find(t, "vendor");
    if (!vendor) bad(key + ".vendor", "required");
    d.vendor = parse_vendor(string(*vendor, key + ".vendor"));
    if (const auto* x = find(t, "arch")) d.arch = lower(string(*x, key + ".arch"));
    try {
        if (const auto* x = find(t, "uuid")) d.uuid = normalise_uuid(string(*x, key + ".uuid"));
        if (const auto* x = find(t, "pci")) d.pci = normalise_pci(string(*x, key + ".pci"));
    } catch (const std::invalid_argument& e) {
        bad(key, e.what());
    }
    if (d.uuid.empty() && d.pci.empty()) bad(key, "needs an identity: uuid or pci (an index is not an identity)");
    if (const auto* x = find(t, "name")) d.name = string(*x, key + ".name");
    if (const auto* x = find(t, "memory_bytes")) d.memory_bytes = integer(*x, key + ".memory_bytes");
    if (const auto* x = find(t, "attach")) d.attach = parse_attach(*x, key + ".attach");
    if (const auto* x = find(t, "roles")) {
        const auto* list = std::get_if<Value::Array>(&x->data);
        if (!list) bad(key + ".roles", "expected an array of strings");
        for (const auto& item : *list) {
            auto role = string(item, key + ".roles");
            if (!known_role(role)) bad(key + ".roles", "unknown role " + role);
            for (const auto& seen : d.roles)
                if (seen == role) bad(key + ".roles", "duplicate role " + role);
            d.roles.push_back(std::move(role));
        }
    }
    if (const auto* x = find(t, "excluded")) d.excluded = boolean(*x, key + ".excluded");
    if (const auto* x = find(t, "reason")) d.reason = string(*x, key + ".reason");
    return d;
}

Policy parse_policy(const std::string& name, const Value& v) {
    const std::string key = "device_policies." + name;
    const auto& t = table(v, key);
    only(t, {"name_contains", "vendor", "excluded", "reason"}, key + ".");
    Policy p;
    p.name = name;
    const auto* needle = find(t, "name_contains");
    if (!needle) bad(key + ".name_contains", "required");
    p.name_contains = lower(string(*needle, key + ".name_contains"));
    if (p.name_contains.empty()) bad(key + ".name_contains", "must not be empty");
    if (const auto* x = find(t, "vendor")) p.vendor = std::string(vendor_name(parse_vendor(string(*x, key + ".vendor"))));
    if (const auto* x = find(t, "excluded")) p.excluded = boolean(*x, key + ".excluded");
    // A policy can only deny: an "include" rule would let a name pattern override a declared exclusion.
    if (!p.excluded) bad(key + ".excluded", "a policy can only exclude");
    if (const auto* x = find(t, "reason")) p.reason = string(*x, key + ".reason");
    return p;
}
} // namespace

std::string_view vendor_name(Vendor vendor) noexcept {
    switch (vendor) {
    case Vendor::amd: return "amd";
    case Vendor::nvidia: return "nvidia";
    case Vendor::intel: return "intel";
    }
    return "unknown";
}

Vendor parse_vendor(std::string_view text) {
    const auto v = lower(text);
    if (v == "amd") return Vendor::amd;
    if (v == "nvidia") return Vendor::nvidia;
    if (v == "intel") return Vendor::intel;
    throw std::invalid_argument("vendor: expected amd, nvidia or intel, got " + std::string(text));
}

std::string normalise_uuid(std::string_view text) {
    auto v = lower(text);
    if (v.rfind("gpu-", 0) == 0) v.erase(0, 4);
    std::string digits;
    for (const char c : v) {
        if (c == '-') continue;
        if (!hex(c)) throw std::invalid_argument("uuid: unexpected character in " + std::string(text));
        digits.push_back(c);
    }
    if (digits.size() != 32) throw std::invalid_argument("uuid: expected 32 hex digits in " + std::string(text));
    if (digits.find_first_not_of('0') == std::string::npos) throw std::invalid_argument("uuid: all zero is not an identity");
    return digits.substr(0, 8) + "-" + digits.substr(8, 4) + "-" + digits.substr(12, 4) + "-" + digits.substr(16, 4) + "-" +
           digits.substr(20, 12);
}

std::string normalise_pci(std::string_view text) {
    auto v = lower(text);
    const auto fail = [&] { throw std::invalid_argument("pci: expected [dddd:]bb:dd.f, got " + std::string(text)); };
    if (v.size() == 7) v.insert(0, "0000:");
    // Some tools print an 8-digit domain.
    if (v.size() == 16 && v.rfind("0000", 0) == 0) v.erase(0, 4);
    if (v.size() != 12 || v[4] != ':' || v[7] != ':' || v[10] != '.') fail();
    for (const std::size_t i : {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U})
        if (!hex(v[i])) fail();
    if (v[11] < '0' || v[11] > '7') fail();
    return v;
}

bool known_role(std::string_view role) noexcept {
    for (const auto known : roles_known)
        if (known == role) return true;
    return false;
}

std::string role_refusal(const Descriptor& device, std::string_view role) {
    bool has = false;
    for (const auto& r : device.roles) has = has || r == role;
    if (!has) return "role " + std::string(role) + " is not declared for device " + device.id;
    bool rated = false;
    for (const auto r : roles_rated) rated = rated || r == role;
    if (!rated || device.attach.kind == "integrated") return {};
    const auto unprobed = [&](const char* field) {
        return "role " + std::string(role) + " needs a measured rate: attach." + field + " of device " + device.id + " is not probed";
    };
    if (device.attach.kind.empty()) return "role " + std::string(role) + " needs a measured rate: device " + device.id + " declares no attach";
    if (device.attach.kind == "remote") return device.attach.gbps_measured > 0.0 ? std::string{} : unprobed("gbps_measured");
    return device.attach.h2d_gbytes_ps > 0.0 ? std::string{} : unprobed("h2d_gbytes_ps");
}

Declaration parse_declaration(std::string_view toml, std::string source) {
    const auto parsed = model_description::parse(toml);
    const auto& root = parsed.fields;
    only(root, {"schema", "machine", "devices", "device_policies"}, "");
    const auto* schema = find(root, "schema");
    if (!schema) bad("schema", "required (strix-devices-v1)");
    if (string(*schema, "schema") != "strix-devices-v1") bad("schema", "expected strix-devices-v1");
    Declaration out;
    out.source = std::move(source);
    if (const auto* x = find(root, "machine")) out.machine = string(*x, "machine");
    if (const auto* x = find(root, "devices"))
        for (const auto& [id, value] : table(*x, "devices")) out.devices.push_back(parse_device(id, value));
    if (const auto* x = find(root, "device_policies"))
        for (const auto& [name, value] : table(*x, "device_policies")) out.policies.push_back(parse_policy(name, value));
    // Two entries with one identity would make "which entry wins" depend on file order: refuse the file.
    for (std::size_t i = 0; i < out.devices.size(); ++i)
        for (std::size_t j = i + 1; j < out.devices.size(); ++j) {
            const auto& a = out.devices[i];
            const auto& b = out.devices[j];
            if (!a.uuid.empty() && a.uuid == b.uuid) bad("devices." + b.id + ".uuid", "same identity as device " + a.id);
            if (!a.pci.empty() && a.pci == b.pci) bad("devices." + b.id + ".pci", "same identity as device " + a.id);
        }
    return out;
}

Declaration load_declaration(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read device declaration " + path);
    std::ostringstream text;
    text << in.rdbuf();
    try {
        return parse_declaration(text.str(), path);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(path + ": " + e.what());
    }
}

} // namespace strix::device
