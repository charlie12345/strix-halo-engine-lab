#include "strix/serve/replica_proxy.hpp"

#include "strix/serve/http_client.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <set>
#include <stdexcept>

namespace strix::serve {
namespace {

// Headers that describe one connection, not the message: never copied across the proxy.
bool hop_by_hop(std::string_view name) {
    static constexpr std::array<std::string_view, 10> names{"host", "connection", "content-length",
        "transfer-encoding", "keep-alive", "proxy-connection", "te", "upgrade", "expect", "trailer"};
    for (const auto n : names) if (n == name) return true;
    return false;
}

std::string error_body(std::string_view type, std::string_view message) {
    Json detail = Json::object();
    detail.set("type", Json(type));
    detail.set("message", Json(message));
    Json out = Json::object();
    out.set("error", std::move(detail));
    return out.dump();
}

std::string lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// One part of a numeric IPv4 host as inet_aton (and so getaddrinfo) reads it: 0x hex, a leading 0 octal, else decimal.
std::optional<std::uint64_t> ipv4_part(std::string_view text) {
    int base = 10;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        base = 16;
        text.remove_prefix(2);
    } else if (text.size() > 1 && text[0] == '0') {
        base = 8;
        text.remove_prefix(1);
    }
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return std::nullopt;
    return value;
}

// A numeric IPv4 host in any form inet_aton accepts (a.b.c.d, a.b.c, a.b, a; the last part fills the rest).
std::optional<std::uint32_t> parse_ipv4(std::string_view text) {
    std::array<std::uint64_t, 4> parts{};
    std::size_t count = 0;
    for (;;) {
        if (count == parts.size()) return std::nullopt;
        const auto dot = text.find('.');
        const auto part = ipv4_part(text.substr(0, dot));
        if (!part) return std::nullopt;
        parts[count++] = *part;
        if (dot == std::string_view::npos) break;
        text.remove_prefix(dot + 1);
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i + 1 < count; ++i) {
        if (parts[i] > 0xFFU) return std::nullopt;
        value |= parts[i] << (8U * (3U - i));
    }
    if (parts[count - 1] > (std::uint64_t{0xFFFFFFFFU} >> (8U * (count - 1U)))) return std::nullopt;
    return static_cast<std::uint32_t>(value | parts[count - 1]);
}

std::string dotted(std::uint32_t v) {
    return std::to_string(v >> 24U) + "." + std::to_string((v >> 16U) & 0xFFU) + "." +
           std::to_string((v >> 8U) & 0xFFU) + "." + std::to_string(v & 0xFFU);
}

using Ipv6 = std::array<std::uint16_t, 8>;

// The colon-separated groups of one side of an IPv6 literal; a dotted IPv4 tail (strict a.b.c.d) only when allowed.
bool ipv6_groups(std::string_view part, Ipv6& out, std::size_t& count, bool tail) {
    if (part.empty()) return true;
    for (;;) {
        const auto colon = part.find(':');
        const auto group = part.substr(0, colon);
        if (colon == std::string_view::npos && tail && group.find('.') != std::string_view::npos) {
            if (count + 2U > out.size() || std::count(group.begin(), group.end(), '.') != 3 ||
                group.find_first_not_of("0123456789.") != std::string_view::npos)
                return false;
            const auto v4 = parse_ipv4(group);
            if (!v4) return false;
            out[count++] = static_cast<std::uint16_t>(*v4 >> 16U);
            out[count++] = static_cast<std::uint16_t>(*v4 & 0xFFFFU);
            return true;
        }
        unsigned value = 0;
        const auto parsed = std::from_chars(group.data(), group.data() + group.size(), value, 16);
        if (group.empty() || group.size() > 4 || count == out.size() || parsed.ec != std::errc{} ||
            parsed.ptr != group.data() + group.size())
            return false;
        out[count++] = static_cast<std::uint16_t>(value);
        if (colon == std::string_view::npos) return true;
        part.remove_prefix(colon + 1);
    }
}

// A numeric IPv6 literal (RFC 4291 text forms: one "::" for one or more zero groups, an optional dotted IPv4 tail).
std::optional<Ipv6> parse_ipv6(std::string_view text) {
    Ipv6 head{}, tail{};
    std::size_t heads = 0, tails = 0;
    const auto gap = text.find("::");
    if (gap == std::string_view::npos) {
        if (!ipv6_groups(text, head, heads, true) || heads != head.size()) return std::nullopt;
        return head;
    }
    const auto right = text.substr(gap + 2);
    if (right.find("::") != std::string_view::npos || !ipv6_groups(text.substr(0, gap), head, heads, false) ||
        !ipv6_groups(right, tail, tails, true) || heads + tails > head.size() - 1U)
        return std::nullopt;
    std::copy_n(tail.begin(), tails, head.end() - static_cast<std::ptrdiff_t>(tails));
    return head;
}

} // namespace

std::string canonical_replica_host(std::string_view host) {
    if (host.size() > 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
    // a zone id (fe80::1%eth0, URI form %25eth0) names an interface: kept and compared as written
    std::string zone;
    const auto percent = host.find('%');
    if (percent != std::string_view::npos && host.find(':') != std::string_view::npos) {
        zone = std::string(host.substr(percent + 1));
        if (zone.size() > 2 && zone.starts_with("25")) zone.erase(0, 2);
        host = host.substr(0, percent);
    }
    if (zone.empty()) {
        if (const auto v4 = parse_ipv4(host)) return dotted(*v4);
    }
    if (const auto v6 = parse_ipv6(host)) {
        const auto& a = *v6;
        // an IPv4-mapped address (::ffff:a.b.c.d) reaches the IPv4 listener a.b.c.d
        if (zone.empty() && std::all_of(a.begin(), a.begin() + 5, [](std::uint16_t g) { return g == 0; }) &&
            a[5] == 0xFFFFU)
            return dotted((std::uint32_t{a[6]} << 16U) | a[7]);
        std::string out;
        for (std::size_t g = 0; g < a.size(); ++g) {
            std::array<char, 4> digits{};
            const auto r = std::to_chars(digits.data(), digits.data() + digits.size(), unsigned{a[g]}, 16);
            if (g != 0) out += ':';
            out.append(digits.data(), r.ptr);
        }
        return zone.empty() ? out : out + "%" + zone;
    }
    // a host name: case-insensitive, the root label's trailing dot dropped
    std::string name = lower(host);
    if (name.size() > 1 && name.back() == '.') name.pop_back();
    return zone.empty() ? name : name + "%" + zone;
}

std::optional<ReplicaEndpoint> parse_replica_endpoint(std::string_view text) {
    constexpr std::string_view scheme = "http://";
    if (text.starts_with(scheme)) text.remove_prefix(scheme.size());
    while (!text.empty() && text.back() == '/') text.remove_suffix(1);
    const auto colon = text.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 == text.size()) return std::nullopt;
    const auto digits = text.substr(colon + 1);
    unsigned port = 0;
    const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), port);
    if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || port == 0 || port > 65535)
        return std::nullopt;
    auto host = text.substr(0, colon);
    if (host.find('/') != std::string_view::npos) return std::nullopt;
    if (host.size() > 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
    return ReplicaEndpoint{std::string(host), static_cast<std::uint16_t>(port)};
}

void require_distinct_replicas(const std::vector<ReplicaEndpoint>& replicas) {
    std::set<std::pair<std::string, std::uint16_t>> seen;
    for (const auto& replica : replicas)
        if (!seen.emplace(canonical_replica_host(replica.host), replica.port).second)
            throw std::invalid_argument("--replica " + replica.host + ":" + std::to_string(replica.port) +
                                        " is given more than once; each replica must be a distinct worker");
}

ReplicaProxy::ReplicaProxy(ReplicaProxyConfig config)
    : config_(std::move(config)),
      router_(config_.replicas.size(), config_.affinity_entries, config_.block_bytes),
      states_(config_.replicas.size()), counters_(config_.replicas.size()) {}

void ReplicaProxy::refresh() {
    for (std::size_t r = 0; r < config_.replicas.size(); ++r) {
        HttpClientRequest request;
        request.host = config_.replicas[r].host;
        request.port = config_.replicas[r].port;
        request.target = config_.status_target;
        request.connect_timeout_ms = config_.connect_timeout_ms;
        request.idle_timeout_ms = config_.status_timeout_ms;
        std::string body;
        const auto result = http_fetch(request, body);
        ReplicaState next;
        if (result.ok && result.head.status == 200)
            if (auto parsed = replica_state_from_status(body)) next = *parsed;
        bool lost = false;
        {
            const std::lock_guard lock(mutex_);
            lost = states_[r].healthy && !next.healthy;
            next.ewma_ms_per_token = states_[r].ewma_ms_per_token;
            states_[r] = next;
        }
        if (lost) router_.forget(r);
    }
}

std::vector<ReplicaState> ReplicaProxy::states() const {
    const std::lock_guard lock(mutex_);
    return states_;
}

Json ReplicaProxy::status() const {
    const std::lock_guard lock(mutex_);
    Json replicas = Json::array();
    for (std::size_t r = 0; r < states_.size(); ++r) {
        const auto& s = states_[r];
        Json capacity = Json::object();
        capacity.set("slots_total", Json(s.capacity.slots_total));
        capacity.set("slots_busy", Json(s.capacity.slots_busy));
        capacity.set("kv_bytes_free", s.capacity.kv_bytes_free ? Json(*s.capacity.kv_bytes_free) : Json());
        capacity.set("queue_depth", Json(s.capacity.queue_depth));
        Json row = Json::object();
        row.set("index", Json(static_cast<std::uint64_t>(r)));
        row.set("endpoint", Json(config_.replicas[r].host + ":" + std::to_string(config_.replicas[r].port)));
        row.set("healthy", Json(s.healthy));
        row.set("capacity", std::move(capacity));
        row.set("inflight", Json(router_.inflight(r)));
        row.set("forwarded", Json(counters_[r].forwarded));
        row.set("failed", Json(counters_[r].failed));
        row.set("by_session", Json(counters_[r].by_session));
        row.set("by_prefix", Json(counters_[r].by_prefix));
        row.set("by_load", Json(counters_[r].by_load));
        replicas.as_array().push_back(std::move(row));
    }
    Json out = Json::object();
    out.set("replicas", std::move(replicas));
    out.set("retried_before_send", Json(retried_.load()));
    out.set("unplaced", Json(unplaced_.load()));
    return out;
}

void ReplicaProxy::handle(const HttpRequest& request, HttpResponseWriter& response) {
    if (request.method == "GET" && request.path == "/router/status") {
        response.send(200, "application/json", status().dump());
        return;
    }
    if (request.method == "GET" && (request.path == "/health" || request.path == "/v1/health")) {
        std::uint64_t healthy = 0;
        for (const auto& s : states()) healthy += s.healthy ? 1U : 0U;
        Json out = Json::object();
        out.set("status", Json(healthy ? "ok" : "unavailable"));
        out.set("replicas_healthy", Json(healthy));
        out.set("replicas_total", Json(static_cast<std::uint64_t>(config_.replicas.size())));
        response.send(healthy ? 200 : 503, "application/json", out.dump());
        return;
    }
    forward(request, response);
}

void ReplicaProxy::forward(const HttpRequest& request, HttpResponseWriter& response) {
    RouteKeyStorage storage;
    const auto route = route_request_from_http(request.path, request.header("x-session-id"), request.body, storage);
    std::uint64_t excluded = 0;
    for (;;) {
        const auto snapshot = states();
        const std::size_t replica = router_.place(route, snapshot, excluded);
        if (replica == kNoReplica) {
            ++unplaced_;
            response.send(503, "application/json", error_body("overloaded_error", "no healthy replica"),
                {{"Retry-After", "1"}});
            return;
        }
        const RouteReason reason = router_.last_reason();

        HttpClientRequest out;
        out.host = config_.replicas[replica].host;
        out.port = config_.replicas[replica].port;
        out.method = request.method;
        out.target = request.target.empty() ? request.path : request.target;
        out.body = request.body;
        out.connect_timeout_ms = config_.connect_timeout_ms;
        out.idle_timeout_ms = config_.idle_timeout_ms;
        for (const auto& h : request.headers) if (!hop_by_hop(h.first)) out.headers.push_back(h);

        // Streams (SSE, or any body without a length) pass through as they arrive; a body with a Content-Length is
        // delivered whole, so the client sees the same bytes and the same framing a replica would send.
        bool started = false, streaming = false, client_lost = false;
        int status = 0;
        std::string content_type, buffered;
        std::vector<std::pair<std::string, std::string>> headers;
        HttpClientSink sink;
        sink.cancelled = [&] { return response.client_gone(); };
        sink.on_head = [&](const HttpClientHead& head) {
            status = head.status;
            content_type = head.header("content-type");
            for (const auto& h : head.headers)
                if (!hop_by_hop(h.first) && h.first != "content-type" && h.first != "cache-control"
                    && h.first != "x-accel-buffering")
                    headers.push_back(h);
            headers.emplace_back("X-Strix-Replica", std::to_string(replica));
            streaming = head.header("content-length").empty() && head.status != 204 && head.status != 304
                && request.method != "HEAD";
            if (!streaming) {
                for (const auto& h : head.headers) if (h.first == "cache-control") headers.push_back(h);
                return true;
            }
            started = true;
            if (!response.begin_stream(status, content_type, headers)) { client_lost = true; return false; }
            return true;
        };
        sink.on_data = [&](std::string_view data) {
            if (!streaming) { buffered.append(data); return true; }
            if (!response.write(data)) { client_lost = true; return false; }
            return true;
        };
        const auto result = http_exchange(out, sink);
        router_.finished(replica, route);

        if (!result.ok && !result.cancelled && !result.request_sent) {
            // Nothing reached the replica: it cannot have started the request, so another replica may take it.
            {
                const std::lock_guard lock(mutex_);
                states_[replica].healthy = false;
                ++counters_[replica].failed;
            }
            excluded |= 1ULL << replica;
            ++retried_;
            continue;
        }
        {
            const std::lock_guard lock(mutex_);
            auto& c = counters_[replica];
            ++c.forwarded;
            if (reason == RouteReason::session) ++c.by_session;
            else if (reason == RouteReason::prefix) ++c.by_prefix;
            else ++c.by_load;
            if (!result.ok && !result.cancelled) ++c.failed;
        }
        if (streaming) {
            // A stream cut by the replica ends here; the missing terminator tells the client it is incomplete.
            if (started && !client_lost) response.end_stream();
            return;
        }
        if (result.ok) response.send(status, content_type.empty() ? "application/octet-stream" : content_type, buffered, headers);
        else if (!result.cancelled && !response.client_gone())
            response.send(502, "application/json",
                error_body("api_error", "replica " + std::to_string(replica) + " failed: " + result.error),
                {{"X-Strix-Replica", std::to_string(replica)}});
        return;
    }
}

} // namespace strix::serve
