#include "strix/serve/conversation_keep.hpp"

#include <algorithm>
#include <utility>

namespace strix::serve {

ConversationKeep::ConversationKeep(std::uint64_t budget, PrefixBlockHash hash)
    : budget_(budget), hash_(hash != nullptr ? hash : &prefix_block_hash) {}

std::uint64_t ConversationKeep::key_hash(const KeepIdentity& identity, std::span<const std::uint32_t> ids) const {
    const std::uint32_t word[2] = {static_cast<std::uint32_t>(identity.variant),
                                   static_cast<std::uint32_t>(identity.variant >> 32)};
    return hash_(hash_(identity.model, std::span<const std::uint32_t>(word, 2)), ids);
}

const KeepEntry* ConversationKeep::find(const KeepIdentity& identity, std::span<const std::uint32_t> prompt) const {
    const KeepEntry* best = nullptr;
    for (const auto& e : entries_) {
        if (!(e.identity == identity) || e.ids.size() >= prompt.size() || e.frontier >= prompt.size()) continue;
        if (best && e.ids.size() <= best->ids.size()) continue;
        const auto head = prompt.first(e.ids.size());
        if (key_hash(identity, head) == e.hash && std::equal(e.ids.begin(), e.ids.end(), head.begin())) best = &e;
    }
    return best;
}

const KeepEntry* ConversationKeep::lookup(const KeepIdentity& identity, std::span<const std::uint32_t> prompt) {
    const auto* e = find(identity, prompt);
    if (!e) { ++counters_.misses; return nullptr; }
    const auto k = static_cast<std::size_t>(e->kind);
    ++counters_.hits;
    ++counters_.hits_by_kind[k];
    counters_.hit_tokens[k] += e->frontier;
    return e;
}

const KeepEntry* ConversationKeep::get(std::uint64_t id) const {
    for (const auto& e : entries_) if (e.id == id) return &e;
    return nullptr;
}

bool ConversationKeep::prepare(KeepCandidate c) {
    pending_.reset();
    const auto bytes = charged_bytes(c.payload_bytes, c.ids.size());
    if (!budget_ || c.payload_bytes == 0 || c.id == 0 || get(c.id) || bytes > budget_) {
        ++counters_.refused;
        return false;
    }
    KeepEntry e;
    e.id = c.id;
    e.identity = c.identity;
    e.kind = c.kind;
    e.frontier = c.frontier;
    e.payload_bytes = c.payload_bytes;
    e.bytes = bytes;
    e.hash = key_hash(c.identity, c.ids);
    e.ids = std::move(c.ids);
    pending_ = std::move(e);
    return true;
}

std::uint64_t ConversationKeep::replacement_id() const {
    if (!pending_) return 0;
    for (const auto& e : entries_)
        if (e.hash == pending_->hash && e.identity == pending_->identity && e.ids == pending_->ids) return e.id;
    return 0;
}

bool ConversationKeep::victims(std::vector<std::uint64_t>& out) const {
    out.clear();
    if (!pending_) return false;
    std::uint64_t held = bytes_;
    if (const auto r = replacement_id()) {
        const auto* e = get(r);
        if (e->leases) return false;
        out.push_back(r);
        held -= e->bytes;
    }
    for (const auto& e : entries_) {
        if (held + pending_->bytes <= budget_) break;
        if (e.leases || std::find(out.begin(), out.end(), e.id) != out.end()) continue;
        out.push_back(e.id);
        held -= e.bytes;
    }
    return held + pending_->bytes <= budget_;
}

bool ConversationKeep::pending_fits() const {
    return pending_ && replacement_id() == 0 && bytes_ <= budget_ && budget_ - bytes_ >= pending_->bytes;
}

bool ConversationKeep::commit(bool accept) {
    const bool ok = accept && pending_fits();
    if (accept && !ok) ++counters_.refused;
    if (ok) {
        bytes_ += pending_->bytes;
        ++counters_.stores;
        ++counters_.stores_by_kind[static_cast<std::size_t>(pending_->kind)];
        entries_.push_back(std::move(*pending_));
    }
    pending_.reset();
    return ok;
}

bool ConversationKeep::store(KeepCandidate candidate) {
    if (!prepare(std::move(candidate))) return false;
    std::vector<std::uint64_t> v;
    if (!victims(v)) { pending_.reset(); ++counters_.refused; return false; }
    const auto replaced = replacement_id();
    for (const auto id : v) erase(id, id == replaced ? KeepEvict::replaced : KeepEvict::lru);
    return commit(true);
}

bool ConversationKeep::erase(std::uint64_t id, KeepEvict reason) {
    for (auto i = entries_.begin(); i != entries_.end(); ++i) {
        if (i->id != id) continue;
        bytes_ -= i->bytes;
        entries_.erase(i);
        ++counters_.evictions;
        if (reason == KeepEvict::pressure) ++counters_.pressure_evictions;
        return true;
    }
    return false;
}

void ConversationKeep::touch(std::uint64_t id) {
    for (auto i = entries_.begin(); i != entries_.end(); ++i)
        if (i->id == id) { entries_.splice(entries_.end(), entries_, i); return; }
}

void ConversationKeep::clear() {
    entries_.clear();
    pending_.reset();
    bytes_ = 0;
}

bool ConversationKeep::acquire(std::uint64_t id) {
    for (auto& e : entries_) if (e.id == id) { ++e.leases; return true; }
    return false;
}

void ConversationKeep::release(std::uint64_t id) {
    for (auto& e : entries_) if (e.id == id) { if (e.leases) --e.leases; return; }
}

std::vector<std::uint64_t> ConversationKeep::reclaim_plan(std::uint64_t target_bytes) const {
    std::vector<std::uint64_t> out;
    std::uint64_t freed = 0;
    for (const auto& e : entries_) {
        if (freed >= target_bytes) break;
        if (e.leases) continue;
        out.push_back(e.id);
        freed += e.bytes;
    }
    return out;
}

std::vector<std::uint64_t> ConversationKeep::shrink_plan() const {
    return bytes_ > budget_ ? reclaim_plan(bytes_ - budget_) : std::vector<std::uint64_t>{};
}

std::uint64_t ConversationKeep::oldest() const {
    for (const auto& e : entries_) if (!e.leases) return e.id;
    return 0;
}

KeepStats ConversationKeep::stats() const {
    KeepStats s = counters_;
    s.bytes = bytes_;
    s.entries = entries_.size();
    s.budget = budget_;
    return s;
}

}  // namespace strix::serve
