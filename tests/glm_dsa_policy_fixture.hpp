// Host emulation of the shared lane-token / tree256 softmax and FMA PV family.
// Included by the anomaly fixture after tree() and the namespace aliases.
template <bool Partial>
O::Partial policy_attention(const std::vector<O::Row>& rows, unsigned rank) {
    using Policy = strix::glm::fast::DsaAttentionPolicy<Partial>;
    const strix::glm::fast::DsaPartialContext context{rank, nullptr, nullptr};
    auto owns = [&](std::uint32_t token) {
        if constexpr (Partial) return Policy::owns(token, context);
        else return Policy::owns(token);
    };
    auto address = [&](std::uint32_t token) {
        if constexpr (Partial) return Policy::address(token, context);
        else return Policy::address(token);
    };
    O::Partial result;
    result.a.resize(512, 0.0F);
    std::map<std::uint32_t, const O::Row*> cache;
    for (const auto& row : rows) {
        const auto token = static_cast<std::uint32_t>(row.token);
        if (owns(token)) {
            require(address(token) == (Partial ? K::local_token(token) : token), "policy cache address");
            require(cache.emplace(address(token), &row).second, "policy cache address collision");
            result.m = std::max(result.m, row.score);
        }
    }
    std::vector<float> weights(rows.size(), 0.0F);
    std::array<float, 256> lanes{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto token = static_cast<std::uint32_t>(rows[i].token);
        const float score = owns(token) ? rows[i].score : -std::numeric_limits<float>::infinity();
        weights[i] = Policy::weight(score, result.m);
        lanes[i % 256] = lanes[i % 256] + weights[i];
    }
    result.l = tree(lanes);
    for (auto& weight : weights) weight = Policy::probability(weight, result.l);
    for (unsigned dim = 0; dim < 512; ++dim) {
        lanes.fill(0.0F);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto token = static_cast<std::uint32_t>(rows[i].token);
            if (owns(token)) lanes[i % 256] = std::fma(weights[i], cache.at(address(token))->value[dim], lanes[i % 256]);
        }
        result.a[dim] = tree(lanes);
        require(Policy::output_index(3, dim) == 3 * (Partial ? 514U : 512U) + (Partial ? 2U : 0U) + dim,
                "policy partial output stride");
    }
    return result;
}

void policy_empty_ranks() {
    for (unsigned rank = 0; rank < 2; ++rank) {
        const auto empty = policy_attention<true>({}, rank);
        require(empty.m == -std::numeric_limits<float>::infinity() && empty.l == 0, "empty partial header");
        for (const auto value : empty.a) require(value == 0, "empty partial accumulator");
    }
}

void policy_stress() {
    // Irregular selected IDs, partial waves, maximum selection, opposite score
    // maxima on the ranks, and underflow; two independent query-score patterns.
    for (const unsigned count : {1U, 4U, 5U, 255U, 256U, 257U, 1024U, 2051U}) {
        for (unsigned query = 0; query < 2; ++query) {
            std::vector<O::Row> rows;
            std::array<std::vector<O::Row>, 2> owned;
            for (unsigned i = 0; i < count; ++i) {
                O::Row row{};
                row.id = row.token = (i / 4) * 12 + i % 4;
                const float rank_bias = (K::owner(row.token) == query) ? 0.0F : -104.0F;
                row.score = rank_bias - static_cast<float>((i * 31 + query * 7) % 101) / 8.0F;
                row.value.resize(512);
                for (unsigned dim = 0; dim < 512; ++dim)
                    row.value[dim] = static_cast<float>(static_cast<int>((i * 11 + dim * 13) % 71) - 35) / 16.0F;
                owned[K::owner(row.token)].push_back(row);
                rows.push_back(std::move(row));
            }
            const auto a = policy_attention<true>(rows, 0), b = policy_attention<true>(rows, 1);
            const auto replica = policy_attention<false>(rows, 0);
            const auto oracle = O::normalize(O::merge_rank0_rank1(O::accumulate(owned[0], 512), O::accumulate(owned[1], 512)));
            for (unsigned dim = 0; dim < 512; ++dim) {
                const float merged = K::merge_component(a.m, a.l, a.a[dim], b.m, b.l, b.a[dim]);
                require(std::isfinite(merged) && std::abs(merged - oracle[dim]) <= 1e-5F * (1.0F + std::abs(oracle[dim])),
                        "irregular/underflow partial oracle");
                require(std::abs(merged - replica.a[dim]) <= 1e-5F * (1.0F + std::abs(replica.a[dim])),
                        "irregular/underflow partial versus replica");
            }
        }
    }
}
