// Frozen original GLM process() trailer path, before the shared-core extraction.
    Trailer t;
    t.epoch = options_.epoch;
    t.plan = options_.plan;
    t.sequence = job.sequence;
    t.step = job.tag.step;
    t.bytes = static_cast<std::uint32_t>(job.bytes);
    t.kind = static_cast<std::uint16_t>(job.tag.kind);
    t.layer = job.tag.layer;
    t.rank = options_.rank;
    if (deferred) t.flags |= kTrailerDeferred;
    else t.crc32c = options_.crc ? dist::crc32c(dist::View{tx, job.bytes}) : 0U;
    if (prev_deferred_) {
        t.flags |= kTrailerPrev;
        t.prev_crc32c = prev_tx_crc_;
    }
    return t;
