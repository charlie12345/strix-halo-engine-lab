#include "session_replay/legacy_reference.hpp"
#include <condition_variable>
#include <future>
#include <thread>

using namespace session_fixture;
namespace {
// A fault-injection model for the mixed result contract. Production families
// supply native planners; this fixture deliberately reuses the scripted state.
struct MixedFixture final : ss::Engine {
    FakeEngine model;
    bool lane_error = false;
    std::function<void()> after_first;
    void prefill(std::span<const std::uint32_t> t,ss::KvHandle& h) override {model.prefill(t,h);}
    ss::StepResult decode(ss::KvHandle& h,const ss::SamplerState& s) override {return model.decode(h,s);}
    ss::Capabilities caps() const override {auto c=model.caps();c.mixed_advance=true;return c;}
    strix::profile::ExecutionReceipt receipt() const override {return model.receipt();}
    std::vector<ss::AdvanceResult> advance(std::span<const ss::AdvanceItem> items) override {
        std::vector<ss::AdvanceResult> results;
        for(std::size_t i=0;i<items.size();++i){const auto& item=items[i];ss::AdvanceResult r;
            r.output.before={model.store.consumed(item.state),model.store.pending(item.state)};
            if(item.cancelled && item.cancelled()) {r.output.after=r.output.before;r.output.rng_draws=item.sampler.rng_draws;r.output.finish=ss::FinishReason::cancelled;}
            else {
                const auto before=r.output.before;model.prefill(item.prompt,item.state);r.input_tokens=item.prompt.size();
                r.output=model.decode(item.state,item.sampler);r.output.before=before;
                if(lane_error && i==0)r.error="injected lane failure after commit";
            }
            results.push_back(std::move(r));if(i==0 && after_first)after_first();
        }return results;
    }
};
sv::SessionHooks mixed_hooks() {
    sv::SessionHooks h;h.schedule=[](std::span<const sv::SessionView> views){sv::SessionWork w{sv::SessionPhase::mixed,{},{}};
        for(std::size_t i=0;i<views.size();++i)w.lanes.push_back(i);return std::vector<sv::SessionWork>{w};};return h;
}
}
int main() {
    try {
        {
            FakeEngine e; sv::GenerationQueue q;
            sv::SessionRuntime r(q, e, e.store, {}, runtime_hooks(e));
            auto cold = execute(r, q, job());
            CHECK(cold.error.empty() && cold.ids == std::vector<std::uint32_t>({65, 66, 0}));
            CHECK(cold.stats.finish == "stop" && cold.stats.cached_tokens == 0 && e.releases == 1);
            CHECK(r.cache_entries() == 1 && e.captures == 1);
            auto j = job({10, 11, 12, 13}); j.cache_retention = sv::CacheRetention::no_store;
            const auto captures = e.captures;
            auto warm = execute(r, q, j);
            CHECK(warm.error.empty() && warm.stats.cached_tokens == 3);
            CHECK(e.suffixes.back() == std::vector<std::uint32_t>({13}));
            CHECK(e.captures == captures + 1 && r.cache_entries() == 1); // restore rollback capture only
            CHECK(e.restores == 1 && e.physical[0].rows.empty());
            auto mismatch = execute(r, q, job({20, 21}));
            CHECK(mismatch.stats.cached_tokens == 0 && e.initializes == 3);
        }
        {
            FakeEngine e; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto a = execute(r, q, job()); auto b = execute(r, q, job());
            CHECK(a.error.empty() && b.stats.cached_tokens == 0 && e.captures == 0 && e.restores == 0);
        }
        {
            FakeEngine e; sv::GenerationQueue q;
            sv::SessionRuntime r(q, e, e.store, {}, runtime_hooks(e));
            auto j = job(); j.cache_retention = sv::CacheRetention::no_store;
            CHECK(execute(r, q, j).error.empty()); CHECK(e.captures == 0 && r.cache_entries() == 0);
            CHECK(execute(r, q, job()).error.empty());
            e.fail_restore = true;
            auto bad = execute(r, q, job());
            CHECK(!bad.error.empty() && bad.stats.cached_tokens == 0 && bad.ids.empty());
            e.fail_restore = false;
            CHECK(execute(r, q, job()).error.empty());
        }
        {
            FakeEngine e; sv::GenerationQueue q; e.bad_capture = true;
            sv::SessionRuntime r(q, e, e.store, {}, runtime_hooks(e));
            CHECK(!execute(r, q, job()).error.empty()); CHECK(r.cache_entries() == 0 && e.releases == 1);
            e.bad_capture = false; CHECK(execute(r, q, job()).error.empty());
        }
        {
            FakeEngine e; sv::GenerationQueue q; auto h = runtime_hooks(e);
            ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, h);
            auto a = q.submit(job({10}, 2)); auto b = q.submit(job({20}, 2));
            r.step(); CHECK(r.live() == 2 && e.batch_calls == 0);
            r.step(); auto x = drain(a); auto y = drain(b);
            CHECK(x.ids == y.ids && x.ids == std::vector<std::uint32_t>({65, 66}));
            CHECK(x.stats.finish == "length" && e.batch_calls == 1 && e.releases == 2);
            CHECK(x.stats.engine_extra != y.stats.engine_extra); // distinct physical slots
            CHECK(execute(r, q, job()).error.empty());
        }
        {
            FakeEngine e; e.script = {65, 0, 66}; e.width = 3;
            sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            auto h = runtime_hooks(e);
            h.receipt = [&e](const sv::SessionView& v, sv::GenerationStats&) {
                CHECK(e.store.pending(v.state) == 0U); CHECK(e.store.consumed(v.state) == 4);
                CHECK(e.physical[v.state.slot].outputs == 2 && e.physical[v.state.slot].draws == 2);
                CHECK(e.physical[v.state.slot].rows == std::vector<std::uint32_t>({10, 11, 12, 65}));
            };
            sv::SessionRuntime r(q, e, e.store, p, h);
            auto j = job(); j.sampling.temperature = 1; j.sampling.seeded = true; j.sampling.seed = 77;
            auto out = execute(r, q, j);
            CHECK(out.error.empty() && out.ids == std::vector<std::uint32_t>({65, 0}) && e.rolled_back == 1);
            CHECK(e.allowances == std::vector<std::uint32_t>({3}));
        }
        {
            FakeEngine e; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto last = execute(r, q, job({}, 0)); CHECK(!last.error.empty());
            CHECK(execute(r, q, job({10}, 3)).stats.finish == "stop");
            CHECK(execute(r, q, job({10}, 2)).stats.finish == "length");
            e.limits.max_context = 4; // use a new runtime for effective limits
        }
        {
            FakeEngine e; e.limits.max_context = 4;
            sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto out = execute(r, q, job());
            CHECK(out.error.empty() && out.ids.size() == 1 && out.stats.finish == "context" && e.allowances.back() == 1);
            CHECK(execute(r, q, job({1, 2, 3, 4, 5})).stats.error_status == 400);
        }
        {
            FakeEngine e; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto a = q.submit(job()); a->cancel(); r.step(); CHECK(drain(a).stats.finish == "cancelled");
            auto b = q.submit(job()); r.step(); b->cancel(); r.step();
            CHECK(drain(b).stats.finish == "cancelled" && e.releases == 1);
            auto c = q.submit(job()); r.step(); r.shutdown(); r.step();
            CHECK(drain(c).stats.finish == "cancelled" && e.releases == 2);
        }
        {
            FakeEngine e; e.fail_release = true; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto bad = execute(r, q, job()); CHECK(!bad.error.empty() && bad.ids.size() == 3 && bad.stats.generated_tokens == 3);
            CHECK(execute(r, q, job()).error.empty()); // quarantined slot 0, use slot 1
            auto held = e.store.acquire(); // sole reusable slot exhausted by external holder
            CHECK(!execute(r, q, job()).error.empty()); e.store.release(held);
            CHECK(execute(r, q, job()).error.empty());
        }
        for (int mode = 1; mode <= 5; ++mode) {
            FakeEngine e; e.malformed = mode; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            CHECK(!execute(r, q, job()).error.empty() && e.releases == 1);
            e.malformed = 0; CHECK(execute(r, q, job()).error.empty());
        }
        {
            FakeEngine e; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            auto h = runtime_hooks(e); h.receipt = [](const sv::SessionView&, sv::GenerationStats&) { throw 7; };
            sv::SessionRuntime r(q, e, e.store, p, h);
            auto out = execute(r, q, job()); CHECK(!out.error.empty() && out.stats.generated_tokens == 3 && out.ids.size() == 3);
            CHECK(e.releases == 1);
        }
        {
            FakeEngine e; sv::GenerationQueue q;
            auto h = runtime_hooks(e); h.schedule = [](std::span<const sv::SessionView>) -> std::vector<sv::SessionWork> { throw 7; };
            sv::SessionRuntime r(q, e, e.store, {}, h);
            CHECK(!execute(r, q, job()).error.empty() && e.releases == 1);
        }
        {
            FakeEngine e; sv::GenerationQueue q;
            sv::SessionRuntime r(q, e, e.store, {}, runtime_hooks(e));
            for (unsigned n = 0; n < 7; ++n) {
                auto j = job();
                if (n == 0) j.sampling.top_p = 0.5;
                if (n == 1) j.sampling.top_k = 2;
                if (n == 2) j.sampling.min_p = 0.1;
                if (n == 3) j.ngram_draft = false;
                if (n == 4) j.session = "named";
                if (n == 5) { auto c = std::make_shared<sv::ConstraintSpec>(); c->thinking.emplace(); j.constraint = c; }
                if (n == 6) { auto c = std::make_shared<sv::ConstraintSpec>(); c->logprobs = true; j.constraint = c; }
                CHECK(execute(r, q, j).stats.error_status == 400);
            }
            CHECK(e.initializes == 0 && e.captures == 0);
        }
        {
            FakeEngine e; auto state = e.store.acquire(); auto ids = std::vector<std::uint32_t>{10, 11};
            e.prefill(ids, state);
            auto snapshot = std::make_shared<ss::PrefixSnapshot>();
            snapshot->identity = runtime_hooks(e).identity; snapshot->state = e.store.capture(state);
            snapshot->attention_frontier = snapshot->recurrent_frontier = 2;
            ss::PrefixCache cache; CHECK(cache.admit(snapshot));
            auto identity = snapshot->identity;
            CHECK(cache.lookup(ids, identity));
            identity.profile = "different"; CHECK(!cache.lookup(ids, identity));
            identity = snapshot->identity; identity.state_format = "different"; CHECK(!cache.lookup(ids, identity));
            identity = snapshot->identity; identity.model = "different"; CHECK(!cache.lookup(ids, identity));
            identity = snapshot->identity; identity.ranks = 2; CHECK(!cache.lookup(ids, identity));
            snapshot->state.pending = 12; CHECK(!cache.admit(snapshot));
            snapshot->state.pending.reset(); snapshot->state.ranks[0].owner.reset(); CHECK(!cache.admit(snapshot));
            e.store.release(state);
        }
        {
            FakeEngine e; e.pending_prefill = true; sv::GenerationQueue q;
            sv::SessionRuntime r(q, e, e.store, {}, runtime_hooks(e));
            CHECK(execute(r, q, job()).error.empty());
            CHECK(r.cache_entries() == 0 && e.captures == 0); // pending anchors never snapshotted
        }
        {
            FakeEngine e; e.width = 1; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            auto h = runtime_hooks(e);
            h.initial_allowance = [](const sv::GenerationJob&) { return 1U; };
            unsigned phases = 0;
            h.continuation = [&](const sv::SessionView& v, ss::FinishReason why) -> std::optional<sv::SessionContinuation> {
                CHECK(why == ss::FinishReason::length && v.rng_draws == 1 && e.store.pending(v.state) == 65U);
                ++phases; return sv::SessionContinuation{{90, 91}, {0}, 2};
            };
            h.receipt = [&](const sv::SessionView& v, sv::GenerationStats&) {
                CHECK(v.rng_draws == 3 && e.physical[v.state.slot].draws == 3);
                CHECK(e.physical[v.state.slot].rows == std::vector<std::uint32_t>({10, 11, 12, 65, 90, 91, 66}));
            };
            sv::SessionRuntime r(q, e, e.store, p, h);
            auto j = job(); j.sampling.temperature = 1;
            const auto out = execute(r, q, j);
            CHECK(out.error.empty() && out.ids == std::vector<std::uint32_t>({65, 66, 0}) && phases == 1);
            CHECK(out.stats.prompt_tokens == 5 && e.suffixes.back() == std::vector<std::uint32_t>({90, 91}));
        }
        {
            FakeEngine e; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            e.malformed = 2; e.malformed_slot = 0;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto a = q.submit(job()); auto b = q.submit(job()); r.step(); r.step();
            CHECK(!drain(a).error.empty()); auto valid = drain(b);
            CHECK(valid.error.empty() && valid.ids == std::vector<std::uint32_t>({65, 66, 0}));
        }
        {
            FakeEngine e; e.width = 1; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto j = job(); j.sampling.temperature = 1;
            auto a = q.submit(j); auto b = q.submit(j);
            r.step(); r.step(); a->cancel(); r.step(); r.step();
            auto x = drain(a); auto y = drain(b);
            CHECK(x.ids.size() == 1 && x.stats.finish == "cancelled");
            CHECK(y.ids.size() == 3 && y.error.empty() && e.batch_calls == 1);
            auto c = q.submit(job()); r.step(); r.step(); e.fail_decode = true; r.step();
            auto failed = drain(c); CHECK(!failed.error.empty() && failed.ids.size() == 1 && failed.stats.generated_tokens == 1);
        }
        {
            FakeEngine e; sv::GenerationQueue q; ss::Policy p; p.enabled = false;
            sv::SessionRuntime r(q, e, e.store, p, runtime_hooks(e));
            auto a = q.submit(job()); r.step();
            e.during_decode = [&] { a->cancel(); };
            r.step(); auto out = drain(a);
            CHECK(out.ids.size() == 3 && out.stats.generated_tokens == 3 && out.stats.finish == "cancelled");
        }
        {
            FakeEngine e; sv::GenerationQueue q;
            std::shared_ptr<sv::GenerationTicket> a, b;
            {
                sv::SessionRuntime r(q, e, e.store, {}, runtime_hooks(e));
                a = q.submit(job()); r.step(); b = q.submit(job());
            }
            CHECK(drain(a).stats.finish == "cancelled" && drain(b).stats.finish == "cancelled" && e.releases == 1);
        }
        {
            FakeEngine e; sv::GenerationQueue q; auto h = runtime_hooks(e);
            h.schedule = [](std::span<const sv::SessionView> views) {
                sv::SessionWork prefill{sv::SessionPhase::prefill, {}, {}}, decode{sv::SessionPhase::decode, {}, {}};
                for (std::size_t i = 0; i < views.size(); ++i) {
                    if (views[i].prompt_offset < views[i].job.prompt.size()) {
                        prefill.lanes.push_back(i); prefill.chunks.push_back(2);
                    } else decode.lanes.push_back(i);
                }
                std::vector<sv::SessionWork> out;
                if (!prefill.lanes.empty()) out.push_back(std::move(prefill));
                if (!decode.lanes.empty()) out.push_back(std::move(decode));
                return out;
            };
            h.checkpoint = [](const sv::SessionView& v) { return v.prompt_offset == 2; };
            sv::SessionRuntime r(q, e, e.store, {}, h);
            auto a = q.submit(job({10, 11, 12, 13})); auto b = q.submit(job({20, 21, 22, 23}));
            r.step(); CHECK(a->progress().prefilled_tokens == 2 && b->progress().prefilled_tokens == 2);
            r.step(); CHECK(a->progress().phase == sv::GenerationPhase::decode);
            r.step(); CHECK(drain(a).error.empty() && drain(b).error.empty());
            CHECK(e.batch_prefills == 2 && e.batch_calls == 1 && r.cache_entries() == 2);
            CHECK(e.suffixes == std::vector<std::vector<std::uint32_t>>({{10, 11}, {20, 21}, {12, 13}, {22, 23}}));
        }
        {
            FakeEngine e; sv::GenerationQueue q; auto h = runtime_hooks(e);
            h.schedule = [](std::span<const sv::SessionView>) {
                return std::vector<sv::SessionWork>{{sv::SessionPhase::prefill, {0, 0}, {1, 1}}};
            };
            sv::SessionRuntime r(q, e, e.store, {}, h);
            CHECK(!execute(r, q, job()).error.empty() && e.suffixes.empty() && e.releases == 1);
        }
        {
            FakeEngine e; sv::GenerationQueue q; auto h = runtime_hooks(e); bool first = true;
            h.checkpoint = [&](const sv::SessionView&) {
                if (first) { first = false; throw std::runtime_error("checkpoint callback failed"); }
                return true;
            };
            sv::SessionRuntime r(q, e, e.store, {}, h);
            CHECK(!execute(r, q, job()).error.empty() && e.releases == 1);
            CHECK(execute(r, q, job()).error.empty());
        }
        {
            FakeEngine e; e.width = 1; sv::GenerationQueue q; auto h = runtime_hooks(e);
            h.initial_allowance = [](const sv::GenerationJob&) { return 1U; };
            h.continuation = [](const sv::SessionView&, ss::FinishReason) -> std::optional<sv::SessionContinuation> { throw 7; };
            sv::SessionRuntime r(q, e, e.store, {}, h);
            auto out = execute(r, q, job());
            CHECK(!out.error.empty() && out.ids == std::vector<std::uint32_t>({65}) && out.stats.generated_tokens == 1 && e.releases == 1);
        }
        {
            MixedFixture e;e.lane_error=true;sv::GenerationQueue q;ss::Policy p;p.enabled=false;
            sv::SessionRuntime runtime(q,e,e.model.store,p,mixed_hooks());
            auto a=q.submit(job()),b=q.submit(job());runtime.step();
            const auto failed=drain(a),peer=drain(b);
            CHECK(failed.error.find("injected lane failure")!=std::string::npos &&
                failed.ids==std::vector<std::uint32_t>({65,66,0}) && failed.stats.generated_tokens==3);
            CHECK(peer.error.empty() && peer.ids==failed.ids && e.model.releases==2);
        }
        {
            MixedFixture e;sv::GenerationQueue q;ss::Policy p;p.enabled=false;
            sv::SessionRuntime runtime(q,e,e.model.store,p,mixed_hooks());
            auto a=q.submit(job()),b=q.submit(job());e.after_first=[&]{b->cancel();};runtime.step();
            CHECK(drain(a).error.empty());const auto cancelled=drain(b);
            CHECK(cancelled.ids.empty() && cancelled.stats.finish=="cancelled" && e.model.suffixes.size()==1);
        }
        {
            // Named sessions require explicit family validation, for both forms.
            FakeEngine e; sv::GenerationQueue q; ss::Policy p; p.enabled=false;
            sv::SessionRuntime ordinary(q,e,e.store,p,runtime_hooks(e));
            auto j=job();j.session_id="named";
            CHECK(execute(ordinary,q,j).stats.error_status==400 && e.initializes==0);
            auto h=runtime_hooks(e);unsigned validations=0;
            h.validate_session=[&](const sv::GenerationJob& request) {
                ++validations;
                if(request.session!="allowed" || !request.session_id.empty())
                    throw std::invalid_argument("family session identifier refused");
            };
            sv::SessionRuntime named(q,e,e.store,p,h);
            j=job();j.session="allowed";
            CHECK(execute(named,q,j).error.empty() && validations>=1);
            const auto acquired=e.initializes;
            j.session_id="conflicting";
            CHECK(execute(named,q,j).stats.error_status==400 && e.initializes==acquired);
        }
        {
            // One unfinished prompt prevents taking the next FIFO head. This
            // exercises native one-open-prefill admission without row slicing.
            FakeEngine e;sv::GenerationQueue q;ss::Policy p;p.enabled=false;
            auto h=runtime_hooks(e);
            h.can_admit=[](std::span<const sv::SessionView> live) {
                return std::none_of(live.begin(),live.end(),[](const auto& v){return v.prompt_offset<v.job.prompt.size();});
            };
            sv::SessionRuntime runtime(q,e,e.store,p,h);
            auto a=q.submit(job()),b=q.submit(job());
            runtime.step();
            CHECK(runtime.live()==1 && q.snapshot().queued==1 && q.snapshot().running==1 && e.initializes==1);
            runtime.step();
            CHECK(q.snapshot().queued==0 && e.initializes==2);
            runtime.step();
            CHECK(drain(a).error.empty() && drain(b).error.empty());
        }
        {
            // A blocked head is retained during the wait. wake_control neither
            // needs a generation nor runs its callback under the queue lock.
            FakeEngine e;sv::GenerationQueue q;ss::Policy p;p.enabled=false;
            auto h=runtime_hooks(e);bool allow=false;
            std::mutex mutex;std::condition_variable cv;bool idle=false;
            unsigned idles=0,controls=0,stops=0;
            h.can_admit=[&](std::span<const sv::SessionView>){return allow;};
            h.idle_interval=std::chrono::hours(1);
            h.idle=[&]{std::lock_guard lock(mutex);++idles;idle=true;cv.notify_one();};
            h.stopped=[&]{++stops;};
            q.set_idle_control([&]{CHECK(q.snapshot().queued==1);++controls;allow=true;});
            sv::SessionRuntime runtime(q,e,e.store,p,h);
            auto ticket=q.submit(job());
            std::exception_ptr failure;
            std::thread owner([&]{try{runtime.run();}catch(...){failure=std::current_exception();}});
            {std::unique_lock lock(mutex);cv.wait(lock,[&]{return idle;});}
            CHECK(q.snapshot().queued==1 && e.initializes==0);
            q.wake_control();
            bool done=false;std::vector<std::uint32_t> ids;
            while(!done){const auto part=ticket->wait(std::chrono::seconds(5),done,false);
                ids.insert(ids.end(),part.begin(),part.end());if(!done && part.empty())break;}
            runtime.shutdown();owner.join();
            if(failure)std::rethrow_exception(failure);
            CHECK(done && ticket->error().empty() && ids.size()==3 && controls==1 && stops==1 && idles<=3);
        }
        {
            // Force the exact empty-take / submission / idle-wait boundary.
            // The old separate has_queued_work() observation is demonstrably
            // true here, but this newly queued request was never inadmissible.
            FakeEngine e;sv::GenerationQueue q;ss::Policy p;p.enabled=false;
            auto h=runtime_hooks(e);h.idle=[]{};h.idle_interval=std::chrono::hours(1);
            sv::SessionRuntime runtime(q,e,e.store,p,h);
            bool blocked=true;
            CHECK(!q.take(false,sv::QueueSelection::fifo,{},{},{},{},&blocked));
            auto ticket=q.submit(job()); // forced arrival after failed take
            CHECK(q.has_queued_work() && !blocked);
            auto waited=std::async(std::launch::async,[&]{
                q.wait_for_work_until(std::chrono::steady_clock::now()+std::chrono::hours(1),!blocked);
            });
            const bool woke=waited.wait_for(std::chrono::seconds(5))==std::future_status::ready;
            if(!woke)q.shutdown(); // bounded cleanup on regression, never polling
            waited.get();CHECK(woke);
            for(unsigned n=0;n<4;++n)runtime.step();
            const auto out=drain(ticket);CHECK(out.error.empty() && out.ids.size()==3);
            // A real inadmissible head is observed under selection's lock.
            ticket=q.submit(job());blocked=false;
            CHECK(!q.take(false,sv::QueueSelection::fifo,[](const auto&){return false;},{},{},{},&blocked) && blocked);
            ticket->cancel();
            // Control callback re-enters the queue, proving outside-lock dispatch.
            unsigned controls=0;q.set_idle_control([&]{CHECK(q.snapshot().queued==1);++controls;});
            q.wake_control();
            q.wait_for_work_until(std::chrono::steady_clock::now()+std::chrono::hours(1),!blocked);
            CHECK(controls==1 && q.snapshot().queued==1);
            CHECK(!q.take(false,sv::QueueSelection::fifo,{},{},{},{},&blocked) && !blocked);
            CHECK(drain(ticket).stats.finish=="cancelled");
            q.shutdown();
            q.wait_for_work_until(std::chrono::steady_clock::now()+std::chrono::hours(1),false);
        }
        {
            // An initially empty timed owner wakes for a generation submission,
            // even with a long idle deadline. No request-triggered control hack.
            FakeEngine e;sv::GenerationQueue q;ss::Policy p;p.enabled=false;
            auto h=runtime_hooks(e);std::mutex mutex;std::condition_variable cv;bool waiting=false;
            h.idle_interval=std::chrono::hours(1);
            h.idle=[&]{std::lock_guard lock(mutex);waiting=true;cv.notify_one();};
            sv::SessionRuntime runtime(q,e,e.store,p,h);std::exception_ptr failure;
            std::thread owner([&]{try{runtime.run();}catch(...){failure=std::current_exception();}});
            {std::unique_lock lock(mutex);cv.wait(lock,[&]{return waiting;});}
            auto ticket=q.submit(job());bool done=false;std::vector<std::uint32_t> ids;
            while(!done){const auto part=ticket->wait(std::chrono::seconds(5),done,false);
                ids.insert(ids.end(),part.begin(),part.end());if(!done && part.empty())break;}
            runtime.shutdown();owner.join();if(failure)std::rethrow_exception(failure);
            CHECK(done && ids.size()==3 && ticket->error().empty());
        }
        {
            // A service with no generation requests still processes bounded
            // control work and its shutdown lifecycle, sleeping between wakes.
            FakeEngine e;sv::GenerationQueue q;ss::Policy p;p.enabled=false;
            auto h=runtime_hooks(e);std::mutex mutex;std::condition_variable cv;
            unsigned idles=0,commands=0,stops=0;bool pending=false;
            h.idle_interval=std::chrono::hours(1);
            h.idle=[&]{std::lock_guard lock(mutex);++idles;
                if(pending){CHECK(q.try_begin_external_work());++commands;q.finish_external_work();pending=false;}
                cv.notify_one();};
            h.stopped=[&]{++stops;};
            q.set_idle_control([&]{std::lock_guard lock(mutex);pending=true;});
            sv::SessionRuntime runtime(q,e,e.store,p,h);
            std::exception_ptr failure;
            std::thread owner([&]{try{runtime.run();}catch(...){failure=std::current_exception();}});
            {std::unique_lock lock(mutex);cv.wait(lock,[&]{return idles==1;});}
            q.wake_control();
            {std::unique_lock lock(mutex);cv.wait(lock,[&]{return commands==1;});}
            CHECK(q.begin_drain() && q.drained() && !q.try_begin_external_work());
            runtime.shutdown();owner.join();if(failure)std::rethrow_exception(failure);
            CHECK(idles==2 && stops==1 && e.initializes==0);
        }
        {
            FakeEngine e;sv::GenerationQueue q;auto h=runtime_hooks(e);unsigned stops=0;
            h.idle=[]{throw std::runtime_error("idle protocol failed");};h.stopped=[&]{++stops;};
            sv::SessionRuntime runtime(q,e,e.store,{},h);bool failed=false;
            try{runtime.run();}catch(const std::runtime_error&){failed=true;}
            CHECK(failed && q.stopping() && stops==1);
            runtime.run();CHECK(stops==1);
        }
        std::cout << "session runtime checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
