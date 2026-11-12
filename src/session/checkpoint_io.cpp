#include "strix/session/checkpoint_io.hpp"
#include "strix/unbuffered_file.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <vector>
#include <fstream>
#include <limits>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace strix::session::checkpoint_io {
namespace {
void inject(Operation op) {
    if (faults().fail == op) {
        if (faults().skip) --faults().skip;
        else throw Error(op, "checkpoint IO injected failure");
    }
}
void check(bool ok, Operation op, const char* message) {
    if (!ok) throw Error(op, message);
}
std::filesystem::path parent(const std::filesystem::path& p) {
    return p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
}
}
Faults& faults() noexcept { static thread_local Faults value; return value; }
std::span<const std::byte> take(std::span<const std::byte> bytes, std::size_t& at, std::uint64_t n) {
    check(at <= bytes.size() && n <= bytes.size() - at, Operation::read, "checkpoint frame bounds");
    auto out = bytes.subspan(at, static_cast<std::size_t>(n));
    at += static_cast<std::size_t>(n); return out;
}
void sync_directory(const std::filesystem::path& dir, bool durable) {
    if (!durable) return;
    inject(Operation::directory_sync);
#ifndef _WIN32
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    check(fd >= 0, Operation::directory_sync, "checkpoint directory open failed");
    int result; do { result = ::fsync(fd); } while (result < 0 && errno == EINTR);
    const auto closed = ::close(fd);
    check(result == 0 && closed == 0, Operation::directory_sync, "checkpoint directory fsync failed");
#else
    // Windows has no supported directory FlushFileBuffers contract. Replacement uses
    // MOVEFILE_WRITE_THROUGH; file contents are explicitly flushed before that call.
    (void)dir;
#endif
}
void create_directories(const std::filesystem::path& dir, bool durable) {
    std::vector<std::filesystem::path> missing;
    for (auto p = std::filesystem::absolute(dir); !std::filesystem::exists(p); p = p.parent_path())
        missing.push_back(p);
    std::filesystem::create_directories(dir);
    for (auto it = missing.rbegin(); it != missing.rend(); ++it) sync_directory(it->parent_path(), durable);
}
void sync_existing(const std::filesystem::path& p) {
    inject(Operation::sync);
#ifdef _WIN32
    const auto h = CreateFileW(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(h != INVALID_HANDLE_VALUE, Operation::sync, "checkpoint retained file open failed");
    const auto ok = FlushFileBuffers(h); CloseHandle(h);
    check(ok != 0, Operation::sync, "checkpoint retained file fsync failed");
#else
    const auto fd = ::open(p.c_str(), O_RDWR | O_CLOEXEC);
    check(fd >= 0, Operation::sync, "checkpoint retained file open failed");
    int ok; do { ok = ::fsync(fd); } while (ok < 0 && errno == EINTR);
    const auto closed = ::close(fd);
    check(ok == 0 && closed == 0, Operation::sync, "checkpoint retained file fsync failed");
#endif
}
struct DirectoryLock::Impl {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~Impl() { if (handle != INVALID_HANDLE_VALUE) { OVERLAPPED o{}; UnlockFileEx(handle,0,MAXDWORD,MAXDWORD,&o); CloseHandle(handle); } }
#else
    int fd = -1;
    ~Impl() { if (fd >= 0) { (void)::flock(fd, LOCK_UN); ::close(fd); } }
#endif
};
DirectoryLock::DirectoryLock(const std::filesystem::path& dir, LockMode mode) : impl_(std::make_unique<Impl>()) {
    check(!dir.empty(), Operation::create, "checkpoint empty lock directory");
    checkpoint_io::create_directories(dir, true);
    const auto path = dir / ".session.lock"; // Persistent inode: never remove or replace.
#ifdef _WIN32
    impl_->handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(impl_->handle != INVALID_HANDLE_VALUE, Operation::create, "checkpoint directory lock open failed");
    OVERLAPPED o{};
    check(LockFileEx(impl_->handle, LOCKFILE_EXCLUSIVE_LOCK | (mode == LockMode::nonblocking ? LOCKFILE_FAIL_IMMEDIATELY : 0),
        0, MAXDWORD, MAXDWORD, &o) != 0, Operation::create, "checkpoint directory locked/unavailable");
#else
    impl_->fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    check(impl_->fd >= 0, Operation::create, "checkpoint directory lock open failed");
    int result; do { result = ::flock(impl_->fd, LOCK_EX | (mode == LockMode::nonblocking ? LOCK_NB : 0)); }
        while (result < 0 && errno == EINTR);
    check(result == 0, Operation::create, "checkpoint directory locked/unavailable");
#endif
}
DirectoryLock::~DirectoryLock() = default;
struct Publication::Impl {
    std::filesystem::path target, temporary;
    Options options;
    std::uint64_t offset = 0;
    bool committed = false, created = false;
    std::unique_ptr<state_file_io::SequentialWriter> direct;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    void close() noexcept {
        direct.reset();
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); handle = INVALID_HANDLE_VALUE; }
#else
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
    }
    ~Impl() {
        close();
        if (created && !committed && !temporary.empty()) { std::error_code ec; std::filesystem::remove(temporary, ec); }
    }
};
Publication::Publication(const std::filesystem::path& target, Options options) : impl_(std::make_unique<Impl>()) {
    auto& p = *impl_; p.target = target; p.options = options;
    checkpoint_io::create_directories(parent(target), options.durable);
    static std::atomic<std::uint64_t> sequence{0};
    // CREATE_NEW/O_EXCL is authoritative even across processes and crashed leftovers.
    const std::filesystem::path prefix(options.temporary_prefix);
    check(options.temporary_prefix.find_first_of("/\\") == std::string::npos && !prefix.has_root_path() &&
        options.temporary_prefix!="." && options.temporary_prefix!="..", Operation::create, "checkpoint temporary prefix is not a filename");
    p.temporary = options.temporary_prefix.empty() ? target : parent(target)/options.temporary_prefix;
    p.temporary += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(++sequence);
    inject(Operation::create);
    if (options.direct == DirectIo::automatic && state_file_io::unbuffered_io_enabled()) {
        p.direct = std::make_unique<state_file_io::SequentialWriter>(p.temporary, true, options.permissions);
        check(p.direct->open(), Operation::create, "checkpoint temporary create failed");
        p.created = true; return;
    }
#ifdef _WIN32
    p.handle = CreateFileW(p.temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(p.handle != INVALID_HANDLE_VALUE, Operation::create, "checkpoint temporary create failed");
#else
    p.fd = ::open(p.temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, options.permissions);
    check(p.fd >= 0, Operation::create, "checkpoint temporary create failed");
#endif
    p.created = true;
}
Publication::~Publication() = default;
void Publication::write(std::span<const std::byte> bytes) {
    auto& p = *impl_;
    check(!p.committed, Operation::write, "checkpoint already published");
    while (!bytes.empty()) {
        inject(Operation::write);
        check(p.offset < faults().fail_write_at, Operation::write, "checkpoint write failed");
        auto n = std::min<std::uint64_t>(bytes.size(), 1U << 26);
        if (faults().max_write) n = std::min(n, faults().max_write);
        n = std::min(n, faults().fail_write_at - p.offset);
        check(n <= UINT64_MAX - p.offset, Operation::write, "checkpoint write overflow");
        std::uint64_t done = 0;
        if (p.direct) { p.direct->write(bytes.data(), n); done = n; }
        else {
#ifdef _WIN32
            DWORD written = 0;
            check(WriteFile(p.handle, bytes.data(), static_cast<DWORD>(n), &written, nullptr) != 0,
                Operation::write, "checkpoint write failed"); done = written;
#else
            ssize_t written; do { written = ::write(p.fd, bytes.data(), static_cast<std::size_t>(n)); } while (written < 0 && errno == EINTR);
            check(written > 0, Operation::write, "checkpoint write failed"); done = static_cast<std::uint64_t>(written);
#endif
        }
        check(done > 0 && done <= n, Operation::write, "checkpoint short write failed");
        p.offset += done; bytes = bytes.subspan(static_cast<std::size_t>(done));
    }
}
void Publication::commit() {
    auto& p = *impl_; check(!p.committed, Operation::replace, "checkpoint already published");
    if (p.options.durable) inject(Operation::sync);
    if (p.direct) { p.direct->finish(p.options.durable); p.direct.reset(); }
    else {
#ifdef _WIN32
        if (p.options.durable) check(FlushFileBuffers(p.handle) != 0, Operation::sync, "checkpoint fsync failed");
        const auto h = p.handle; p.handle = INVALID_HANDLE_VALUE;
        check(CloseHandle(h) != 0, Operation::sync, "checkpoint close failed");
#else
        if (p.options.durable) {
            int result; do { result = ::fsync(p.fd); } while (result < 0 && errno == EINTR);
            check(result == 0, Operation::sync, "checkpoint fsync failed");
        }
        const auto fd = p.fd; p.fd = -1;
        check(::close(fd) == 0, Operation::sync, "checkpoint close failed");
#endif
    }
    inject(Operation::replace);
#ifdef _WIN32
    check(MoveFileExW(p.temporary.c_str(), p.target.c_str(), MOVEFILE_REPLACE_EXISTING |
        (p.options.durable ? MOVEFILE_WRITE_THROUGH : 0)) != 0, Operation::replace, "checkpoint rename failed");
#else
    check(::rename(p.temporary.c_str(), p.target.c_str()) == 0, Operation::replace, "checkpoint rename failed");
#endif
    p.committed = true; // A following directory-sync error is an uncertain durable commit.
    sync_directory(parent(p.target), p.options.durable && p.options.sync_parent);
}
void atomic_write(const std::filesystem::path& p, std::span<const std::byte> bytes, Options options) {
    Publication out(p, options); out.write(bytes); out.commit();
}
void read(const std::filesystem::path& p, std::uint64_t minimum, std::uint64_t maximum,
    const std::function<std::span<std::byte>(std::uint64_t)>& allocate, DirectIo direct,
    const std::function<void(std::span<const std::byte>, std::uint64_t)>& observe) {
    auto admitted = [&](std::uint64_t n) {
        check(n >= minimum && n <= maximum && n <= std::numeric_limits<std::size_t>::max(), Operation::read, "checkpoint size outside admission budget");
        inject(Operation::allocation); auto bytes = allocate(n);
        check(bytes.size() == n && (n == 0 || bytes.data()), Operation::allocation, "checkpoint allocation failed"); return bytes;
    };
    if (direct == DirectIo::automatic && state_file_io::unbuffered_io_enabled()) {
        state_file_io::SequentialReader in(p); check(in.open(), Operation::read, "checkpoint open failed");
        auto bytes = admitted(in.size());
        for (std::size_t at = 0; at < bytes.size();) {
            inject(Operation::read); const auto n = std::min(bytes.size() - at, state_file_io::chunk_bytes);
            in.read(bytes.data() + at, n);
            if (observe) observe(bytes.subspan(at,n),at);
            at += n;
        }
        check(!in.grew(), Operation::read, "checkpoint grew during read"); return;
    }
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    check(bool(in), Operation::read, "checkpoint open failed");
    const auto end = in.tellg(); check(end >= 0, Operation::read, "checkpoint size failed");
    auto bytes = admitted(static_cast<std::uint64_t>(end)); in.seekg(0);
    for (std::size_t at = 0; at < bytes.size();) {
        inject(Operation::read); const auto n = std::min<std::size_t>(bytes.size() - at, 1U << 26);
        in.read(reinterpret_cast<char*>(bytes.data() + at), static_cast<std::streamsize>(n));
        check(bool(in), Operation::read, "checkpoint truncated during read");
        if (observe) observe(bytes.subspan(at,n),at);
        at += n;
    }
    check(in.peek() == std::ifstream::traits_type::eof(), Operation::read, "checkpoint grew during read");
}
} // namespace strix::session::checkpoint_io
