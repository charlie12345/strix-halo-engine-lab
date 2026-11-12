#pragma once
// Host-only publication mechanics. Callers own codecs, admission and rank protocols.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace strix::session::checkpoint_io {
enum class LockMode { blocking, nonblocking };
enum class DirectIo { buffered, automatic };
struct Options {
    bool durable = true;
    DirectIo direct = DirectIo::buffered;
    unsigned permissions = 0600;
    std::string temporary_prefix; // optional legacy accounting name, within target directory
    bool sync_parent = true; // a grouped adopter may sync all payload names before its manifest
};
enum class Operation { none, allocation, create, read, write, sync, replace, directory_sync };
class Error : public std::runtime_error {
public:
    Error(Operation op, const char* message) : std::runtime_error(message), operation(op) {}
    Operation operation;
};
// Thread-local, deterministic host-test seams. Defaults perform real platform IO.
struct Faults {
    Operation fail = Operation::none;
    std::uint64_t max_write = 0, fail_write_at = UINT64_MAX;
    unsigned skip = 0; // matching operations before the injected failure
};
Faults& faults() noexcept;
// Overflow-safe byte framing shared by the legacy readers; no allocation.
std::span<const std::byte> take(std::span<const std::byte>, std::size_t& cursor, std::uint64_t size);
void create_directories(const std::filesystem::path&, bool durable);
void sync_directory(const std::filesystem::path&, bool durable = true);
void sync_existing(const std::filesystem::path&);
class DirectoryLock {
public:
    DirectoryLock(const std::filesystem::path&, LockMode);
    ~DirectoryLock();
    DirectoryLock(const DirectoryLock&) = delete;
    DirectoryLock& operator=(const DirectoryLock&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// No implicit directory lock: adopters can publish while holding their protocol lock.
// Exclusive same-directory temporary, full writes, close, replacement, directory sync.
class Publication {
public:
    explicit Publication(const std::filesystem::path& target, Options = {});
    ~Publication();
    Publication(const Publication&) = delete;
    Publication& operator=(const Publication&) = delete;
    void write(std::span<const std::byte>);
    void commit();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
void atomic_write(const std::filesystem::path&, std::span<const std::byte>, Options = {});
// Allocate only after the open file's length passes admission. The returned span must
// have exactly that length. Read/growth failures never return a partially valid record.
void read(const std::filesystem::path&, std::uint64_t minimum, std::uint64_t maximum,
          const std::function<std::span<std::byte>(std::uint64_t)>& allocate,
          DirectIo = DirectIo::buffered,
          const std::function<void(std::span<const std::byte>, std::uint64_t)>& observe = {});
} // namespace strix::session::checkpoint_io
