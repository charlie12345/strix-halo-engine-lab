#pragma once
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#ifdef __linux__
#include <sys/stat.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#include <io.h>
#include <cstdlib>
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif
#endif
namespace strix {
struct FileIdentity {
    std::uint64_t dev = 0, ino = 0, size = 0;
    std::int64_t mtime_sec = 0, mtime_ns = 0, ctime_sec = 0, ctime_ns = 0;
    bool operator==(const FileIdentity&) const = default;
};
#ifdef __linux__
inline FileIdentity file_identity(const struct stat& s) {
    return {s.st_dev, s.st_ino, static_cast<std::uint64_t>(s.st_size),
            s.st_mtim.tv_sec, s.st_mtim.tv_nsec, s.st_ctim.tv_sec, s.st_ctim.tv_nsec};
}
#endif
#ifdef _WIN32
namespace file_snapshot_detail {
// Loader identities use ChangeTime AND the file's journal USN, unlike the
// ordinary v3 cache's creation-time stamp. ChangeTime alone can coalesce rapid
// writes. Unsupported journals/IDs are refused; 128-bit IDs are never truncated.
class Handle {
public:
    HANDLE value;
    explicit Handle(const std::filesystem::path& path)
        : value(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ |
              FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
              FILE_ATTRIBUTE_NORMAL, nullptr)) {}
    ~Handle() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
inline FileIdentity identity(HANDLE handle) {
    FILE_ID_INFO id{};
    FILE_BASIC_INFO basic{};
    FILE_STANDARD_INFO standard{};
    if (handle == INVALID_HANDLE_VALUE || GetFileType(handle) != FILE_TYPE_DISK ||
        !GetFileInformationByHandleEx(handle, FileIdInfo, &id, sizeof(id)) ||
        !GetFileInformationByHandleEx(handle, FileBasicInfo, &basic, sizeof(basic)) ||
        !GetFileInformationByHandleEx(handle, FileStandardInfo, &standard, sizeof(standard)) ||
        standard.Directory || standard.DeletePending || standard.NumberOfLinks == 0 ||
        standard.EndOfFile.QuadPart < 0 || basic.LastWriteTime.QuadPart <= 0 ||
        basic.ChangeTime.QuadPart <= 0)
        throw std::runtime_error("model files changed during startup");
    std::uint64_t ino = 0;
    for (unsigned i = 0; i < 8; ++i) {
        ino |= std::uint64_t(id.FileId.Identifier[i]) << (8 * i);
        if (id.FileId.Identifier[8 + i] != 0)
            throw std::runtime_error("unsupported loader file identity");
    }
    if (!ino) throw std::runtime_error("unsupported loader file identity");
    READ_FILE_USN_DATA request{2, 2};
    struct {
        USN_RECORD_V2 record;
        WCHAR filename[1024];
    } journal{};
    DWORD returned = 0;
    if (!DeviceIoControl(handle, FSCTL_READ_FILE_USN_DATA, &request, sizeof(request),
            &journal, sizeof(journal), &returned, nullptr) ||
        returned < sizeof(USN_RECORD_V2) || journal.record.RecordLength > returned ||
        journal.record.RecordLength < sizeof(USN_RECORD_V2) ||
        journal.record.MajorVersion != 2 || journal.record.MinorVersion != 0 ||
        journal.record.FileReferenceNumber != ino || journal.record.Usn <= 0)
        throw std::runtime_error("unsupported loader change journal during startup");
    constexpr std::int64_t ticks_per_second = 10000000;
    // Preserve the shared layout. Windows mtime is seconds/nanoseconds since
    // 1601; the two ctime words carry raw ChangeTime ticks and USN respectively.
    return {id.VolumeSerialNumber, ino, static_cast<std::uint64_t>(standard.EndOfFile.QuadPart),
        basic.LastWriteTime.QuadPart / ticks_per_second,
        (basic.LastWriteTime.QuadPart % ticks_per_second) * 100,
        basic.ChangeTime.QuadPart, journal.record.Usn};
}
inline FileIdentity path_identity(const std::filesystem::path& path) {
    Handle handle(path);
    return identity(handle.value);
}
// The CRT otherwise invokes the application's invalid-parameter handler for a
// closed descriptor. Restrict suppression to this fallible observation, per thread.
inline void invalid_descriptor(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {}
#if defined(_MSC_VER) && defined(_DEBUG)
inline thread_local unsigned descriptor_observations = 0;
inline int descriptor_report(int type, wchar_t*, int* result) {
    if (type != _CRT_ASSERT || descriptor_observations == 0) return 0;
    *result = 0;
    return 1;
}
#endif
class DescriptorObservation {
    _invalid_parameter_handler previous_;
public:
    DescriptorObservation() {
#if defined(_MSC_VER) && defined(_DEBUG)
        // Debug UCRT asserts before invoking the invalid-parameter handler.
        // Hook2 is reference-counted; the hook handles only this thread's probe.
        // Other threads and all other assertions keep their existing handling.
        if (_CrtSetReportHookW2(_CRT_RPTHOOK_INSTALL, descriptor_report) == -1)
            throw std::runtime_error("cannot observe loader descriptor during startup");
        ++descriptor_observations;
#endif
        previous_ = _set_thread_local_invalid_parameter_handler(invalid_descriptor);
    }
    ~DescriptorObservation() {
        _set_thread_local_invalid_parameter_handler(previous_);
#if defined(_MSC_VER) && defined(_DEBUG)
        --descriptor_observations;
        _CrtSetReportHookW2(_CRT_RPTHOOK_REMOVE, descriptor_report);
#endif
    }
    DescriptorObservation(const DescriptorObservation&) = delete;
};
} // namespace file_snapshot_detail
#endif
inline FileIdentity fd_identity([[maybe_unused]] int fd) {
#ifdef __linux__
    struct stat s{};
    if (::fstat(fd, &s) == 0 && S_ISREG(s.st_mode)) return file_identity(s);
#elif defined(_WIN32)
    if (fd >= 0) {
        const auto handle = [&] {
            file_snapshot_detail::DescriptorObservation observation;
            return _get_osfhandle(fd);
        }();
        if (handle != -1)
            return file_snapshot_detail::identity(reinterpret_cast<HANDLE>(handle));
    }
#endif
    throw std::runtime_error("model files changed during startup");
}
struct LoadedFile {
    std::filesystem::path path, resolved_path; // resolved path is diagnostic only
    FileIdentity identity;
    int fd = -1; // borrowed; -1 means the loader closed its descriptor
};
using LoaderSnapshot = std::vector<LoadedFile>;
inline LoadedFile loaded_file(const std::filesystem::path& path, int fd, bool retained = false) {
    const auto identity = fd_identity(fd);
#ifdef _WIN32
    if (file_snapshot_detail::path_identity(path) != identity)
        throw std::runtime_error("model files changed during startup");
#endif
    std::error_code ec;
    auto resolved = std::filesystem::canonical(path, ec);
    return {std::filesystem::absolute(path), resolved, identity, retained ? fd : -1};
}
inline void verify_loader_snapshot(const LoaderSnapshot& files) {
    for ([[maybe_unused]] const auto& file : files) {
#ifdef __linux__
        struct stat s{};
        const int result = file.fd >= 0 ? ::fstat(file.fd, &s) : ::stat(file.path.c_str(), &s);
        if (result == 0 && S_ISREG(s.st_mode) && file_identity(s) == file.identity) continue;
#elif defined(_WIN32)
        // Both the original pathname and any retained borrowed fd must still
        // name the captured file. Replacement/deletion is refused even if the
        // retained descriptor continues to bind the original Windows file.
        if (file_snapshot_detail::path_identity(file.path) == file.identity &&
            (file.fd == -1 || (file.fd >= 0 && fd_identity(file.fd) == file.identity))) continue;
#endif
        throw std::runtime_error("model files changed during startup");
    }
}
} // namespace strix

namespace strix {
// Shared by startup loaders, including their worker threads. No descriptor is
// borrowed: admission conservatively rechecks original paths even for live fds.
class LoaderCapture {
    inline static std::mutex mutex_;
    inline static std::map<std::filesystem::path, std::vector<FileIdentity>> files_;
    inline static std::atomic<bool> enabled_{false};
public:
    explicit LoaderCapture(bool enable = true) {
        std::lock_guard lock(mutex_);
        files_.clear();
        enabled_ = enable;
    }
    ~LoaderCapture() { enabled_ = false; }
    LoaderCapture(const LoaderCapture&) = delete;
    static bool active() { return enabled_; }
    static void record(const std::filesystem::path& path, int fd) {
#if defined(__linux__) || defined(_WIN32)
        if (!enabled_) return;
        const auto file = loaded_file(path, fd);
        std::lock_guard lock(mutex_);
        auto& identities = files_[file.path];
        for (const auto& old : identities) if (old == file.identity) return;
        identities.push_back(file.identity);
#else
        (void)path; (void)fd;
#endif
    }
    static LoaderSnapshot snapshot() {
        std::lock_guard lock(mutex_);
        LoaderSnapshot out;
        for (const auto& [path, identities] : files_)
            for (const auto& identity : identities) out.push_back({path, {}, identity, -1});
        return out;
    }
};
} // namespace strix
