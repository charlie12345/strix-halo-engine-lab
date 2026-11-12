#include "strix/file_snapshot.hpp"
#include "strix/model_content_hash.hpp"
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <thread>
#ifndef _WIN32
#include <unistd.h>
#endif
namespace {
void check(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
template <class F> void refuses(F &&f, const char *why) {
    try {
        f();
    } catch (const std::exception &e) {
        check(std::string(e.what()).find("startup") != std::string::npos, "refusal diagnostic");
        return;
    }
    throw std::runtime_error(why);
}
struct Root {
    std::filesystem::path path;
    Root() {
        auto base = std::filesystem::temp_directory_path();
        for (unsigned i = 0; i < 100; ++i) {
            path = base /
                   ("strix-snapshot-" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                    "-" + std::to_string(i));
            if (std::filesystem::create_directory(path))
                return;
        }
        throw std::runtime_error("unique snapshot root");
    }
    ~Root() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};
void write(const std::filesystem::path &p, std::string_view data) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(data.data(), static_cast<std::streamsize>(data.size()));
    f.close();
    check(bool(f), "fixture write");
}
struct Fd {
    int value = -1;
    explicit Fd(const std::filesystem::path &p, bool directory = false) {
#ifdef _WIN32
        HANDLE h = CreateFileW(
            p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, directory ? FILE_FLAG_BACKUP_SEMANTICS : FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            value = _open_osfhandle(reinterpret_cast<intptr_t>(h), _O_RDONLY | _O_BINARY);
            if (value == -1)
                CloseHandle(h);
        }
#else
        (void)directory;
        value = ::open(p.c_str(), O_RDONLY);
#endif
        check(value >= 0, "fixture open");
    }
    void close() {
        if (value < 0)
            return;
#ifdef _WIN32
        _close(value);
#else
        ::close(value);
#endif
        value = -1;
    }
    ~Fd() { close(); }
    Fd(const Fd &) = delete;
};
void run() {
    Root root;
    const auto p = root.path / std::filesystem::path(u8"model-\u00e9-\u6a21.bin");
    write(p, "abc");
    Fd fd(p);
    const auto retained = strix::loaded_file(p, fd.value, true);
    const auto closed = strix::loaded_file(p, fd.value);
    const auto identity = strix::fd_identity(fd.value);
    check(retained.identity == identity && retained.fd == fd.value && closed.fd == -1,
          "descriptor retention");
    strix::verify_loader_snapshot({retained, closed});
    check(strix::model_content_identity_hex(strix::sha256_file_content(p, identity)) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "loader Unicode abc SHA vector");
    check(strix::fd_identity(fd.value) == identity, "borrowed fd remains open");
    {
        strix::LoaderCapture capture;
        check(strix::LoaderCapture::active(), "capture active");
        std::exception_ptr error;
        std::thread worker([&] {
            try {
                strix::LoaderCapture::record(p, fd.value);
            } catch (...) {
                error = std::current_exception();
            }
        });
        worker.join();
        if (error)
            std::rethrow_exception(error);
        strix::LoaderCapture::record(p, fd.value);
        auto snapshot = strix::LoaderCapture::snapshot();
        check(snapshot.size() == 1 && snapshot[0].fd == -1 && snapshot[0].identity == identity,
              "capture owns identities, never borrows fds");
        fd.close();
        strix::verify_loader_snapshot(snapshot);
    }
    check(!strix::LoaderCapture::active(), "capture inactive after owner destruction");
    refuses([&] { strix::verify_loader_snapshot({retained}); }, "closed retained fd accepted");
    strix::verify_loader_snapshot({closed});
    refuses([&] { (void)strix::fd_identity(-1); }, "invalid fd accepted");
    refuses([&] { (void)strix::fd_identity(1234567); }, "out of range fd accepted");
    for (unsigned field = 0; field < 7; ++field) {
        auto bad = identity;
        switch (field) {
        case 0:
            bad.dev ^= 1;
            break;
        case 1:
            bad.ino ^= 1;
            break;
        case 2:
            ++bad.size;
            break;
        case 3:
            ++bad.mtime_sec;
            break;
        case 4:
            bad.mtime_ns = -1;
            break;
        case 5:
            ++bad.ctime_sec;
            break;
        default:
            bad.ctime_ns = 1000000000;
            break;
        }
        refuses([&] { (void)strix::sha256_file_content(p, bad); },
                "malformed expected identity accepted");
    }
    refuses([&] { (void)strix::sha256_file_content(p, strix::FileIdentity{}); },
            "zero synthetic identity accepted");
    {
        Fd dir(root.path, true);
        refuses([&] { (void)strix::fd_identity(dir.value); }, "directory accepted");
    }
    const auto other = root.path / "other";
    write(other, "xyz");
    {
        Fd different(other);
#ifdef _WIN32
        refuses([&] { (void)strix::loaded_file(p, different.value); },
                "unrelated descriptor accepted");
#else
        auto wrong = strix::loaded_file(p, different.value);
        refuses([&] { strix::verify_loader_snapshot({wrong}); }, "unrelated descriptor accepted");
#endif
    }
    // Real, completed same-length IO, not a claimed mutation-during-read race.
#ifdef _WIN32
    // Completed rapid same-length writes, with LastWriteTime restored every time.
    // The journal USN must distinguish them even when ChangeTime coalesces.
    for (unsigned iteration = 0; iteration < 16; ++iteration) {
        Fd before_write(p);
        const auto before = strix::fd_identity(before_write.value);
        before_write.close();
        const auto stamp = std::filesystem::last_write_time(p);
        write(p, iteration % 2 ? "abc" : "xyz");
        std::filesystem::last_write_time(p, stamp);
        Fd after_write(p);
        const auto after = strix::fd_identity(after_write.value);
        check(before.ctime_ns != after.ctime_ns, "completed overwrite must move loader USN");
        refuses([&] { (void)strix::sha256_file_content(p, before); }, "rapid overwrite accepted");
    }
    // An existing writer prevents a read admission; no concurrent write is
    // permitted through the handle used for loader-bound hashing.
    {
        Fd current_file(p);
        const auto expected = strix::fd_identity(current_file.value);
        HANDLE writer = CreateFileW(p.c_str(), GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        check(writer != INVALID_HANDLE_VALUE, "writer admission fixture");
        bool rejected = false;
        try {
            (void)strix::sha256_file_content(p, expected);
        } catch (const std::exception &) {
            rejected = true;
        }
        CloseHandle(writer);
        check(rejected, "loader-bound hash admitted an existing writer");
    }
#endif
    const auto mtime = std::filesystem::last_write_time(p);
    write(p, "xyz");
    std::filesystem::last_write_time(p, mtime);
    refuses([&] { strix::verify_loader_snapshot({closed}); },
            "same-length overwrite with restored mtime accepted");
    refuses([&] { (void)strix::sha256_file_content(p, identity); },
            "same-length hash overwrite accepted");
    Fd current(p);
    auto current_file = strix::loaded_file(p, current.value, true);
    auto by_path = current_file;
    by_path.fd = -1;
    const auto moved = root.path / "original";
    std::filesystem::rename(p, moved);
    write(p, "xyz");
#ifdef _WIN32
    // Windows retained handles still bind the renamed file; admission also requires
    // the original loader pathname. Rename itself may move ChangeTime.
    check(strix::fd_identity(current.value).ino == current_file.identity.ino,
          "retained fd lost original file ID");
    refuses([&] { strix::verify_loader_snapshot({current_file}); },
            "retained replaced pathname accepted");
#endif
    refuses([&] { strix::verify_loader_snapshot({by_path}); }, "same-length replacement accepted");
    refuses([&] { (void)strix::sha256_file_content(p, by_path.identity); },
            "replacement hash accepted");
    std::filesystem::remove(p);
    refuses([&] { strix::verify_loader_snapshot({by_path}); }, "missing path accepted");
    refuses([&] { (void)strix::sha256_file_content(p, by_path.identity); },
            "missing hash accepted");
    current.close();
    std::filesystem::remove(moved);
    write(p, "");
    Fd empty(p);
    check(strix::model_content_identity_hex(
              strix::sha256_file_content(p, strix::fd_identity(empty.value))) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "empty loader SHA vector");
    std::cout << "real snapshot identity/Unicode/ownership/overwrite/replacement/closed/missing "
                 "refusals PASS\n";
}
} // namespace
int main() {
    try {
        run();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "snapshot fixture: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "snapshot fixture: unknown exception\n";
        return 2;
    }
}
