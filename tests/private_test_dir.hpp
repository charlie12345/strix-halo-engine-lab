#pragma once
// A private directory per test run under the test's working directory (ctest runs each test in the build tree, not
// in the system temporary directory): created exclusively (an existing path is never reused), owner-only permissions,
// and file names handed out only while absent, so a test never truncates or follows a file it did not create. The
// directory and what the test wrote into it are removed at the end.
#include <chrono>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

namespace strix::test {

class PrivateTestDir {
public:
    explicit PrivateTestDir(const std::string& prefix) {
        namespace fs = std::filesystem;
        std::random_device entropy;
        for (int attempt = 0; attempt < 16; ++attempt) {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const fs::path candidate = fs::current_path() / (prefix + "-" + std::to_string(stamp) + "-" + std::to_string(entropy()));
            std::error_code error;
            if (fs::create_directory(candidate, error) && !error) {  // false when the path exists
                fs::permissions(candidate, fs::perms::owner_all, fs::perm_options::replace, error);
                path_ = candidate;
                return;
            }
        }
        throw std::runtime_error("could not create a private test directory under " + fs::current_path().string());
    }
    ~PrivateTestDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    PrivateTestDir(const PrivateTestDir&) = delete;
    PrivateTestDir& operator=(const PrivateTestDir&) = delete;

    // A file name in the directory that does not exist yet (not even as a link).
    [[nodiscard]] std::filesystem::path fresh(const std::string& name) const {
        const auto file = path_ / name;
        std::error_code error;
        if (std::filesystem::exists(std::filesystem::symlink_status(file, error)))
            throw std::runtime_error("test file already exists: " + file.string());
        return file;
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace strix::test
