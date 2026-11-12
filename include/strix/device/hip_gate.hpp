#pragma once

#include "strix/device/gate.hpp"

#include <cstdio>
#include <string>

// HIP side of the start gate, without HIP headers: any main can include this. bind_hip_main() is the one call a HIP
// binary makes before its first allocation.
//
// AMD devices are identified by PCI address. Without a declaration, a lone integrated device keeps working (receipt:
// declared=implicit); a discrete or a second device always needs the file. The current HIP device is per host thread
// and a new thread starts on index 0, so the bound device has to be index 0: when it is not, the gate refuses and names
// the HIP_VISIBLE_DEVICES value that makes it so.
//
// In a build without the HIP runtime the same functions exist: bind_hip_main() only removes the two flags, and
// bind_hip() refuses.
namespace strix::device {

// Resolves the device once per process (throws Refusal), makes it current for the calling thread and prints the
// receipt line on `receipt` (null = silent). Later calls return the same binding and only make the device current for
// their thread; their arguments are ignored.
const Bound& bind_hip(Request request = {}, const std::string& devices_path = {}, std::FILE* receipt = stderr);

// For a main(): takes --devices FILE / --device-id ID out of the argument vector and binds. On a refusal it prints the
// one-line reason and exits with status 2.
void bind_hip_main(int& argc, char** argv, std::FILE* receipt = stderr);

// The binding made by this process, or null before the first bind (and always null without the HIP runtime).
[[nodiscard]] const Bound* hip_bound() noexcept;

} // namespace strix::device
