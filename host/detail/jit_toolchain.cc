// Driving the UPMEM compiler and linker.  See detail/jit_toolchain.h.

#include <jit.h>

#if JIT
#include <common.h>
#include <detail/fusion.h>
#include <detail/jit_toolchain.h>
#include <dlfcn.h>
#include <logger.h>
#include <opcodes.h>
#include <perfetto/trace.h>
#include <queue.h>
#include <runtime.h>
#include <stats.h>
#include <vectordpu.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

namespace fs = std::filesystem;

// Anchor for dladdr
extern "C" void vectordpu_jit_dladdr_anchor() {}

std::string detail::get_include_flags() {
  Dl_info dl_info;
  void* fptr = (void*)&vectordpu_jit_dladdr_anchor;
  std::vector<std::string> include_dirs;

  if (dladdr(fptr, &dl_info) != 0) {
    fs::path lib_path = fs::absolute(dl_info.dli_fname);
    fs::path base = lib_path.parent_path().parent_path();
    if (fs::exists(base / "include" / "vectordpu"))
      include_dirs.push_back((base / "include" / "vectordpu").string());
    if (fs::exists(base.parent_path() / "common"))
      include_dirs.push_back((base.parent_path() / "common").string());
    if (fs::exists(base / "common"))
      include_dirs.push_back((base / "common").string());
  }

  if (include_dirs.empty()) include_dirs.push_back("include/vectordpu");

  std::string flags;
  for (const auto& dir : include_dirs) flags += " -I" + dir;
  return flags;
}

bool detail::compile_dpu_source(const std::string& filepath,
                                const std::string& binpath, bool is_object,
                                const std::string& include_flags) {
  std::string cmd = "dpu-upmem-dpurte-clang -DNR_TASKLETS=" +
                    std::to_string(DpuRuntime::get().num_tasklets()) +
                    include_flags + " -O3 " + (is_object ? "-c " : "") + "-o " +
                    binpath + " " + filepath;

  if (system(cmd.c_str()) != 0) {
    std::cerr << "JIT Compilation failed: " << cmd << std::endl;
    return false;
  }
#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER)
      << "Compiled " << (is_object ? "object " : "kernel ") << "to " << binpath
      << std::endl;
#endif
  return true;
}

bool detail::link_dpu_objects(const std::string& main_path,
                              const std::vector<std::string>& objects,
                              const std::string& binpath,
                              const std::string& include_flags,
                              const std::string& batch_hash) {
  std::string cmd = "dpu-upmem-dpurte-clang -DNR_TASKLETS=" +
                    std::to_string(DpuRuntime::get().num_tasklets()) +
                    include_flags + " -O3 -o " + binpath + " " + main_path;
  for (const auto& obj : objects) cmd += " " + obj;

  if (system(cmd.c_str()) != 0) {
    std::cerr << "JIT Linking failed: " << cmd << std::endl;
    return false;
  }
#if ENABLE_DPU_LOGGING >= 1
  auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER);
  log.first() << "linked binary";
  log.second() << "batch_hash=" << batch_hash << " path=" << binpath
               << std::endl;
#endif
  return true;
}

// Compiles one kernel signature to a DPU object, reusing the cached object when
// the same RPN and element type were compiled before.  Throws on a compiler
// error.

#endif  // JIT
