#pragma once
// Invoking the UPMEM toolchain: where its headers live, and how to compile and
// link generated kernels.

#include <string>
#include <vector>

namespace detail {

// -I flags pointing at the installed vectordpu headers.
std::string get_include_flags();

// Compiles one generated .c to a DPU object.  Returns false on failure.
bool compile_dpu_source(const std::string& filepath, const std::string& binpath,
                        bool is_object, const std::string& include_flags);

// Links a batch's main() plus its kernel objects into a loadable binary.
bool link_dpu_objects(const std::string& main_path,
                      const std::vector<std::string>& objects,
                      const std::string& binpath,
                      const std::string& include_flags,
                      const std::string& batch_hash);

}  // namespace detail
