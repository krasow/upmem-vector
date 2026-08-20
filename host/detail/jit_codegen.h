#pragma once
// The DPU source generator: turns an RPN program into a C kernel.
//
// host/jit.cc decides *which* kernels to build and how to cache them; this
// decides what their source looks like.

#include <ostream>
#include <string>
#include <vector>

namespace detail {

// Writes the translation unit shared by every kernel in a JIT batch: launch
// args, the tasklet barrier, and the WRAM workspace.
void write_dpu_main_header(std::ostream& out);

// Writes one kernel function compiling `rpn_ops` for `type_name`.
void write_kernel_function(std::ostream& out, const std::string& func_name,
                           const std::vector<uint8_t>& rpn_ops,
                           const std::string& type_name);

}  // namespace detail
