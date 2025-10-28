#pragma once

#include <common.h>

#include <iomanip>
#include <iostream>

#include "allocator.h"

inline const char* kernel_id_to_string(KernelID id) {
  switch (id) {
    case K_UNARY_FLOAT_NEGATE:
      return "UNARY_FLOAT_NEGATE";
    case K_UNARY_FLOAT_ABS:
      return "UNARY_FLOAT_ABS";
    case K_UNARY_INT_NEGATE:
      return "UNARY_INT_NEGATE";
    case K_UNARY_INT_ABS:
      return "UNARY_INT_ABS";
    case K_BINARY_FLOAT_ADD:
      return "BINARY_FLOAT_ADD";
    case K_BINARY_FLOAT_SUB:
      return "BINARY_FLOAT_SUB";
    case K_BINARY_INT_ADD:
      return "BINARY_INT_ADD";
    case K_BINARY_INT_SUB:
      return "BINARY_INT_SUB";
    case KERNEL_COUNT:
      return "KERNEL_COUNT";
    default:
      return "UNKNOWN_KERNEL_ID";
  }
}

inline void print_vector_desc(vector_desc desc) {
  std::cout << "[debug-help] Vector Description:" << std::endl;
  for (size_t i = 0; i < desc.first.size(); i++) {
    std::cout << "\t DPU[" << i << "] \t"
              << "ptr=0x" << std::hex << desc.first[i] << " size=" << std::dec
              << desc.second[i] << std::endl;
  }
}

inline void log_allocation(const std::type_info& type, uint32_t n,
                           std::string_view debug_name, const char* debug_file,
                           int debug_line) {
  std::cout << "[mem-logger] Allocated dpu_vector<" << type.name()
            << "> of size " << n;
  if (!debug_name.empty()) {
    std::cout << " (name=\"" << debug_name << "\")";
  }
  if (debug_file != nullptr && debug_line >= 0) {
    std::cout << " at " << debug_file << ":" << debug_line;
  }
  std::cout << std::endl;
}

#ifdef ENABLE_DPU_LOGGING
inline void log_dpu_launch_args(const DPU_LAUNCH_ARGS* args,
                                uint32_t nr_of_dpus) {
  std::cout << "[task-logger] kernel="
            << kernel_id_to_string(static_cast<KernelID>(args->kernel))
            << " nr_of_dpus=" << nr_of_dpus << std::endl;
#if ENABLE_DPU_LOGGING >= 2
  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    std::cout << "[task-logger] DPU[" << i << "]\t"
              << "kernel="
              << kernel_id_to_string(static_cast<KernelID>(args[i].kernel))
              << " is_binary=" << static_cast<int>(args[i].is_binary)
              << " num_elements=" << args[i].num_elements
              << " size_type=" << args[i].size_type;

    if (args[i].is_binary) {
      std::cout << std::hex << std::setfill('0') << " lhs_offset=0x"
                << std::setw(8) << args[i].binary.lhs_offset << " rhs_offset=0x"
                << std::setw(8) << args[i].binary.rhs_offset << " res_offset=0x"
                << std::setw(8) << args[i].binary.res_offset << std::dec;
    } else {
      std::cout << std::hex << std::setfill('0') << " src_offset=0x"
                << std::setw(8) << args[i].unary.rhs_offset << " res_offset=0x"
                << std::setw(8) << args[i].unary.res_offset << std::dec;
    }

    std::cout << std::endl;
  }
#endif
}
#endif