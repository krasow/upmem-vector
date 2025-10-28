#pragma once

#include <common.h>

#include <iostream>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <vector>

#include "allocator.h"

using std::vector;

#define LOGGER_ARGS_WITH_DEFAULTS \
  std::string_view name = "",     \
                   std::source_location loc = std::source_location::current()

// ============================q
// DPU Vector
// ============================
template <typename T>
class dpu_vector {
 public:
  dpu_vector(uint32_t n, LOGGER_ARGS_WITH_DEFAULTS);

  ~dpu_vector();

  vector<uint32_t> data() const;
  uint32_t size() const;

  vector<T> to_cpu();

  static dpu_vector<T> from_cpu(std::vector<T>& cpu_vec,
                                LOGGER_ARGS_WITH_DEFAULTS);

  vector_desc data_desc() const { return data_; }

 private:
  vector_desc data_;
  uint32_t size_;
  const char* debug_name = nullptr;
  const char* debug_file = nullptr;
  int debug_line = -1;
};

// ============================
// Kernel selectors
// ============================
template <typename T>
struct BinaryKernelSelector;

// float specialization
template <>
struct BinaryKernelSelector<float> {
  static KernelID add() { return KernelID::K_BINARY_FLOAT_ADD; }
  static KernelID sub() { return KernelID::K_BINARY_FLOAT_SUB; }
};

// int specialization
template <>
struct BinaryKernelSelector<int> {
  static KernelID add() { return KernelID::K_BINARY_INT_ADD; }
  static KernelID sub() { return KernelID::K_BINARY_INT_SUB; }
};

template <typename T>
struct UnaryKernelSelector;

// float specialization
template <>
struct UnaryKernelSelector<float> {
  static KernelID negate() { return KernelID::K_UNARY_FLOAT_NEGATE; }
  static KernelID abs() { return KernelID::K_UNARY_FLOAT_ABS; }
};

// int specialization
template <>
struct UnaryKernelSelector<int> {
  static KernelID negate() { return KernelID::K_UNARY_INT_NEGATE; }
  static KernelID abs() { return KernelID::K_UNARY_INT_ABS; }
};

// ============================
// DPU Launch helpers
// ============================
template <typename T>
dpu_vector<T> launch_binop(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs,
                           KernelID kernel_id);

template <typename T>
dpu_vector<T> launch_unary(const dpu_vector<T>& a, KernelID kernel_id);

// ============================
// Operators
// ============================
template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs);

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs, const dpu_vector<T>& rhs);

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a);

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a);
