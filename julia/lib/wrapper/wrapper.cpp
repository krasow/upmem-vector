// Julia bindings for vectordpu, built on the public C++ API.
//
// Everything here goes through the documented surface (operators, abs, sum,
// select, lazy reductions) rather than detail::launch_*, so the host-side
// fusion pipeline sees the same op stream it would from C++.  In particular
// reductions return a *future*: keeping them unread lets independent
// reductions fuse into one kernel pass.

#include <stats.h>
#include <vectordpu.h>

#include <cstring>
#include <jlcxx/array.hpp>
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>
#include <vector>

namespace {

using Vec = dpu_vector<int32_t>;
using Future = lazy_reduction_result<int32_t>;

// Op tables, indexed by the enums in src/operations.jl.  Each entry is the
// public operator, so adding an op here means adding one line, not a kernel id.
using BinaryFn = Vec (*)(const Vec&, const Vec&);
using ScalarFn = Vec (*)(const Vec&, int32_t);
using UnaryFn = Vec (*)(const Vec&);
using ReduceFn = Future (*)(const Vec&);

Vec binary_add(const Vec& a, const Vec& b) { return a + b; }
Vec binary_sub(const Vec& a, const Vec& b) { return a - b; }
Vec binary_mul(const Vec& a, const Vec& b) { return a * b; }
Vec binary_div(const Vec& a, const Vec& b) { return a / b; }
Vec binary_lt(const Vec& a, const Vec& b) { return a < b; }
constexpr BinaryFn kBinaryOps[] = {binary_add, binary_sub, binary_mul,
                                   binary_div, binary_lt};

Vec scalar_add(const Vec& a, int32_t s) { return a + s; }
Vec scalar_sub(const Vec& a, int32_t s) { return a - s; }
Vec scalar_mul(const Vec& a, int32_t s) { return a * s; }
Vec scalar_div(const Vec& a, int32_t s) { return a / s; }
Vec scalar_asr(const Vec& a, int32_t s) { return a >> s; }
Vec scalar_eq(const Vec& a, int32_t s) { return a == s; }
constexpr ScalarFn kScalarOps[] = {scalar_add, scalar_sub, scalar_mul,
                                   scalar_div, scalar_asr, scalar_eq};

Vec unary_negate(const Vec& a) { return -a; }
Vec unary_abs(const Vec& a) { return abs(a); }
constexpr UnaryFn kUnaryOps[] = {unary_negate, unary_abs};

Future reduce_min(const Vec& a) { return min(a); }
Future reduce_max(const Vec& a) { return max(a); }
Future reduce_sum(const Vec& a) { return sum(a); }
Future reduce_product(const Vec& a) { return product(a); }
constexpr ReduceFn kReduceOps[] = {reduce_min, reduce_max, reduce_sum,
                                   reduce_product};

template <typename Table>
int table_size(const Table& t) {
  return (int)(sizeof(t) / sizeof(t[0]));
}

}  // namespace

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
  mod.add_type<Vec>("DpuVectorInt32")
      .constructor<uint32_t>()
      .method("cpp_length",
              [](const Vec& v) -> int64_t { return (int64_t)v.size(); });

  // A queued reduction whose result has not been read yet.  Leaving several
  // of these unread is what lets them share a kernel pass.
  mod.add_type<Future>("DpuFutureInt32")
      .method("cpp_get", [](Future& f) -> int64_t { return (int64_t)f.get(); });

  // ---- host <-> DPU transfers ----

  mod.method("from_cpu_int32", [](jlcxx::ArrayRef<int32_t> arr) {
    std::vector<int32_t> vec(arr.data(), arr.data() + arr.size());
    Vec result = Vec::from_cpu(vec);
    result.add_fence();  // the Julia array may be collected right after
    return result;
  });

  mod.method("to_cpu!", [](Vec& v, jlcxx::ArrayRef<int32_t> out) {
    std::vector<int32_t> cpu = v.to_cpu();
    size_t n = std::min((size_t)v.size(), (size_t)out.size());
    std::copy(cpu.begin(), cpu.begin() + n, out.data());
  });

  // ---- elementwise ----

  mod.method("launch_binary",
             [](const Vec& lhs, const Vec& rhs, int32_t op_idx) {
               if (op_idx < 0 || op_idx >= table_size(kBinaryOps))
                 throw std::out_of_range("binary op index out of range");
               return kBinaryOps[op_idx](lhs, rhs);
             });

  mod.method("launch_binary_scalar",
             [](const Vec& lhs, int32_t scalar, int32_t op_idx) {
               if (op_idx < 0 || op_idx >= table_size(kScalarOps))
                 throw std::out_of_range("scalar op index out of range");
               return kScalarOps[op_idx](lhs, scalar);
             });

  mod.method("launch_unary", [](const Vec& input, int32_t op_idx) {
    if (op_idx < 0 || op_idx >= table_size(kUnaryOps))
      throw std::out_of_range("unary op index out of range");
    return kUnaryOps[op_idx](input);
  });

  mod.method("launch_select",
             [](const Vec& cond, const Vec& then_vec, const Vec& else_vec) {
               return select(cond, then_vec, else_vec);
             });

  // In-place forms: write through the existing buffer instead of allocating a
  // result, which is what keeps an accumulator loop inside MRAM.
  mod.method("apply_binary!", [](Vec& lhs, const Vec& rhs, int32_t op_idx) {
    switch (op_idx) {
      case 0:
        lhs += rhs;
        break;
      case 1:
        lhs -= rhs;
        break;
      case 2:
        lhs *= rhs;
        break;
      case 3:
        lhs /= rhs;
        break;
      default:
        throw std::out_of_range("no in-place form for this op");
    }
  });

  mod.method("apply_scalar!", [](Vec& lhs, int32_t s, int32_t op_idx) {
    switch (op_idx) {
      case 0:
        lhs += s;
        break;
      case 1:
        lhs -= s;
        break;
      case 2:
        lhs *= s;
        break;
      case 3:
        lhs /= s;
        break;
      case 4:
        lhs >>= s;
        break;
      default:
        throw std::out_of_range("no in-place form for this op");
    }
  });

  // ---- reductions ----

  // Lazy: returns a future so several reductions can fuse.
  mod.method("launch_reduction_lazy", [](const Vec& input, int32_t op_idx) {
    if (op_idx < 0 || op_idx >= table_size(kReduceOps))
      throw std::out_of_range("reduction op index out of range");
    return kReduceOps[op_idx](input);
  });

  // Eager convenience: submit and read immediately.
  mod.method("launch_reduction",
             [](const Vec& input, int32_t op_idx) -> int64_t {
               if (op_idx < 0 || op_idx >= table_size(kReduceOps))
                 throw std::out_of_range("reduction op index out of range");
               return (int64_t)kReduceOps[op_idx](input).get();
             });

  // ---- synchronization ----

  mod.method("dpu_fence", [](Vec& v) { v.add_fence(); });
  mod.method("dpu_sync", []() { dpu_fence(); });
  mod.method("cleanup", []() { DpuRuntime::get().shutdown(); });

  // ---- runtime counters (so Julia can assert on fusion, as the C++ suite
  //      does) ----

  mod.method("stat_compute_launches", []() -> int64_t {
    return (int64_t)RuntimeStats::get().snapshot().compute_launches;
  });
  mod.method("stat_horizontal_fusions", []() -> int64_t {
    return (int64_t)RuntimeStats::get().snapshot().horizontal_fusions;
  });
  mod.method("stat_vertical_fusions", []() -> int64_t {
    return (int64_t)RuntimeStats::get().snapshot().vertical_fusions;
  });
}
