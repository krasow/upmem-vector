// Julia bindings for vectordpu, built on the public C++ API.
//
// Everything here goes through the documented surface (operators, abs, sum,
// select, lazy reductions) rather than detail::launch_*, so the host-side
// fusion pipeline sees the same op stream it would from C++.  In particular
// reductions return a *future*: keeping them unread lets independent
// reductions fuse into one kernel pass.

#include <stats.h>
#include <vectordpu.h>

// The Julia package targets the configuration the library is meant to be used
// in.  PIPELINE=0 / JIT=0 exist so the C++ side can measure the alternatives;
// binding them would mean carrying fallbacks for op sets that do not exist.
#if !PIPELINE || !JIT
#error "UpmemVector.jl requires libvectordpu built with PIPELINE=1 JIT=1"
#endif

#include <cstring>
#include <jlcxx/array.hpp>
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>
#include <vector>

namespace {

using Vec = dpu_vector<int32_t>;
using Future = lazy_reduction_result<int32_t>;

// CxxWrap has no clean mapping for std::vector<Vec>, so K-ary APIs (argmin
// over lanes, extra pipeline operands) take a list built up one push at a time.
struct VecList {
  std::vector<Vec> items;
};

// Dispatch is keyed on the opcode from common/opcodes.h, which src/opcodes.jl
// is generated from -- so there is exactly one numbering shared by both sides
// and nothing to keep in step by hand.  Each arm is the public C++ operator, so
// the host-side fusion pipeline sees the same op stream it would from C++.

Vec apply_binary_op(const Vec& a, const Vec& b, uint8_t op) {
  switch (op) {
    case OP_ADD:
      return a + b;
    case OP_SUB:
      return a - b;
    case OP_MUL:
      return a * b;
    case OP_DIV:
      return a / b;
    case OP_LT:
      return a < b;
    default:
      throw std::invalid_argument("no vector-vector form for opcode " +
                                  std::to_string((int)op));
  }
}

Vec apply_scalar_op(const Vec& a, int32_t s, uint8_t op) {
  switch (op) {
    case OP_ADD_SCALAR:
      return a + s;
    case OP_SUB_SCALAR:
      return a - s;
    case OP_MUL_SCALAR:
      return a * s;
    case OP_DIV_SCALAR:
      return a / s;
    case OP_ASR_SCALAR:
      return a >> s;
    case OP_EQ_SCALAR:
      return a == s;
    default:
      throw std::invalid_argument("no vector-scalar form for opcode " +
                                  std::to_string((int)op));
  }
}

Vec apply_unary_op(const Vec& a, uint8_t op) {
  switch (op) {
    case OP_NEGATE:
      return -a;
    case OP_ABS:
      return abs(a);
    default:
      throw std::invalid_argument("no unary form for opcode " +
                                  std::to_string((int)op));
  }
}

Future apply_reduce_op(const Vec& a, uint8_t op) {
  switch (op) {
    case OP_MIN:
      return min(a);
    case OP_MAX:
      return max(a);
    case OP_SUM:
      return sum(a);
    case OP_PRODUCT:
      return product(a);
    default:
      throw std::invalid_argument("no reduction for opcode " +
                                  std::to_string((int)op));
  }
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

  mod.method("launch_binary", [](const Vec& lhs, const Vec& rhs, uint8_t op) {
    return apply_binary_op(lhs, rhs, op);
  });

  mod.method("launch_binary_scalar",
             [](const Vec& lhs, int32_t scalar, uint8_t op) {
               return apply_scalar_op(lhs, scalar, op);
             });

  mod.method("launch_unary", [](const Vec& input, uint8_t op) {
    return apply_unary_op(input, op);
  });

  mod.method("launch_select",
             [](const Vec& cond, const Vec& then_vec, const Vec& else_vec) {
               return select(cond, then_vec, else_vec);
             });

  // In-place forms: write through the existing buffer instead of allocating a
  // result, which is what keeps an accumulator loop inside MRAM.
  mod.method("apply_binary!", [](Vec& lhs, const Vec& rhs, uint8_t op) {
    switch (op) {
      case OP_ADD:
        lhs += rhs;
        break;
      case OP_SUB:
        lhs -= rhs;
        break;
      case OP_MUL:
        lhs *= rhs;
        break;
      case OP_DIV:
        lhs /= rhs;
        break;
      default:
        throw std::invalid_argument("no in-place vector form for opcode " +
                                    std::to_string((int)op));
    }
  });

  mod.method("apply_scalar!", [](Vec& lhs, int32_t s, uint8_t op) {
    switch (op) {
      case OP_ADD_SCALAR:
        lhs += s;
        break;
      case OP_SUB_SCALAR:
        lhs -= s;
        break;
      case OP_MUL_SCALAR:
        lhs *= s;
        break;
      case OP_DIV_SCALAR:
        lhs /= s;
        break;
      case OP_ASR_SCALAR:
        lhs >>= s;
        break;
      default:
        throw std::invalid_argument("no in-place scalar form for opcode " +
                                    std::to_string((int)op));
    }
  });

  // ---- reductions ----

  // Lazy: returns a future so several reductions can fuse.
  mod.method("launch_reduction_lazy", [](const Vec& input, uint8_t op) {
    return apply_reduce_op(input, op);
  });

  // Eager convenience: submit and read immediately.
  mod.method("launch_reduction", [](const Vec& input, uint8_t op) -> int64_t {
    return (int64_t)apply_reduce_op(input, op).get();
  });

  // ---- build limits, so Julia validates against the real library ----

  mod.method("limit_operands", []() -> int64_t { return MAX_VFUSE_INPUTS; });
  mod.method("limit_scalars", []() -> int64_t { return MAX_PIPELINE_SCALARS; });

  // Checked at module load: a stale wrapper built against a different
  // configuration should fail with a sentence, not a missing symbol.
  mod.method("built_with_jit", []() -> bool { return JIT != 0; });
  mod.method("built_with_pipeline", []() -> bool { return PIPELINE != 0; });

  // ---- vector lists, for the K-ary APIs ----

  mod.add_type<VecList>("DpuVecList")
      .constructor<>()
      .method("veclist_push!",
              [](VecList& l, const Vec& v) { l.items.push_back(v); })
      .method("veclist_length", [](const VecList& l) -> int64_t {
        return (int64_t)l.items.size();
      });

  // ---- explicit RPN programs ----
  //
  // Julia builds the opcode stream itself (see src/expr.jl), which gives it the
  // equivalent of the C++ transform()/reduce() expression lambdas without
  // binding a C++ callable.  It also works under JIT=0, where those two
  // templates do not exist.

  mod.method("launch_pipeline",
             [](Vec& input, jlcxx::ArrayRef<uint8_t> ops, VecList& operands,
                jlcxx::ArrayRef<int32_t> scalars) {
               std::vector<uint8_t> rpn(ops.data(), ops.data() + ops.size());
               std::vector<uint32_t> sc;
               sc.reserve(scalars.size());
               for (size_t i = 0; i < scalars.size(); ++i)
                 sc.push_back((uint32_t)scalars[i]);
               return input.pipeline(rpn, operands.items, sc).vec;
             });

  // Write into an existing vector, for `dest .= expr`.  dpu_vector is a handle
  // type, so rebinding dest would leave other handles on the old buffer -- the
  // program has to target dest's own descriptor.  This is the same event
  // pipeline() submits, minus the fresh allocation and the absorbed_rpn
  // marking, which would be wrong for a destination the user already holds.
  mod.method("launch_pipeline_into",
             [](Vec& dest, Vec& input, jlcxx::ArrayRef<uint8_t> ops,
                VecList& operands, jlcxx::ArrayRef<int32_t> scalars) {
               std::vector<uint8_t> rpn(ops.data(), ops.data() + ops.size());
               std::vector<detail::VectorDescRef> operand_refs;
               operand_refs.reserve(operands.items.size());
               for (auto& o : operands.items)
                 operand_refs.push_back(o.data_desc_ref());
               std::vector<uint32_t> sc;
               sc.reserve(scalars.size());
               for (size_t i = 0; i < scalars.size(); ++i)
                 sc.push_back((uint32_t)scalars[i]);
               detail::launch_universal_pipeline(
                   dest.data_desc_ref(), input.data_desc_ref(), rpn,
                   operand_refs, OpInfo<int32_t>::universal_pipeline, sc);
             });

  mod.method("launch_pipeline_reduce",
             [](Vec& input, jlcxx::ArrayRef<uint8_t> ops, VecList& operands,
                jlcxx::ArrayRef<int32_t> scalars) {
               std::vector<uint8_t> rpn(ops.data(), ops.data() + ops.size());
               std::vector<uint32_t> sc;
               sc.reserve(scalars.size());
               for (size_t i = 0; i < scalars.size(); ++i)
                 sc.push_back((uint32_t)scalars[i]);
               return input.pipeline_reduce(rpn, operands.items, sc);
             });

  // ---- K-ary argmin / argmax over whole vectors ----

  mod.method("launch_argmin_k", [](VecList& lanes) {
    if (lanes.items.empty())
      throw std::invalid_argument("argmin needs at least one lane");
    return argmin(lanes.items);
  });
  mod.method("launch_argmax_k", [](VecList& lanes) {
    if (lanes.items.empty())
      throw std::invalid_argument("argmax needs at least one lane");
    return argmax(lanes.items);
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
