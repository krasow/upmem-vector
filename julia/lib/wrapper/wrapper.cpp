// Julia bindings for vectordpu, built on the public C++ API.
//
// Everything here goes through the documented surface (operators, abs, sum,
// select, lazy reductions) rather than detail::launch_*, so the host-side
// fusion pipeline sees the same op stream it would from C++.  In particular
// reductions return a *future*: keeping them unread lets independent
// reductions fuse into one kernel pass.

#include <jit.h>
#include <stats.h>
#include <vectordpu.h>

// The Julia package targets the configuration the library is meant to be used
// in.  PIPELINE=0 / JIT=0 exist so the C++ side can measure the alternatives;
// binding them would mean carrying fallbacks for op sets that do not exist.
#if !PIPELINE || !JIT
#error "PolymerPIM.jl requires libvectordpu built with PIPELINE=1 JIT=1"
#endif

#include <cstring>
#include <jlcxx/array.hpp>
#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>
#include <memory>
#include <string>
#include <vector>

namespace {

using Vec = dpu_vector<int32_t>;
using Future = lazy_reduction_result<int32_t>;

// CxxWrap has no clean mapping for std::vector<Vec>, so K-ary APIs (argmin
// over lanes, extra pipeline operands) take a list built up one push at a time.
struct VecList {
  std::vector<Vec> items;
};

using Local = dpu_local_vector<int32_t>;

// Every Julia vector is Int32, so a program's JIT signature is just its opcodes
// paired with the canonical name the cache keys on.
Signature make_signature(jlcxx::ArrayRef<uint8_t> ops) {
  return {std::vector<uint8_t>(ops.data(), ops.data() + ops.size()),
          jit_canonical_type_name(typeid(int32_t).name())};
}

// Same story for the scatter targets of a local-reduce program.  Held by
// shared_ptr because dpu_local_vector has no copy assignment and Julia needs to
// keep its own handle alive alongside the list.
struct LocalList {
  std::vector<std::shared_ptr<Local>> items;
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
    // Straight out of the Julia array: the std::vector copy this replaces cost
    // as much as the transfer.  Safe only because add_fence() below finishes
    // it.
    Vec result = Vec::from_cpu(arr.data(), arr.size());
    result.add_fence();  // the Julia array may be collected right after
    return result;
  });

  mod.method("to_cpu!", [](Vec& v, jlcxx::ArrayRef<int32_t> out) {
    // Straight into the Julia array; the std::vector temporary cost an extra
    // allocation and copy.
    v.to_cpu_into(out.data(), out.size());
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
  mod.method("limit_locals",
             []() -> int64_t { return MAX_LOCAL_SCRATCH_VECTORS; });

  // Checked at module load: a stale wrapper built against a different
  // configuration should fail with a sentence, not a missing symbol.
  mod.method("built_with_jit", []() -> bool { return JIT != 0; });
  mod.method("built_with_pipeline", []() -> bool { return PIPELINE != 0; });

  // The whole build.config the *loaded* libvectordpu was compiled from.  The
  // package snapshots this at wrapper-build time; comparing the two at load
  // catches an install prefix rebuilt with different flags underneath a wrapper
  // that is still linked against it.
  mod.method("build_config",
             []() -> std::string { return BUILD_CONFIG_STRING; });
  mod.method("backend", []() -> std::string { return BACKEND; });

  // ---- vector lists, for the K-ary APIs ----

  mod.add_type<VecList>("DpuVecList")
      .constructor<>()
      .method("veclist_push!",
              [](VecList& l, const Vec& v) { l.items.push_back(v); })
      .method("veclist_length", [](const VecList& l) -> int64_t {
        return (int64_t)l.items.size();
      });

  // ---- local (WRAM-resident) scatter accumulators ----
  //
  // These back histogram-style workloads: an RPN program indexes into a small
  // per-DPU array and accumulates, instead of producing one value per lane.
  // dpu_local_vector's to_cpu() gathers every DPU's copy and merges them.

  mod.add_type<Local>("DpuLocalVectorInt32");

  mod.method("local_alloc", [](int32_t n, int32_t reduce_op_idx) {
    static const dpu_local_reduce_op kOps[] = {
        dpu_local_reduce_op::sum, dpu_local_reduce_op::product,
        dpu_local_reduce_op::min, dpu_local_reduce_op::max};
    if (n <= 0) throw std::invalid_argument("local vector needs n > 0");
    if (reduce_op_idx < 0 || reduce_op_idx >= 4)
      throw std::out_of_range("local reduce op out of range");
    return std::make_shared<Local>((uint32_t)n, kOps[reduce_op_idx]);
  });

  mod.method("local_to_cpu!",
             [](const std::shared_ptr<Local>& l, jlcxx::ArrayRef<int32_t> out) {
               std::vector<int32_t> merged = l->to_cpu();
               size_t n = std::min(merged.size(), (size_t)out.size());
               std::copy(merged.begin(), merged.begin() + n, out.data());
             });

  mod.method("local_length", [](const std::shared_ptr<Local>& l) -> int64_t {
    return (int64_t)l->size();
  });

  mod.add_type<LocalList>("DpuLocalList")
      .constructor<>()
      .method("locallist_push!",
              [](LocalList& l, const std::shared_ptr<Local>& v) {
                l.items.push_back(v);
              });

  // The scatter launch.  This is what dpu_jit_foreach reduces to: no
  // elementwise result, the locals carried as the program's outputs.
  mod.method("launch_pipeline_scatter",
             [](Vec& input, jlcxx::ArrayRef<uint8_t> ops, VecList& operands,
                jlcxx::ArrayRef<int32_t> scalars, LocalList& locals) {
               std::vector<uint8_t> rpn(ops.data(), ops.data() + ops.size());
               std::vector<detail::VectorDescRef> operand_refs;
               operand_refs.reserve(operands.items.size());
               for (auto& o : operands.items)
                 operand_refs.push_back(o.data_desc_ref());
               std::vector<uint32_t> sc;
               sc.reserve(scalars.size());
               for (size_t i = 0; i < scalars.size(); ++i)
                 sc.push_back((uint32_t)scalars[i]);
               std::vector<detail::VectorDescRef> local_refs;
               local_refs.reserve(locals.items.size());
               for (auto& l : locals.items)
                 local_refs.push_back(l->data_desc_ref());
               detail::launch_universal_pipeline(
                   detail::VectorDescRef{}, input.data_desc_ref(), rpn,
                   operand_refs, OpInfo<int32_t>::universal_pipeline, sc, {},
                   local_refs);
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

  // ---- JIT introspection (@code_jitted) ----
  //
  // The RPN Julia built, rendered as the C the DPU toolchain would compile.
  // Nothing is compiled or written: this is the codegen, not the cache.

  mod.method("jit_source", [](jlcxx::ArrayRef<uint8_t> ops) -> std::string {
    return jit_kernel_source(make_signature(ops));
  });
  mod.method("jit_hash", [](jlcxx::ArrayRef<uint8_t> ops) -> std::string {
    return jit_signature_hash(make_signature(ops));
  });
  mod.method("jit_main", []() -> std::string { return jit_main_source(); });
  mod.method("jit_dir", []() -> std::string { return jit_build_dir(); });

  // ---- runtime shape ----
  //
  // DPUs are claimed on the first vector allocation, so before that report the
  // count the runtime will take rather than the uninitialized member.
  mod.method("num_dpus", []() -> int64_t {
    auto& rt = DpuRuntime::get();
    return rt.is_initialized() ? (int64_t)rt.num_dpus()
                               : (int64_t)DpuRuntime::configured_num_dpus();
  });
  mod.method("num_tasklets", []() -> int64_t {
    return (int64_t)DpuRuntime::get().num_tasklets();
  });
  mod.method("runtime_initialized",
             []() -> bool { return DpuRuntime::get().is_initialized(); });

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
