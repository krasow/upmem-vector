#include "jit.h"

#if JIT
#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

#include "common.h"
#include "fusion.h"
#include "logger.h"
#include "opcodes.h"
#include "perfetto/trace.h"
#include "queue.h"
#include "runtime.h"
#include "vectordpu.h"

namespace fs = std::filesystem;

namespace {
using CacheKey = std::vector<Signature>;
// One primary result slot + one per extra horizontal chain.
static constexpr int MAX_RESULT_SLOTS = MAX_HFUSE_CHAINS + 1;
std::map<CacheKey, std::string> g_jit_cache;
std::map<Signature, std::string> g_kernel_obj_cache;
std::recursive_mutex g_jit_cache_mutex;

size_t jit_link_batch_limit() {
  constexpr size_t kIramSafeLinkBatch = 6;
  return JIT_BATCH_SIZE < kIramSafeLinkBatch ? JIT_BATCH_SIZE
                                             : kIramSafeLinkBatch;
}
}  // namespace

std::string jit_canonical_type_name(const char* raw_type_name) {
  if (!raw_type_name) return "int32_t";
  std::string tn = raw_type_name;
  if (tn == "i" || tn == "int" || tn == "int32_t") return "int32_t";
  if (tn == "j" || tn == "uint32_t") return "uint32_t";
  if (tn == "f" || tn == "float") return "float";
  if (tn == "d" || tn == "double") return "double";
  return raw_type_name;
}

std::string jit_signature_hash(const Signature& sig) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint8_t byte) {
    h ^= byte;
    h *= 1099511628211ull;
  };
  for (char c : sig.second) mix(static_cast<uint8_t>(c));
  mix(0xff);
  for (uint8_t b : sig.first) mix(b);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf);
}

std::string jit_batch_hash(const std::vector<Signature>& kernels) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint8_t byte) {
    h ^= byte;
    h *= 1099511628211ull;
  };
  for (const auto& sig : kernels) {
    std::string hash = jit_signature_hash(sig);
    for (char c : hash) mix(static_cast<uint8_t>(c));
    mix(0xfe);
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return std::string(buf);
}

// Anchor for dladdr
extern "C" void vectordpu_jit_dladdr_anchor() {}

// Only valid inside write_kernel_function
// out, stack_type, res, s1, s2, rhs are in scope.
#define EMIT_BINOP(sym)                     \
  out << s1 << " " #sym " " << s2 << ";\n"; \
  break
#define EMIT_SCALAROP(sym)                                         \
  out << s1 << " " #sym " (" << stack_type << ")" << rhs << ";\n"; \
  break
#define EMIT_SHIFTOP                   \
  out << s1 << " >> " << rhs << ";\n"; \
  break

static void write_dpu_main_header(std::ofstream& out) {
  out << R"(#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

__host DPU_LAUNCH_ARGS args;
BARRIER_INIT(my_barrier, NR_TASKLETS);
// Scratchpad for cross-tasklet reduction: one slot per tasklet per chain.
// Oversized for alignment safety; actual usage is MAX_RESULT_SLOTS * NR_TASKLETS.
uint64_t reduction_scratchpad[NR_TASKLETS * 16] __attribute__((aligned(8)));
__dma_aligned uint8_t dpu_workspace[NR_TASKLETS][TASKLET_WORKSPACE_SIZE];

)";
}

static void write_kernel_function(std::ofstream& out,
                                  const std::string& func_name,
                                  const std::vector<uint8_t>& rpn_ops,
                                  const std::string& type_name) {
  std::string stack_type = type_name;
  // necessary to include these headers for the generated kernel code
  // each fused kernel is a separate compilation unit
  out << R"(#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
extern barrier_t my_barrier;
extern uint64_t reduction_scratchpad[NR_TASKLETS * 16];

)";

  out << "int " << func_name << "(void) {\n"
      << "    unsigned int id = me();\n"
      << "    uint32_t n = args.num_elements;\n"
      << "    __mram_ptr " << type_name << " *in_ptr = (__mram_ptr "
      << type_name << " *)(args.pipeline.init_offset);\n\n";

  // Horizontal fusion result support
  out << "    __mram_ptr " << type_name << " *res_ptrs[" << MAX_RESULT_SLOTS
      << "];\n"
      << "    res_ptrs[0] = (__mram_ptr " << type_name
      << " *)(args.pipeline.res_offset);\n";
  for (int i = 0; i < MAX_HFUSE_CHAINS; ++i) {
    out << "    res_ptrs[" << (i + 1) << "] = (__mram_ptr " << type_name
        << " *)(args.pipeline.extra_res_offsets[" << i << "]);\n";
  }

  // WRAM workspace layout:
  //   slot 0:                        input_blk
  //   slots 1..MAX_VFUSE_INPUTS: op_blks[0..MAX_VFUSE_INPUTS-1]
  //   slots MAX_VFUSE_INPUTS+1..+MAX_RESULT_SLOTS: res_blks (reuse scratch
  //   slots)
  out << "\n    " << type_name << " *input_blk = (" << type_name
      << " *)dpu_workspace[id];\n"
      << "    " << type_name << " *op_blks[MAX_VFUSE_INPUTS];\n"
      << "    for (int k = 0; k < MAX_VFUSE_INPUTS; k++)\n"
      << "        op_blks[k] = (" << type_name
      << " *)&dpu_workspace[id][(k + 1) * BLOCK_SIZE * MINIMUM_WRITE_SIZE];\n"
      << "    " << type_name << " *res_blks[" << MAX_RESULT_SLOTS << "];\n"
      << "    for (int k = 0; k < " << MAX_RESULT_SLOTS << "; k++)\n"
      << "        res_blks[k] = (" << type_name
      << " *)&dpu_workspace[id][(1 + MAX_VFUSE_INPUTS + k) * BLOCK_SIZE * "
         "MINIMUM_WRITE_SIZE];\n\n";

  // Scan RPN to find which operand slots are needed and where chain boundaries
  // are.
  bool uses_input = false;
  bool uses_op[MAX_VFUSE_INPUTS] = {false};
  bool uses_local[MAX_HFUSE_CHAINS] = {false};
  bool uses_scalar[MAX_PIPELINE_SCALARS] = {false};
  struct Chain {
    size_t start_op, end_op;
    bool is_reduction;
    uint8_t reduction_op;
  };
  std::vector<Chain> chains;
  size_t current_chain_start = 0;

  auto identify_chain = [&](size_t start, size_t end) {
    Chain c{start, end, false, 0};
    for (size_t i = start; i < end; ++i) {
      uint8_t op = rpn_ops[i];
      if (op == OP_PUSH_INPUT)
        uses_input = true;
      else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS)
        uses_op[op - OP_PUSH_OPERAND_0] = true;
      else if (IS_OP_INDIRECT_UPDATE(op)) {
        if (i + 1 < end) uses_local[rpn_ops[i + 1]] = true;
        i += OP_INLINE_BYTES(op);
      } else if (op == OP_PUSH_SCALAR_VAR) {
        if (i + 1 < end) uses_scalar[rpn_ops[i + 1]] = true;
        i += OP_INLINE_BYTES(op);
      } else if (OP_INLINE_BYTES(op) > 0)
        i += OP_INLINE_BYTES(op);
      else if (IS_OP_REDUCTION(op)) {
        c.is_reduction = true;
        c.reduction_op = op;
      }
    }
    chains.push_back(c);
  };

  for (size_t i = 0; i < rpn_ops.size(); ++i) {
    uint8_t op = rpn_ops[i];
    if (op == OP_NEXT_CHAIN) {
      identify_chain(current_chain_start, i);
      current_chain_start = i + 1;
    } else if (OP_INLINE_BYTES(op) > 0)
      i += OP_INLINE_BYTES(op);
  }
  identify_chain(current_chain_start, rpn_ops.size());

  for (int k = 0; k < MAX_HFUSE_CHAINS; ++k) {
    if (!uses_local[k]) continue;
    out << "    uint32_t local_size_" << k << " = args.pipeline.local_sizes["
        << k << "];\n"
        << "    " << type_name << " *local_accum_" << k
        << " = (" << type_name
        << " *)&dpu_workspace[id][BASE_TASKLET_WORKSPACE_SIZE + " << k
        << " * LOCAL_VECTOR_WORKSPACE_BYTES];\n";
  }
  if (std::any_of(std::begin(uses_local), std::end(uses_local),
                  [](bool x) { return x; })) {
    out << "\n";
  }

#if ENABLE_PROMOTION_REDUCTIONS == 1
  for (const auto& c : chains)
    if (c.is_reduction && type_name == "int32_t") {
      stack_type = "int64_t";
      break;
    }
#endif

  // Reduction accumulators with identity values.
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!chains[c_idx].is_reduction) continue;
    const bool is_float = (stack_type == "float");
    out << "    " << stack_type << " acc_" << c_idx << " = ";
    switch (chains[c_idx].reduction_op) {
      case OP_SUM:
        out << "0;\n";
        break;
      case OP_PRODUCT:
        out << "1;\n";
        break;
      case OP_MIN:
        out << (is_float ? "3.402823466e+38f" : "INT32_MAX") << ";\n";
        break;
      case OP_MAX:
        out << (is_float ? "-3.402823466e+38f" : "INT32_MIN") << ";\n";
        break;
    }
  }

  out << "    " << stack_type << " scalar_vars[MAX_PIPELINE_SCALARS] = {0};\n";
  for (int k = 0; k < MAX_PIPELINE_SCALARS; ++k) {
    if (!uses_scalar[k]) continue;
    out << "    scalar_vars[" << k << "] = (" << stack_type
        << ")args.pipeline.scalars[" << k << "];\n";
  }
  out << "\n";
  for (int k = 0; k < MAX_HFUSE_CHAINS; ++k) {
    if (!uses_local[k]) continue;
    out << "    if (local_size_" << k << " > MAX_LOCAL_VECTOR_SIZE)\n"
        << "        return -2;\n"
        << "    {\n"
        << "        uint32_t local_init_" << k << " = local_size_" << k << ";\n"
        << "        switch (args.pipeline.local_reduce_ops[" << k << "]) {\n"
        << "            case OP_SUM:\n"
        << "                for (uint32_t j = 0; j < local_init_" << k
        << "; ++j) local_accum_" << k << "[j] = 0;\n"
        << "                break;\n"
        << "            case OP_PRODUCT:\n"
        << "                for (uint32_t j = 0; j < local_init_" << k
        << "; ++j) local_accum_" << k << "[j] = 1;\n"
        << "                break;\n"
        << "            case OP_MIN:\n"
        << "                for (uint32_t j = 0; j < local_init_" << k
        << "; ++j) local_accum_" << k << "[j] = INT32_MAX;\n"
        << "                break;\n"
        << "            case OP_MAX:\n"
        << "                for (uint32_t j = 0; j < local_init_" << k
        << "; ++j) local_accum_" << k << "[j] = INT32_MIN;\n"
        << "                break;\n"
        << "        }\n"
        << "    }\n";
  }
  if (std::any_of(std::begin(uses_local), std::end(uses_local),
                  [](bool x) { return x; })) {
    out << "\n";
  }
  // Main per-block loop.
  out << "    uint32_t blk, i, b_e, b_b, b_b_aligned;\n"
      << "    for (blk = id << BLOCK_SIZE_LOG2; blk < n; blk += (NR_TASKLETS "
         "<< BLOCK_SIZE_LOG2)) {\n"
      << "        b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;\n"
      << "        b_b = b_e * sizeof(" << type_name << ");\n"
      << "        b_b_aligned = (b_b + 7) & ~7;\n\n";

  if (uses_input)
    out << "        mram_read((__mram_ptr void const *)(in_ptr + blk), "
           "input_blk, b_b_aligned);\n";
  for (int k = 0; k < MAX_VFUSE_INPUTS; k++) {
    if (!uses_op[k]) continue;
    out << "        {\n"
        << "            __mram_ptr " << type_name << " *p = (__mram_ptr "
        << type_name << " *)(args.pipeline.binary_operands[" << k << "]);\n"
        << "            if (p) mram_read((__mram_ptr void const *)(p + blk), "
           "op_blks["
        << k << "], b_b_aligned);\n"
        << "        }\n";
  }

  out << "        for (i = 0; i < b_e; i++) {\n";

  std::vector<bool> chain_has_output(chains.size(), false);
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    const auto& chain = chains[c_idx];
    out << "            // Chain " << c_idx << "\n";

    struct StackValue {
      std::string value;
      std::string id;
    };
    std::vector<StackValue> stack;
    std::map<std::string, StackValue> cse;
    int expr_id = 0;
    int tmp_n = 0;
    auto get_tmp = [&]() {
      return "t_" + std::to_string(c_idx) + "_" + std::to_string(tmp_n++);
    };
    auto leaf = [](const std::string& value, const std::string& id) {
      return StackValue{value, id};
    };
    auto emit_cached = [&](const std::string& sig,
                           const std::string& expr) -> StackValue {
      auto it = cse.find(sig);
      if (it != cse.end()) return it->second;
      std::string res = get_tmp();
      StackValue out_val{res, "e" + std::to_string(expr_id++)};
      out << "            " << stack_type << " " << res << " = " << expr
          << ";\n";
      cse.emplace(sig, out_val);
      return out_val;
    };
    auto scalar_expr = [&](uint8_t base, const std::string& lhs,
                           const std::string& rhs) {
      switch (base) {
        case OP_ADD_SCALAR:
          return lhs + " + (" + stack_type + ")" + rhs;
        case OP_SUB_SCALAR:
          return lhs + " - (" + stack_type + ")" + rhs;
        case OP_MUL_SCALAR:
          return lhs + " * (" + stack_type + ")" + rhs;
        case OP_DIV_SCALAR:
          return lhs + " / (" + stack_type + ")" + rhs;
        case OP_ASR_SCALAR:
          return lhs + " >> " + rhs;
        case OP_EQ_SCALAR:
          return lhs + " == (" + stack_type + ")" + rhs;
        case OP_LT_SCALAR:
          return lhs + " < (" + stack_type + ")" + rhs;
        case OP_GT_SCALAR:
          return lhs + " > (" + stack_type + ")" + rhs;
        case OP_GE_SCALAR:
          return lhs + " >= (" + stack_type + ")" + rhs;
        case OP_LE_SCALAR:
          return lhs + " <= (" + stack_type + ")" + rhs;
        default:
          return std::string("0");
      }
    };
    auto binary_expr = [&](uint8_t op, const std::string& lhs,
                           const std::string& rhs) {
      switch (op) {
        case OP_ADD:
          return lhs + " + " + rhs;
        case OP_SUB:
          return lhs + " - " + rhs;
        case OP_MUL:
          return lhs + " * " + rhs;
        case OP_DIV:
          return lhs + " / " + rhs;
        case OP_ASR:
          return lhs + " >> " + rhs;
        case OP_EQ:
          return lhs + " == " + rhs;
        case OP_LT:
          return lhs + " < " + rhs;
        case OP_GT:
          return lhs + " > " + rhs;
        case OP_GE:
          return lhs + " >= " + rhs;
        case OP_LE:
          return lhs + " <= " + rhs;
        default:
          return std::string("0");
      }
    };

    for (size_t op_idx = chain.start_op; op_idx < chain.end_op; ++op_idx) {
      uint8_t op = rpn_ops[op_idx];
      if (op == OP_PUSH_INPUT) {
        stack.push_back(leaf("((" + stack_type + ")input_blk[i])", "input"));

      } else if (op >= OP_PUSH_OPERAND_0 &&
                 op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
        uint8_t idx = op - OP_PUSH_OPERAND_0;
        stack.push_back(leaf("((" + stack_type + ")op_blks[" +
                                 std::to_string(idx) + "][i])",
                             "op" + std::to_string(idx)));

      } else if (IS_OP_SCALAR(op) || IS_OP_SCALAR_VAR(op)) {
        std::string rhs;
        std::string rhs_id;
        if (IS_OP_SCALAR(op)) {
          uint8_t b0 = rpn_ops[op_idx + 1], b1 = rpn_ops[op_idx + 2],
                  b2 = rpn_ops[op_idx + 3], b3 = rpn_ops[op_idx + 4];
          int32_t val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
          op_idx += SCALAR_INLINE_BYTES;
          rhs = std::to_string(val);
          rhs_id = "scalar:" + rhs;
        } else {
          uint8_t idx = rpn_ops[op_idx + 1];
          op_idx += SCALAR_VAR_INDEX_BYTES;
          rhs = "args.pipeline.scalars[" + std::to_string(idx) + "]";
          rhs_id = "scalar_var:" + std::to_string(idx);
        }
        StackValue s1 = stack.back();
        stack.pop_back();
        // Normalize SCALAR_VAR opcode to the equivalent SCALAR opcode for a
        // unified switch; both forms share the same operator symbol.
        uint8_t base = IS_OP_SCALAR_VAR(op)
                           ? (op - (OP_ADD_SCALAR_VAR - OP_ADD_SCALAR))
                           : op;
        std::string sig = "scalar_op:" + std::to_string(base) + ":" + s1.id +
                          ":" + rhs_id;
        stack.push_back(emit_cached(sig, scalar_expr(base, s1.value, rhs)));

      } else if (op == OP_DUP) {
        stack.push_back(stack.back());

      } else if (IS_OP_UNARY(op)) {
        StackValue s1 = stack.back();
        stack.pop_back();
        std::string expr;
        switch (op) {
          case OP_NEGATE:
            expr = "-" + s1.value;
            break;
          case OP_ABS:
            expr = "(" + s1.value + " < 0) ? -" + s1.value + " : " + s1.value;
            break;
        }
        stack.push_back(
            emit_cached("unary:" + std::to_string(op) + ":" + s1.id, expr));

      } else if (IS_OP_BINARY(op)) {
        if (stack.size() < 2) {
          fprintf(stderr,
                  "[JIT-DBG] STACK UNDERFLOW at binary op %u, stack size=%zu\n",
                  (unsigned)op, stack.size());
          abort();
        }
        StackValue s2 = stack.back();
        stack.pop_back();
        StackValue s1 = stack.back();
        stack.pop_back();
        std::string sig = "binary:" + std::to_string(op) + ":" + s1.id + ":" +
                          s2.id;
        stack.push_back(emit_cached(sig, binary_expr(op, s1.value, s2.value)));

      } else if (IS_OP_TERNARY(op)) {
        StackValue s1 = stack.back();
        stack.pop_back();
        StackValue s2 = stack.back();
        stack.pop_back();
        StackValue s3 = stack.back();
        stack.pop_back();
        if (op == OP_SELECT) {
          std::string sig =
              "select:" + s3.id + ":" + s2.id + ":" + s1.id;
          stack.push_back(emit_cached(sig, "(" + s3.value + " != 0) ? " +
                                               s2.value + " : " + s1.value));
        }

      } else if (IS_OP_REDUCTION(op)) {
        StackValue s = stack.back();
        stack.pop_back();
        switch (op) {
          case OP_SUM:
            out << "            acc_" << c_idx << " += " << s.value << ";\n";
            break;
          case OP_PRODUCT:
            out << "            acc_" << c_idx << " *= " << s.value << ";\n";
            break;
          case OP_MIN:
            out << "            if (" << s.value << " < acc_" << c_idx
                << ") acc_" << c_idx << " = " << s.value << ";\n";
            break;
          case OP_MAX:
            out << "            if (" << s.value << " > acc_" << c_idx
                << ") acc_" << c_idx << " = " << s.value << ";\n";
            break;
        }
      } else if (op == OP_PUSH_INDEX) {
        stack.push_back(leaf("(blk + i)", "idx"));
      } else if (op == OP_LOAD_INDIRECT) {
        uint8_t op_id = rpn_ops[++op_idx];
        StackValue idx = stack.back();
        stack.pop_back();
        std::string sig =
            "load_indirect:" + std::to_string(op_id) + ":" + idx.id;
        stack.push_back(emit_cached(
            sig, "((__mram_ptr " + type_name +
                     " *)args.pipeline.binary_operands[" +
                     std::to_string((int)op_id) + "])[" + idx.value + "]"));
      } else if (op == OP_ADD_INDIRECT || op == OP_APPLY_INDIRECT) {
        uint8_t local_id = rpn_ops[++op_idx];
        uint8_t reduce_op =
            (op == OP_ADD_INDIRECT) ? OP_SUM : rpn_ops[++op_idx];
        StackValue val = stack.back();
        stack.pop_back();
        StackValue idx = stack.back();
        stack.pop_back();
        std::string slot =
            "local_accum_" + std::to_string((int)local_id) + "[" + idx.value +
            "]";
        switch (reduce_op) {
          case OP_SUM:
            out << "            " << slot << " += " << val.value << ";\n";
            break;
          case OP_PRODUCT:
            out << "            " << slot << " *= " << val.value << ";\n";
            break;
          case OP_MIN:
            out << "            if (" << val.value << " < " << slot << ") "
                << slot << " = " << val.value << ";\n";
            break;
          case OP_MAX:
            out << "            if (" << val.value << " > " << slot << ") "
                << slot << " = " << val.value << ";\n";
            break;
          default:
            out << "            " << slot << " += " << val.value << ";\n";
            break;
        }
      } else if (op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR) {
        if (op == OP_PUSH_SCALAR_VAR) {
          uint8_t idx = rpn_ops[op_idx + 1];
          op_idx += SCALAR_VAR_INDEX_BYTES;
          stack.push_back(
              leaf("scalar_vars[" + std::to_string((uint32_t)idx) + "]",
                   "scalar_var:" + std::to_string((uint32_t)idx)));
        } else {
          uint8_t b0 = rpn_ops[op_idx + 1], b1 = rpn_ops[op_idx + 2],
                  b2 = rpn_ops[op_idx + 3], b3 = rpn_ops[op_idx + 4];
          int32_t val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
          op_idx += SCALAR_INLINE_BYTES;
          stack.push_back(leaf(std::to_string(val),
                               "scalar:" + std::to_string(val)));
        }
      }
    }  // op_idx

    if (!chain.is_reduction && !stack.empty()) {
      out << "            res_blks[" << c_idx << "][i] = " << stack.back().value
          << ";\n";
      chain_has_output[c_idx] = true;
    }
  }  // c_idx

  out << "        }\n";  // end inner element loop

  // Write computed blocks back to MRAM for non-reduction chains.
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (chains[c_idx].is_reduction) continue;
    if (!chain_has_output[c_idx]) continue;
    out << "        if (res_ptrs[" << c_idx << "])\n"
        << "            mram_write(res_blks[" << c_idx
        << "], (__mram_ptr void *)(res_ptrs[" << c_idx
        << "] + blk), b_b_aligned);\n";
  }
  out << "    }\n";  // end block loop

  // Cross-tasklet reduction for scalar reduction chains: each tasklet writes
  // its local accumulator to scratchpad, tasklet 0 merges the per-tasklet
  // partials, and then writes one per-DPU result back to MRAM.
  bool has_reduction_chain = false;
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!chains[c_idx].is_reduction) continue;
    has_reduction_chain = true;
    out << "    {\n"
        << "        uint64_t bf_scratch_" << c_idx << " = 0;\n"
        << "        memcpy(&bf_scratch_" << c_idx << ", &acc_" << c_idx
        << ", sizeof(" << stack_type << "));\n"
        << "        reduction_scratchpad[id * 16 + " << c_idx
        << "] = bf_scratch_" << c_idx << ";\n"
        << "    }\n";
  }
  if (has_reduction_chain) {
    out << "    barrier_wait(&my_barrier);\n"
        << "    if (id == 0) {\n";
    for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
      if (!chains[c_idx].is_reduction) continue;
      out << "        if (res_ptrs[" << c_idx << "]) {\n"
          << "            " << stack_type << " tot_" << c_idx << ";\n"
          << "            memcpy(&tot_" << c_idx << ", &reduction_scratchpad["
          << c_idx << "], sizeof(" << stack_type << "));\n"
          << "            for (uint32_t t = 1; t < NR_TASKLETS; ++t) {\n"
          << "                " << stack_type << " v_" << c_idx << ";\n"
          << "                memcpy(&v_" << c_idx
          << ", &reduction_scratchpad[t * 16 + " << c_idx << "], sizeof("
          << stack_type << "));\n";
      switch (chains[c_idx].reduction_op) {
        case OP_SUM:
          out << "                tot_" << c_idx << " += v_" << c_idx << ";\n";
          break;
        case OP_PRODUCT:
          out << "                tot_" << c_idx << " *= v_" << c_idx << ";\n";
          break;
        case OP_MIN:
          out << "                if (v_" << c_idx << " < tot_" << c_idx
              << ") tot_" << c_idx << " = v_" << c_idx << ";\n";
          break;
        case OP_MAX:
          out << "                if (v_" << c_idx << " > tot_" << c_idx
              << ") tot_" << c_idx << " = v_" << c_idx << ";\n";
          break;
      }
      out << "            }\n"
          << "            uint64_t bf_final_" << c_idx << " = 0;\n"
          << "            memcpy(&bf_final_" << c_idx << ", &tot_" << c_idx
          << ", sizeof(" << stack_type << "));\n"
          << "            mram_write(&bf_final_" << c_idx
          << ", (__mram_ptr void *)res_ptrs[" << c_idx << "], "
          << "MINIMUM_WRITE_SIZE);\n"
          << "        }\n";
    }
    out << "    }\n"
        << "    barrier_wait(&my_barrier);\n";
  }

  // Cross-tasklet reduction: tasklet-local shards are merged in WRAM by tasklet
  // 0, then the combined local vector is written to MRAM once.
  for (size_t c_idx = 0; c_idx < chains.size(); ++c_idx) {
    if (!uses_local[c_idx]) continue;
    std::string local_ptr =
        "args.pipeline.extra_res_offsets[" + std::to_string(c_idx) + "]";
    out << "    {\n"
        << "        barrier_wait(&my_barrier);\n"
        << "        if (id == 0) {\n"
        << "            __mram_ptr " << type_name
        << " *local_ptr = (__mram_ptr " << type_name << " *)(" << local_ptr
        << ");\n"
        << "            if (local_ptr) {\n"
        << "                for (uint32_t t = 1; t < NR_TASKLETS; ++t) {\n"
        << "                    " << type_name << " *src = (" << type_name
        << " *)&dpu_workspace[t][BASE_TASKLET_WORKSPACE_SIZE + " << c_idx
        << " * LOCAL_VECTOR_WORKSPACE_BYTES];\n"
        << "                    for (uint32_t j = 0; j < local_size_" << c_idx
        << "; ++j) {\n"
        << "                        switch (args.pipeline.local_reduce_ops["
        << c_idx << "]) {\n"
        << "                            case OP_SUM:\n"
        << "                                local_accum_" << c_idx
        << "[j] += src[j];\n"
        << "                                break;\n"
        << "                            case OP_PRODUCT:\n"
        << "                                local_accum_" << c_idx
        << "[j] *= src[j];\n"
        << "                                break;\n"
        << "                            case OP_MIN:\n"
        << "                                if (src[j] < local_accum_" << c_idx
        << "[j]) local_accum_" << c_idx << "[j] = src[j];\n"
        << "                                break;\n"
        << "                            case OP_MAX:\n"
        << "                                if (src[j] > local_accum_" << c_idx
        << "[j]) local_accum_" << c_idx << "[j] = src[j];\n"
        << "                                break;\n"
        << "                        }\n"
        << "                    }\n"
        << "                }\n"
        << "                uint32_t local_bytes = local_size_" << c_idx
        << " * sizeof(" << type_name << ");\n"
        << "                uint32_t local_bytes_aligned = (local_bytes + 7) & "
           "~7;\n"
        << "                mram_write(local_accum_" << c_idx
        << ", (__mram_ptr void *)local_ptr, local_bytes_aligned);\n"
        << "            }\n"
        << "        }\n"
        << "        barrier_wait(&my_barrier);\n"
        << "    }\n";
  }

  for (int k = 0; k < MAX_HFUSE_CHAINS; ++k) {
    if (!uses_local[k]) continue;
  }

  out << "    return 0;\n}\n\n";
}

#undef EMIT_BINOP
#undef EMIT_SCALAROP
#undef EMIT_SHIFTOP

static std::string get_include_flags() {
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

static bool compile_dpu_source(const std::string& filepath,
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
      << "Compiled " << (is_object ? "object " : "kernel ") << "to "
      << binpath << std::endl;
#endif
  return true;
}

static bool link_dpu_objects(const std::string& main_path,
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

std::string jit_compile(
    const std::vector<std::pair<std::vector<uint8_t>, std::string>>& kernels) {
  const std::string batch_hash = jit_batch_hash(kernels);
  {
    std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
    auto it = g_jit_cache.find(kernels);
    if (it != g_jit_cache.end()) {
#if ENABLE_DPU_LOGGING >= 1
      auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER);
      log.first() << "cache hit linked binary";
      log.second() << "batch_hash=" << batch_hash
                   << " kernels=" << kernels.size() << " path=" << it->second
                   << std::endl;
#endif
      return it->second;
    }
  }

  trace::jit_compile_begin(kernels);

  const std::string include_flags = get_include_flags();
  const std::string build_dir = "build/jit";
  fs::create_directories(build_dir);

  // Compile each unique kernel to an object file (cached per signature).
  std::vector<std::string> object_files;
  for (const auto& sig : kernels) {
    const std::string kernel_hash = jit_signature_hash(sig);
    std::string obj_path;
    {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      auto it = g_kernel_obj_cache.find(sig);
      if (it != g_kernel_obj_cache.end()) obj_path = it->second;
    }

    if (obj_path.empty()) {
#if ENABLE_DPU_LOGGING >= 1
      auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER);
      log.first() << "compile kernel object";
      log.second() << "kernel_hash=" << kernel_hash << " type=" << sig.second
                   << " " << fusion_rpn_fields(summarize_fusion_rpn(sig.first))
                   << std::endl;
#endif
      const std::string c_path = build_dir + "/k_" + kernel_hash + ".c";
      obj_path = build_dir + "/k_" + kernel_hash + ".o";

      std::ofstream out(c_path);
      write_kernel_function(out, "k_" + kernel_hash, sig.first, sig.second);
      out.close();
#if ENABLE_DPU_LOGGING >= 1
      auto debug_log =
          DpuRuntime::get().get_logger().lock(logcat::JIT_DEBUG, 2);
      debug_log.first() << "wrote kernel source";
      debug_log.second() << "kernel_hash=" << kernel_hash
                         << " path=" << c_path << std::endl;
#endif

      if (!compile_dpu_source(c_path, obj_path, true, include_flags)) {
        trace::jit_compile_end();
        throw std::runtime_error("JIT Compilation failed for " + c_path);
      }
#if ENABLE_DPU_LOGGING >= 1
      DpuRuntime::get().get_logger().lock(logcat::JIT_DEBUG, 2)
          << "compiled " << obj_path << std::endl;
#endif
      {
        std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
        g_kernel_obj_cache[sig] = obj_path;
      }
    } else {
#if ENABLE_DPU_LOGGING >= 2
      auto log = DpuRuntime::get().get_logger().lock(logcat::JIT_COMPILER, 2);
      log.first() << "cache hit kernel object";
      log.second() << "kernel_hash=" << kernel_hash << " type=" << sig.second
                   << " path=" << obj_path << std::endl;
#endif
    }
    object_files.push_back(obj_path);
  }

  // Generate a main() that dispatches on args.kernel to the right sub-kernel.
  static int binary_counter = 0;
  const std::string main_c_path =
      build_dir + "/main_" + std::to_string(binary_counter++) + ".c";
  const std::string binpath = main_c_path + ".dpu";

  {
    std::ofstream out(main_c_path);
    write_dpu_main_header(out);
    for (size_t k = 0; k < kernels.size(); ++k) {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      out << "extern int k_" << jit_signature_hash(kernels[k]) << "(void);\n";
    }
    out << "\nint main() {\n  switch (args.kernel) {\n";
    for (size_t k = 0; k < kernels.size(); ++k) {
      std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
      out << "    case " << (JIT_STATIC_KERNEL_COUNT + k) << ": return k_"
          << jit_signature_hash(kernels[k]) << "();\n";
    }
    out << "    default: return -1;\n  }\n}\n";
  }

  if (!link_dpu_objects(main_c_path, object_files, binpath, include_flags,
                        batch_hash)) {
    trace::jit_compile_end();
    throw std::runtime_error("JIT Linking failed for " + binpath);
  }
#if ENABLE_DPU_LOGGING >= 1
  DpuRuntime::get().get_logger().lock(logcat::JIT_DEBUG, 2)
      << "linked " << binpath << std::endl;
#endif

  {
    std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
    g_jit_cache[kernels] = binpath;
  }
  trace::jit_compile_end();
  return binpath;
}

void EventQueue::flush_jit_batch() {
  if (pending_unique_kernels_.empty()) return;

  std::vector<std::pair<std::vector<uint8_t>, std::string>> batch =
      pending_unique_kernels_;

#if ENABLE_DPU_LOGGING >= 1
  auto log = DpuRuntime::get().get_logger().lock(logcat::QUEUE_JIT);
  log.first() << "flush JIT batch";
  log.second() << "batch_hash=" << jit_batch_hash(batch)
               << " kernels=" << batch.size() << std::endl;
#endif

  std::shared_future<std::string> future = std::async(
      std::launch::deferred, [batch]() { return jit_compile(batch); });
  for (auto& ev : pending_jit_events_) ev->jit_future = future;

  pending_jit_events_.clear();
  pending_unique_kernels_.clear();
}

void EventQueue::lock_for_jit(std::shared_ptr<Event> e) {
  if (e->op != Event::OperationType::COMPUTE || e->is_locked_for_jit) return;
  e->is_locked_for_jit = true;

  if (e->rpn_ops.empty()) {
    e->rpn_ops.push_back(OP_PUSH_INPUT);
    if (e->is_scalar) {
      e->rpn_ops.push_back(map_to_var_op(e->opcode));
      e->rpn_ops.push_back(0);
      e->scalars.push_back(e->scalar_value);
    } else {
      if (e->inputs.size() > 1) e->rpn_ops.push_back(OP_PUSH_OPERAND_0);
      e->rpn_ops.push_back(e->opcode);
    }
  }

  const char* raw_type_name =
      (e->output && e->output->type_name) ? e->output->type_name : "int32_t";
  std::string canonical_type = jit_canonical_type_name(raw_type_name);
  Signature sig = {e->rpn_ops, canonical_type};
  e->jit_kernel_hash = jit_signature_hash(sig);

  // Check if this signature already has a slot in the current batch.
  for (size_t i = 0; i < pending_unique_kernels_.size(); ++i) {
    if (pending_unique_kernels_[i] == sig) {
      e->jit_sub_kernel_idx = i;
      pending_jit_events_.push_back(e);
#if ENABLE_DPU_LOGGING >= 2
      auto log = DpuRuntime::get().get_logger().lock(logcat::QUEUE_JIT, 2);
      log.first() << "cache hit pending JIT batch";
      log.second() << "event_id=" << e->id
                   << " kernel_hash=" << e->jit_kernel_hash
                   << " sub_kernel=" << i
                   << " pending_events=" << pending_jit_events_.size()
                   << std::endl;
#endif
      if (pending_jit_events_.size() >= jit_link_batch_limit())
        flush_jit_batch();
      return;
    }
  }

  if (pending_unique_kernels_.size() >= jit_link_batch_limit())
    flush_jit_batch();

  e->jit_sub_kernel_idx = pending_unique_kernels_.size();
  pending_unique_kernels_.push_back(sig);
  pending_jit_events_.push_back(e);
  if (pending_jit_events_.size() >= jit_link_batch_limit()) flush_jit_batch();
}

bool jit_find_kernel_in_binary(const Signature& sig,
                               const std::string& bin_path, int& out_idx) {
  std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
  for (const auto& [kernels, path] : g_jit_cache) {
    if (path == bin_path) {
      for (size_t i = 0; i < kernels.size(); ++i) {
        if (kernels[i] == sig) {
          out_idx = (int)i;
          return true;
        }
      }
    }
  }
  return false;
}

void jit_cleanup() {
  std::lock_guard<std::recursive_mutex> lock(g_jit_cache_mutex);
#if DEBUG_KEEP_JIT_DIR
  return;
#endif
  const std::string build_dir = "build/jit";
  if (fs::exists(build_dir)) {
    try {
      fs::remove_all(build_dir);
    } catch (...) {
    }
  }
}

#endif  // JIT
