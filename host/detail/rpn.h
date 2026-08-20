#pragma once
// RPN program utilities: summarising, previewing, normalising and rewriting the
// opcode streams the fusion passes manipulate.  Nothing here knows about
// events or the queue.

#include <common.h>
#include <opinfo.h>
#include <vectordesc.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#if PIPELINE
namespace detail {

struct FusionRpnSummary {
  size_t decoded_ops = 0;
  size_t bytes = 0;
  size_t chains = 0;
  size_t reductions = 0;
  size_t push_ops = 0;
  std::string top_expr_ops;
};

inline bool is_rpn_push_plumbing(uint8_t op) {
  return (op >= OP_PUSH_INPUT &&
          op <= OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS - 1) ||
         op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR;
}

inline FusionRpnSummary summarize_fusion_rpn(const std::vector<uint8_t>& rpn) {
  FusionRpnSummary summary;
  summary.bytes = rpn.size();
  summary.chains = rpn.empty() ? 0 : 1;
  uint32_t counts[256] = {0};

  for (size_t i = 0; i < rpn.size(); ++i) {
    uint8_t op = rpn[i];
    if (op == OP_NEXT_CHAIN) {
      summary.chains++;
      continue;
    }

    summary.decoded_ops++;
    if (is_rpn_push_plumbing(op)) {
      summary.push_ops++;
    } else {
      counts[op]++;
    }
    if (IS_OP_REDUCTION(op)) summary.reductions++;
    i += OP_INLINE_BYTES(op);
  }

  for (size_t rank = 0; rank < 6; ++rank) {
    uint32_t best_op = 0;
    uint32_t best_count = 0;
    for (uint32_t op = 0; op < 256; ++op) {
      if (counts[op] > best_count) {
        best_op = op;
        best_count = counts[op];
      }
    }
    if (best_count == 0) break;

    std::string name = opcode_to_string((uint8_t)best_op);
    if (name.empty()) name = "OP(" + std::to_string(best_op) + ")";
    if (!summary.top_expr_ops.empty()) summary.top_expr_ops += ", ";
    summary.top_expr_ops += name + "=" + std::to_string(best_count);
    counts[best_op] = 0;
  }

  return summary;
}

inline std::string fusion_rpn_short(const FusionRpnSummary& s) {
  auto plural = [](size_t n, const char* singular, const char* plural) {
    return std::to_string(n) + " " + (n == 1 ? singular : plural);
  };
  return std::to_string(s.decoded_ops) + " ops, " + std::to_string(s.bytes) +
         " bytes, " + plural(s.chains, "chain", "chains") + ", " +
         plural(s.reductions, "reduction", "reductions");
}

inline std::string fusion_rpn_fields(const FusionRpnSummary& s,
                                     const char* prefix = "") {
  std::string p = prefix ? prefix : "";
  return p + fusion_rpn_short(s) + ", " + p +
         "stack pushes=" + std::to_string(s.push_ops) + ", " + p +
         "opcode mix=[" + s.top_expr_ops + "]";
}

inline std::string fusion_op_counts(const FusionRpnSummary& s) {
  return s.top_expr_ops.empty() ? "none" : s.top_expr_ops;
}

inline std::string fusion_rpn_expr_preview(
    const std::vector<uint8_t>& rpn, size_t max_chains = 3,
    size_t max_expr_chars = 120, const std::string& input0_name = "in0",
    size_t operand_base = 1) {
  std::vector<std::string> stack;
  std::vector<std::string> chains;
  size_t hidden_chains = 0;

  auto clip = [&](std::string s) {
    if (s.size() <= max_expr_chars) return s;
    if (max_expr_chars <= 3) return std::string(max_expr_chars, '.');
    s.resize(max_expr_chars - 3);
    s += "...";
    return s;
  };
  auto scalar_inline = [&](size_t& i) {
    if (i + 4 >= rpn.size()) return std::string("imm(?)");
    uint32_t bits = (uint32_t)rpn[i + 1] | ((uint32_t)rpn[i + 2] << 8) |
                    ((uint32_t)rpn[i + 3] << 16) | ((uint32_t)rpn[i + 4] << 24);
    i += 4;
    return "imm(" + std::to_string((int32_t)bits) + ")";
  };
  auto scalar_var = [&](size_t& i) {
    if (i + 1 >= rpn.size()) return std::string("s?");
    return "s" + std::to_string(rpn[++i]);
  };
  auto binary_symbol = [](uint8_t op) -> const char* {
    switch (op) {
      case OP_ADD:
      case OP_ADD_SCALAR:
      case OP_ADD_SCALAR_VAR:
        return "+";
      case OP_SUB:
      case OP_SUB_SCALAR:
      case OP_SUB_SCALAR_VAR:
        return "-";
      case OP_MUL:
      case OP_MUL_SCALAR:
      case OP_MUL_SCALAR_VAR:
        return "*";
      case OP_DIV:
      case OP_DIV_SCALAR:
      case OP_DIV_SCALAR_VAR:
        return "/";
      case OP_ASR:
      case OP_ASR_SCALAR:
      case OP_ASR_SCALAR_VAR:
        return ">>";
      case OP_EQ:
      case OP_EQ_SCALAR:
      case OP_EQ_SCALAR_VAR:
        return "==";
      case OP_LT:
      case OP_LT_SCALAR:
      case OP_LT_SCALAR_VAR:
        return "<";
      case OP_GT:
      case OP_GT_SCALAR:
      case OP_GT_SCALAR_VAR:
        return ">";
      case OP_GE:
      case OP_GE_SCALAR:
      case OP_GE_SCALAR_VAR:
        return ">=";
      case OP_LE:
      case OP_LE_SCALAR:
      case OP_LE_SCALAR_VAR:
        return "<=";
      default:
        return nullptr;
    }
  };
  auto reduction_name = [](uint8_t op) -> const char* {
    switch (op) {
      case OP_MIN:
        return "min";
      case OP_MAX:
        return "max";
      case OP_SUM:
        return "sum";
      case OP_PRODUCT:
        return "product";
      default:
        return "reduce";
    }
  };
  auto flush_chain = [&]() {
    if (chains.size() < max_chains) {
      chains.push_back(stack.empty() ? "<empty>" : clip(stack.back()));
    } else {
      hidden_chains++;
    }
    stack.clear();
  };

  for (size_t i = 0; i < rpn.size(); ++i) {
    uint8_t op = rpn[i];
    if (op == OP_NEXT_CHAIN) {
      flush_chain();
    } else if (op == OP_PUSH_INPUT) {
      stack.push_back(input0_name);
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      stack.push_back("in" +
                      std::to_string(op - OP_PUSH_OPERAND_0 + operand_base));
    } else if (op == OP_DUP) {
      if (!stack.empty()) stack.push_back(stack.back());
    } else if (op == OP_PUSH_INDEX) {
      stack.push_back("idx");
    } else if (op == OP_PUSH_SCALAR) {
      stack.push_back(scalar_inline(i));
    } else if (op == OP_PUSH_SCALAR_VAR) {
      stack.push_back(scalar_var(i));
    } else if (IS_OP_SCALAR(op) || IS_OP_SCALAR_VAR(op)) {
      if (stack.empty()) {
        i += OP_INLINE_BYTES(op);
        continue;
      }
      std::string rhs = IS_OP_SCALAR(op) ? scalar_inline(i) : scalar_var(i);
      std::string lhs = stack.back();
      stack.pop_back();
      const char* sym = binary_symbol(op);
      stack.push_back(
          sym ? clip("(" + lhs + " " + sym + " " + rhs + ")")
              : clip(opcode_to_string(op) + "(" + lhs + ", " + rhs + ")"));
    } else if (op == OP_NEGATE || op == OP_ABS) {
      if (stack.empty()) continue;
      std::string v = stack.back();
      stack.pop_back();
      stack.push_back(op == OP_NEGATE ? clip("(-" + v + ")")
                                      : clip("abs(" + v + ")"));
    } else if (IS_OP_BINARY(op)) {
      if (stack.size() < 2) continue;
      std::string rhs = stack.back();
      stack.pop_back();
      std::string lhs = stack.back();
      stack.pop_back();
      const char* sym = binary_symbol(op);
      stack.push_back(
          sym ? clip("(" + lhs + " " + sym + " " + rhs + ")")
              : clip(opcode_to_string(op) + "(" + lhs + ", " + rhs + ")"));
    } else if (op == OP_SELECT) {
      if (stack.size() < 3) continue;
      std::string false_val = stack.back();
      stack.pop_back();
      std::string true_val = stack.back();
      stack.pop_back();
      std::string cond = stack.back();
      stack.pop_back();
      stack.push_back(
          clip("select(" + cond + ", " + true_val + ", " + false_val + ")"));
    } else if (IS_OP_REDUCTION(op)) {
      if (stack.empty()) continue;
      std::string v = stack.back();
      stack.pop_back();
      stack.push_back(clip(std::string(reduction_name(op)) + "(" + v + ")"));
    } else if (op == OP_LOAD_INDIRECT) {
      if (stack.empty() || i + 1 >= rpn.size()) continue;
      uint8_t operand_idx = rpn[++i];
      std::string idx = stack.back();
      stack.pop_back();
      stack.push_back(clip("load(in" +
                           std::to_string(operand_idx + operand_base) + ", " +
                           idx + ")"));
    } else if (op == OP_ADD_INDIRECT || op == OP_APPLY_INDIRECT) {
      size_t extra = op == OP_APPLY_INDIRECT ? 2 : 1;
      if (stack.size() < 2 || i + extra >= rpn.size()) {
        i += std::min(extra, rpn.size() - i - 1);
        continue;
      }
      uint8_t local_idx = rpn[++i];
      uint8_t reduce_op = op == OP_APPLY_INDIRECT ? rpn[++i] : OP_SUM;
      std::string val = stack.back();
      stack.pop_back();
      std::string idx = stack.back();
      stack.pop_back();
      stack.push_back(clip("local" + std::to_string(local_idx) + "[" + idx +
                           "] <- " + reduction_name(reduce_op) + "(" + val +
                           ")"));
    } else {
      i += OP_INLINE_BYTES(op);
    }
  }

  if (!rpn.empty()) flush_chain();
  if (chains.empty()) return "<empty>";

  std::ostringstream out;
  for (size_t i = 0; i < chains.size(); ++i) {
    if (i) out << "; ";
    out << "chain" << i << ": " << chains[i];
  }
  if (hidden_chains) out << "; +" << hidden_chains << " more";
  return out.str();
}

inline uint8_t map_to_var_op(uint8_t op) {
  switch (op) {
    case OP_ADD_SCALAR:
      return OP_ADD_SCALAR_VAR;
    case OP_SUB_SCALAR:
      return OP_SUB_SCALAR_VAR;
    case OP_MUL_SCALAR:
      return OP_MUL_SCALAR_VAR;
    case OP_DIV_SCALAR:
      return OP_DIV_SCALAR_VAR;
    case OP_ASR_SCALAR:
      return OP_ASR_SCALAR_VAR;
    case OP_EQ_SCALAR:
      return OP_EQ_SCALAR_VAR;
    case OP_LT_SCALAR:
      return OP_LT_SCALAR_VAR;
    default:
      return op;
  }
}

inline uint8_t map_from_var_op(uint8_t op) {
  switch (op) {
    case OP_ADD_SCALAR_VAR:
      return OP_ADD_SCALAR;
    case OP_SUB_SCALAR_VAR:
      return OP_SUB_SCALAR;
    case OP_MUL_SCALAR_VAR:
      return OP_MUL_SCALAR;
    case OP_DIV_SCALAR_VAR:
      return OP_DIV_SCALAR;
    case OP_ASR_SCALAR_VAR:
      return OP_ASR_SCALAR;
    case OP_EQ_SCALAR_VAR:
      return OP_EQ_SCALAR;
    case OP_LT_SCALAR_VAR:
      return OP_LT_SCALAR;
    case OP_GT_SCALAR_VAR:
      return OP_GT_SCALAR;
    case OP_GE_SCALAR_VAR:
      return OP_GE_SCALAR;
    case OP_LE_SCALAR_VAR:
      return OP_LE_SCALAR;
    default:
      return op;
  }
}

// Expand a raw (unfused) event's opcode into its canonical RPN sequence.

struct RpnExpr {
  std::vector<uint8_t> tokens;
  uint8_t associative_op = 0;
  std::vector<std::vector<uint8_t>> terms;
};

inline std::vector<uint8_t> emit_rpn_expr(const RpnExpr& expr) {
  if (expr.associative_op == 0) return expr.tokens;
  if (expr.terms.empty()) return {};

  std::vector<uint8_t> out = expr.terms[0];
  for (size_t i = 1; i < expr.terms.size(); ++i) {
    out.insert(out.end(), expr.terms[i].begin(), expr.terms[i].end());
    out.push_back(expr.associative_op);
  }
  return out;
}

inline void append_rpn_terms(std::vector<std::vector<uint8_t>>& terms,
                             const RpnExpr& expr, uint8_t op) {
  if (expr.associative_op == op) {
    terms.insert(terms.end(), expr.terms.begin(), expr.terms.end());
  } else {
    terms.push_back(emit_rpn_expr(expr));
  }
}

inline bool append_rpn_token_with_inline(const std::vector<uint8_t>& in,
                                         size_t& i, std::vector<uint8_t>& out) {
  uint8_t op = in[i];
  out.push_back(op);
  size_t inline_bytes = OP_INLINE_BYTES(op);
  if (i + inline_bytes >= in.size()) return false;
  for (size_t j = 0; j < inline_bytes; ++j) out.push_back(in[++i]);
  return true;
}

inline bool normalize_associative_rpn_chain(const std::vector<uint8_t>& in,
                                            std::vector<uint8_t>& out) {
  std::vector<RpnExpr> stack;

  for (size_t i = 0; i < in.size(); ++i) {
    uint8_t op = in[i];

    if (op == OP_LOAD_INDIRECT || op == OP_ADD_INDIRECT ||
        op == OP_APPLY_INDIRECT) {
      return false;
    }

    if (IS_OP_STACK(op) || op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR ||
        op == OP_PUSH_INDEX) {
      RpnExpr expr;
      if (!append_rpn_token_with_inline(in, i, expr.tokens)) return false;
      stack.push_back(std::move(expr));
    } else if (op == OP_DUP) {
      if (stack.empty()) return false;
      stack.push_back(stack.back());
    } else if (IS_OP_UNARY(op) || IS_OP_SCALAR(op) || IS_OP_SCALAR_VAR(op)) {
      if (stack.empty()) return false;
      RpnExpr expr;
      expr.tokens = emit_rpn_expr(stack.back());
      stack.pop_back();
      if (!append_rpn_token_with_inline(in, i, expr.tokens)) return false;
      stack.push_back(std::move(expr));
    } else if (IS_OP_BINARY(op)) {
      if (stack.size() < 2) return false;
      RpnExpr rhs = std::move(stack.back());
      stack.pop_back();
      RpnExpr lhs = std::move(stack.back());
      stack.pop_back();

      RpnExpr expr;
      if (op == OP_ADD || op == OP_MUL) {
        expr.associative_op = op;
        append_rpn_terms(expr.terms, lhs, op);
        append_rpn_terms(expr.terms, rhs, op);
      } else {
        expr.tokens = emit_rpn_expr(lhs);
        std::vector<uint8_t> rhs_tokens = emit_rpn_expr(rhs);
        expr.tokens.insert(expr.tokens.end(), rhs_tokens.begin(),
                           rhs_tokens.end());
        expr.tokens.push_back(op);
      }
      stack.push_back(std::move(expr));
    } else if (IS_OP_TERNARY(op)) {
      if (stack.size() < 3) return false;
      RpnExpr c = std::move(stack.back());
      stack.pop_back();
      RpnExpr b = std::move(stack.back());
      stack.pop_back();
      RpnExpr a = std::move(stack.back());
      stack.pop_back();
      RpnExpr expr;
      expr.tokens = emit_rpn_expr(a);
      std::vector<uint8_t> b_tokens = emit_rpn_expr(b);
      std::vector<uint8_t> c_tokens = emit_rpn_expr(c);
      expr.tokens.insert(expr.tokens.end(), b_tokens.begin(), b_tokens.end());
      expr.tokens.insert(expr.tokens.end(), c_tokens.begin(), c_tokens.end());
      expr.tokens.push_back(op);
      stack.push_back(std::move(expr));
    } else if (IS_OP_REDUCTION(op)) {
      if (stack.empty()) return false;
      RpnExpr expr;
      expr.tokens = emit_rpn_expr(stack.back());
      stack.pop_back();
      expr.tokens.push_back(op);
      stack.push_back(std::move(expr));
    } else {
      return false;
    }
  }

  if (stack.size() != 1) return false;
  out = emit_rpn_expr(stack.back());
  return true;
}

inline std::vector<uint8_t> normalize_associative_rpn(
    const std::vector<uint8_t>& in) {
  std::vector<uint8_t> out;
  std::vector<uint8_t> chain;

  for (size_t i = 0; i < in.size(); ++i) {
    uint8_t op = in[i];
    if (op == OP_NEXT_CHAIN) {
      std::vector<uint8_t> normalized;
      if (chain.empty() || !normalize_associative_rpn_chain(chain, normalized))
        return in;
      out.insert(out.end(), normalized.begin(), normalized.end());
      out.push_back(OP_NEXT_CHAIN);
      chain.clear();
    } else {
      if (!append_rpn_token_with_inline(in, i, chain)) return in;
    }
  }

  std::vector<uint8_t> normalized;
  if (chain.empty() || !normalize_associative_rpn_chain(chain, normalized))
    return in;
  out.insert(out.end(), normalized.begin(), normalized.end());
  return out;
}
// A chain rewritten into another event's operand frame.  Both fusion passes
// build one of these before deciding whether the merge fits.

inline uint8_t operand_push_op(std::vector<detail::VectorDescRef>& inputs,
                               const detail::VectorDescRef& vec) {
  if (!inputs.empty() && inputs[0] == vec) return OP_PUSH_INPUT;
  for (size_t i = 1; i < inputs.size(); ++i)
    if (inputs[i] == vec) return (uint8_t)(OP_PUSH_OPERAND_0 + (i - 1));
  if (inputs.empty()) {
    inputs.push_back(vec);
    return OP_PUSH_INPUT;
  }
  if (inputs.size() < MAX_COMBINED_INPUTS) {
    inputs.push_back(vec);
    // The new entry sits at index n, which is operand slot n-1.
    return (uint8_t)(OP_PUSH_OPERAND_0 + (inputs.size() - 2));
  }
  return PUSH_OP_BUDGET_EXCEEDED;
}

// Copies `in[i]` and its inline bytes to `out`, stopping at the end of the
// program.  Advances `i` past what it consumed.
inline void append_token_with_inline_bytes(const std::vector<uint8_t>& in,
                                           size_t& i,
                                           std::vector<uint8_t>& out) {
  out.push_back(in[i]);
  for (size_t b = 0; b < OP_INLINE_BYTES(in[i]) && i + 1 < in.size(); ++b)
    out.push_back(in[++i]);
}

// Splices a mapped chain into `last`'s program.  Returns false, leaving `last`
// untouched, when the merge would exceed the RPN or scalar budget.

}  // namespace detail
#endif  // PIPELINE
