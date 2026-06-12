#pragma once
// Internal helpers shared by vfuse.cc, hfuse.cc, and jit.cc.
// Not part of the public API.

#include <cstdint>

#include "common.h"
#include "opinfo.h"
#include "perfetto/trace.h"
#include "perfetto/trace_internal.h"
#include "queue.h"

#if PIPELINE
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
inline void build_default_rpn(const std::shared_ptr<Event>& e,
                              std::vector<uint8_t>& rpn,
                              std::vector<uint32_t>& scalars) {
  rpn = e->rpn_ops;
  scalars = e->scalars;
  if (!rpn.empty()) return;
  if (!e->inputs.empty()) rpn.push_back(OP_PUSH_INPUT);
  for (size_t i = 1; i < e->inputs.size(); ++i)
    rpn.push_back(OP_PUSH_OPERAND_0 + (i - 1));
  if (e->is_scalar) {
    rpn.push_back(map_to_var_op(e->opcode));
    rpn.push_back(0);
    scalars.push_back(e->scalar_value);
  } else {
    rpn.push_back(e->opcode);
  }
}

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
                                         size_t& i,
                                         std::vector<uint8_t>& out) {
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
      if (chain.empty() ||
          !normalize_associative_rpn_chain(chain, normalized))
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
#endif
