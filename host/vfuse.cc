#include <detail/fusion.h>
#include <perfetto/trace.h>
#include <perfetto/trace_internal.h>

#include <algorithm>

#include "jit.h"
#include "runtime.h"
#include "stats.h"

#if PIPELINE

namespace {
bool rpn_contains_indirect(const std::vector<uint8_t>& rpn) {
  for (size_t i = 0; i < rpn.size(); ++i) {
    uint8_t op = rpn[i];
    if (op == OP_LOAD_INDIRECT || op == OP_ADD_INDIRECT ||
        op == OP_APPLY_INDIRECT || op == OP_PUSH_INDEX ||
        op == OP_PUSH_GLOBAL_INDEX)
      return true;
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }
  return false;
}

// Whether `e` is the tail of an accumulator chain -- its last value-producing
// opcode is a bare ADD or an arithmetic-shift-by-scalar.
//
// This is the one case where absorbing a shared intermediate on-stack is still
// correct: linreg's error accumulator forms `previous_error + dx[j]*w[j]`, and
// the previous value is consumed and immediately replaced by the next one, so
// nothing else can observe the skipped MRAM write.  Product and reduction
// chains reading a shared intermediate are not safe and must stay rejected.
bool consumes_accumulator_chain(const std::shared_ptr<Event>& e) {
  bool ends_with_add = e->rpn_ops.empty() && e->opcode == OP_ADD;
  bool ends_with_asr_scalar = e->rpn_ops.empty() && e->opcode == OP_ASR_SCALAR;

  for (size_t k = 0; k < e->rpn_ops.size(); ++k) {
    uint8_t op = e->rpn_ops[k];
    if (IS_OP_SCALAR(op)) {
      ends_with_asr_scalar = op == OP_ASR_SCALAR;
      k += SCALAR_INLINE_BYTES;
    } else if (IS_OP_SCALAR_VAR(op)) {
      ends_with_asr_scalar = op == OP_ASR_SCALAR_VAR;
      k += SCALAR_VAR_INDEX_BYTES;
    } else if (op == OP_PUSH_SCALAR || op == OP_PUSH_SCALAR_VAR ||
               op == OP_LOAD_INDIRECT || op == OP_ADD_INDIRECT ||
               op == OP_APPLY_INDIRECT) {
      k += OP_INLINE_BYTES(op);
    } else if (IS_OP_ARG_K(op)) {
      // Carries an inline k byte; skip it so the scan stays aligned.  An arg
      // op is never a bare accumulator collapse.
      k += OP_INLINE_BYTES(op);
      ends_with_add = false;
      ends_with_asr_scalar = false;
    } else if (op == OP_ADD) {
      ends_with_add = true;
      ends_with_asr_scalar = false;
    } else if (IS_OP_BINARY(op) || IS_OP_UNARY(op) || IS_OP_REDUCTION(op) ||
               IS_OP_TERNARY(op)) {
      ends_with_add = false;
      ends_with_asr_scalar = false;
    }
  }
  return ends_with_add || ends_with_asr_scalar;
}

bool is_commutative(uint8_t op) {
  return op == OP_ADD || op == OP_MUL || op == OP_EQ;
}

void log_vertical_fusion(const std::shared_ptr<Event>& last,
                         const std::shared_ptr<Event>& e,
                         const detail::FusionBefore& before,
                         const detail::FusionOperands& ops) {
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  if (!logger.enabled(2)) return;

  auto log = logger.lock(logcat::FUSION);
  detail::log_fusion_header(log, "vertical fusion", "dependent_on_stack_output",
                            last, e, before);
  log.second() << "producer expr: "
               << detail::fusion_rpn_expr_preview(ops.target_rpn, 1, 90);
  log.second() << "consumer expr: "
               << detail::fusion_rpn_expr_preview(ops.chain_rpn, 1, 90,
                                                  "producer_out", 1)
               << "  [producer_out = producer expr]";
  detail::log_fused_kernel_tail(log, last,
                                detail::summarize_fusion_rpn(last->rpn_ops));
#else
  (void)last;
  (void)e;
  (void)before;
  (void)ops;
#endif
}

// Rewrites the consumer's operand pushes onto the producer's operand table.
// The producer's result is already on the WRAM stack, so a consumer push of it
// is dropped (and a second reference becomes an explicit DUP).  Returns a
// failed MappedChain when the operand budget is exhausted or when the rewrite
// would reorder a non-commutative operator's arguments.
detail::MappedChain map_consumer_onto_producer(
    const std::vector<uint8_t>& consumer_rpn,
    const std::vector<detail::VectorDescRef>& consumer_inputs,
    const std::vector<detail::VectorDescRef>& producer_inputs,
    const detail::VectorDescRef& on_stack, size_t producer_scalar_count) {
  detail::MappedChain mapped;
  mapped.inputs = producer_inputs;

  auto push_op_for = [&](const detail::VectorDescRef& vec) -> uint8_t {
    if (vec == on_stack) return PUSH_OP_ALREADY_ON_STACK;
    return detail::operand_push_op(mapped.inputs, vec);
  };

  bool primary_on_stack = false;
  bool stacked_without_primary = false;

  for (size_t k = 0; k < consumer_rpn.size(); ++k) {
    uint8_t op = consumer_rpn[k];
    if (op == OP_PUSH_INPUT) {
      uint8_t push = push_op_for(consumer_inputs[0]);
      if (push == PUSH_OP_BUDGET_EXCEEDED) return {};
      if (push == PUSH_OP_ALREADY_ON_STACK)
        primary_on_stack = true;
      else
        mapped.rpn.push_back(push);

    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      size_t idx = op - OP_PUSH_OPERAND_0 + 1;
      if (idx >= consumer_inputs.size()) return {};
      uint8_t push = push_op_for(consumer_inputs[idx]);
      if (push == PUSH_OP_BUDGET_EXCEEDED) return {};
      if (push == PUSH_OP_ALREADY_ON_STACK) {
        if (primary_on_stack) mapped.rpn.push_back(OP_DUP);
        stacked_without_primary = true;
      } else {
        mapped.rpn.push_back(push);
      }

    } else if (IS_OP_SCALAR_VAR(op)) {
      // Scalar slots shift up by the producer's scalar table size.
      mapped.rpn.push_back(op);
      if (k + 1 < consumer_rpn.size())
        mapped.rpn.push_back(
            (uint8_t)(producer_scalar_count + consumer_rpn[++k]));

    } else if (OP_INLINE_BYTES(op) > 0) {
      detail::append_token_with_inline_bytes(consumer_rpn, k, mapped.rpn);

    } else {
      // The only stacked operand is the right-hand one, so a non-commutative
      // operator would apply with its arguments swapped.
      if (stacked_without_primary && IS_OP_BINARY(op) && !is_commutative(op))
        return {};
      stacked_without_primary = false;
      mapped.rpn.push_back(op);
    }
  }

  mapped.ok = true;
  return mapped;
}

uint8_t get_or_add_push_op(std::vector<detail::VectorDescRef>& inputs,
                           const detail::VectorDescRef& vec) {
  if (!vec) return PUSH_OP_BUDGET_EXCEEDED;
  return detail::operand_push_op(inputs, vec);
}

void append_inline_scalar(std::vector<uint8_t>& rpn, uint8_t op,
                          uint32_t scalar) {
  rpn.push_back(op);
  rpn.push_back((uint8_t)(scalar & 0xFF));
  rpn.push_back((uint8_t)((scalar >> 8) & 0xFF));
  rpn.push_back((uint8_t)((scalar >> 16) & 0xFF));
  rpn.push_back((uint8_t)((scalar >> 24) & 0xFF));
}

bool append_absorbed_rpn_inline(const detail::VectorDescRef& vec,
                                std::vector<detail::VectorDescRef>& inputs,
                                std::vector<uint8_t>& out) {
  if (!vec || vec->absorbed_rpn.empty() || vec->absorbed_inputs.empty())
    return false;

  for (size_t i = 0; i < vec->absorbed_rpn.size(); ++i) {
    uint8_t op = vec->absorbed_rpn[i];
    if (op == OP_PUSH_INPUT) {
      uint8_t push = get_or_add_push_op(inputs, vec->absorbed_inputs[0]);
      if (push == PUSH_OP_BUDGET_EXCEEDED) return false;
      out.push_back(push);
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      size_t idx = op - OP_PUSH_OPERAND_0 + 1;
      if (idx >= vec->absorbed_inputs.size()) return false;
      uint8_t push = get_or_add_push_op(inputs, vec->absorbed_inputs[idx]);
      if (push == PUSH_OP_BUDGET_EXCEEDED) return false;
      out.push_back(push);
    } else if (IS_OP_SCALAR_VAR(op)) {
      if (i + 1 >= vec->absorbed_rpn.size()) return false;
      uint8_t scalar_idx = vec->absorbed_rpn[++i];
      if (scalar_idx >= vec->absorbed_scalars.size()) return false;
      append_inline_scalar(out, detail::map_from_var_op(op),
                           vec->absorbed_scalars[scalar_idx]);
    } else if (op == OP_LOAD_INDIRECT || IS_OP_INDIRECT_UPDATE(op)) {
      return false;
    } else if (OP_INLINE_BYTES(op) > 0) {
      out.push_back(op);
      for (size_t b = 0;
           b < OP_INLINE_BYTES(op) && i + 1 < vec->absorbed_rpn.size(); ++b)
        out.push_back(vec->absorbed_rpn[++i]);
    } else {
      out.push_back(op);
    }
  }
  return true;
}
// Whether `e` reads `vec` as one of its inputs.
bool reads_input(const std::shared_ptr<Event>& e,
                 const detail::VectorDescRef& vec) {
  for (const auto& in : e->inputs)
    if (in == vec) return true;
  return false;
}

// Absorbing a value on-stack skips its MRAM write, so it is only safe while at
// most one queued event references it -- the producer's own output slot.
//
// This is what makes a *retroactive* fusion reject a multi-consumer named
// output: by then the consumer is itself queued, so it contributes references
// of its own and the count exceeds one.  With linreg's `error_shifted` read by
// 10 reductions, try_vfuse(accumulator, first_reduction) bails here and the
// MRAM materialisation is kept.  Fusion during submit passes cleanly because
// the consumer is not in the queue yet.
//
// (The original form added the consumer's own input refs to both sides of the
// comparison, which cancels; this is the same predicate written out.)
bool absorbing_is_safe(const detail::VectorDescRef& vec,
                       size_t internal_reference_count) {
  return !vec || internal_reference_count <= 1;
}

void log_absorb_producer(const std::shared_ptr<Event>& consumer,
                         size_t erased_id, size_t erased_max_id,
                         size_t deps_before) {
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  if (!logger.enabled(2)) return;
  auto log = logger.lock(logcat::FUSION);
  log.first() << "absorb producer";
  log.second() << "producer #" << erased_id << "..#" << erased_max_id
               << " -> consumer #" << consumer->id << "..#" << consumer->max_id
               << "  deps=" << deps_before << "=>"
               << consumer->dependencies.size();
  log.second() << "reason=inline_absorbed_input" << std::endl;
#else
  (void)consumer;
  (void)erased_id;
  (void)erased_max_id;
  (void)deps_before;
#endif
}

// An absorbed producer inlined into its consumer.
struct InlinedProgram {
  bool ok = false;
  std::vector<detail::VectorDescRef> inputs;
  std::vector<uint8_t> rpn;
  std::vector<uint32_t> scalars;
};

// Gives `e` an explicit RPN when it is still a bare opcode, so the inliner has
// something to rewrite.
void ensure_explicit_rpn(const std::shared_ptr<Event>& e) {
  if (!e->rpn_ops.empty()) return;
  for (size_t k = 0; k < e->inputs.size(); ++k)
    e->rpn_ops.push_back(k == 0 ? OP_PUSH_INPUT : OP_PUSH_OPERAND_0 + (k - 1));
  if (e->is_scalar) {
    e->rpn_ops.push_back(detail::map_to_var_op(e->opcode));
    e->rpn_ops.push_back(0);
    e->scalars.push_back(e->scalar_value);
  } else {
    e->rpn_ops.push_back(e->opcode);
  }
}

// Consumer uses random access (`vec[idx]`), so the absorbed producer can only
// be inlined where the consumer would have loaded it: the PUSH_INDEX +
// LOAD_INDIRECT(slot 0) pair.  Anything else still needs the producer in MRAM.
InlinedProgram inline_through_indirect(const std::shared_ptr<Event>& e,
                                       const detail::VectorDescRef& producer) {
  InlinedProgram result;
  bool rewritten = false;

  for (size_t k = 0; k < e->rpn_ops.size(); ++k) {
    uint8_t op = e->rpn_ops[k];
    if (op == OP_PUSH_INDEX && k + 2 < e->rpn_ops.size() &&
        e->rpn_ops[k + 1] == OP_LOAD_INDIRECT && e->rpn_ops[k + 2] == 0) {
      if (!append_absorbed_rpn_inline(producer, result.inputs, result.rpn))
        return {};
      rewritten = true;
      k += 2;
    } else if (op == OP_LOAD_INDIRECT) {
      return {};  // indirect load of something else: needs real MRAM
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      size_t idx = op - OP_PUSH_OPERAND_0 + 1;
      if (idx >= e->inputs.size()) return {};
      uint8_t push = get_or_add_push_op(result.inputs, e->inputs[idx]);
      if (push == PUSH_OP_BUDGET_EXCEEDED) return {};
      result.rpn.push_back(push);
    } else if (op == OP_PUSH_INPUT) {
      return {};  // a direct read of the absorbed vector alongside a load
    } else if (IS_OP_SCALAR_VAR(op)) {
      result.rpn.push_back(op);
      if (k + 1 < e->rpn_ops.size()) result.rpn.push_back(e->rpn_ops[++k]);
    } else if (OP_INLINE_BYTES(op) > 0) {
      result.rpn.push_back(op);
      for (size_t b = 0; b < OP_INLINE_BYTES(op) && k + 1 < e->rpn_ops.size();
           ++b)
        result.rpn.push_back(e->rpn_ops[++k]);
    } else {
      result.rpn.push_back(op);
    }
  }

  if (!rewritten) return {};
  result.scalars = e->scalars;
  result.ok = true;
  return result;
}

// Consumer reads the absorbed vector as its primary input, so its RPN is
// spliced in wherever the consumer pushes that input, and the consumer's own
// operand and scalar slots shift up past the producer's.
InlinedProgram inline_directly(const std::shared_ptr<Event>& e,
                               const detail::VectorDescRef& producer) {
  InlinedProgram result;
  const std::vector<detail::VectorDescRef>& producer_inputs =
      producer->absorbed_inputs;
  const size_t operand_shift = producer_inputs.size();
  const size_t scalar_shift = producer->absorbed_scalars.size();

  result.inputs.reserve(operand_shift + e->inputs.size() - 1);
  for (const auto& vec : producer_inputs) result.inputs.push_back(vec);
  for (size_t k = 1; k < e->inputs.size(); ++k)
    result.inputs.push_back(e->inputs[k]);
  if (result.inputs.size() > MAX_COMBINED_INPUTS) return {};

  result.scalars = producer->absorbed_scalars;
  for (size_t k = 0; k < e->rpn_ops.size(); ++k) {
    uint8_t op = e->rpn_ops[k];
    if (op == OP_PUSH_INPUT) {
      result.rpn.insert(result.rpn.end(), producer->absorbed_rpn.begin(),
                        producer->absorbed_rpn.end());
    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      uint8_t slot = op - OP_PUSH_OPERAND_0;
      result.rpn.push_back(OP_PUSH_OPERAND_0 +
                           (uint8_t)(operand_shift - 1 + slot));
    } else if (IS_OP_SCALAR_VAR(op)) {
      result.rpn.push_back(op);
      if (k + 1 < e->rpn_ops.size())
        result.rpn.push_back(e->rpn_ops[++k] + (uint8_t)scalar_shift);
    } else {
      result.rpn.push_back(op);
    }
  }
  result.scalars.insert(result.scalars.end(), e->scalars.begin(),
                        e->scalars.end());
  result.ok = true;
  return result;
}

void log_inline_input(const std::shared_ptr<Event>& e,
                      const detail::VectorDescRef& producer,
                      size_t inputs_before, size_t scalars_before,
                      const std::vector<uint8_t>& rpn_before,
                      const InlinedProgram& result) {
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  if (!logger.enabled(2)) return;

  detail::FusionRpnSummary after = detail::summarize_fusion_rpn(result.rpn);
  auto log = logger.lock(logcat::FUSION);
  log.first() << "inline input";
  log.second() << "producer #" << producer->last_producer_id << " -> consumer #"
               << e->id;
  log.second() << "reason=indirect_load_of_absorbed_vector";
  log.second() << "shape inputs=" << inputs_before << "=>"
               << result.inputs.size() << "  scalars=" << scalars_before << "=>"
               << result.scalars.size();
  log.second() << "consumer expr before: "
               << detail::fusion_rpn_expr_preview(rpn_before, 1, 90);
  log.second() << "consumer expr after: "
               << detail::fusion_rpn_expr_preview(result.rpn);
  log.second() << "kernel after: " << detail::fusion_rpn_short(after)
#if JIT
               << "  kernel_hash="
               << jit_signature_hash(Signature{
                      result.rpn,
                      jit_canonical_type_name(e->output ? e->output->type_name
                                                        : nullptr)})
#endif
               << "  opcode mix: " << detail::fusion_op_counts(after)
               << std::endl;
#else
  (void)producer;
  (void)inputs_before;
  (void)scalars_before;
  (void)rpn_before;
  (void)result;
  fprintf(stderr,
          "[VFUSE] inlined absorbed input into indirect consumer id=%zu\n",
          e->id);
#endif
}

}  // namespace

// Inline the RPN of an absorbed intermediate into `e` so it can be computed
// without reading from (unwritten) MRAM.  Called from EventQueue::submit before
// the event is enqueued.
void EventQueue::expand_absorbed_inputs(std::shared_ptr<Event> e) {
  if (e->op != Event::OperationType::COMPUTE || e->inputs.empty()) return;

  auto& in_vec = e->inputs[0];
  if (!in_vec || in_vec->absorbed_rpn.empty() ||
      in_vec->absorbed_inputs.empty())
    return;
  if (in_vec->last_producer_id != 0 &&
      get_last_finished_id() >= in_vec->last_producer_id) {
    in_vec->absorbed_rpn.clear();
    in_vec->absorbed_scalars.clear();
    in_vec->absorbed_inputs.clear();
    in_vec->is_shared_intermediate = false;
    return;
  }
  ensure_explicit_rpn(e);

  const bool contains_indirect = rpn_contains_indirect(e->rpn_ops);
  const size_t inputs_before = e->inputs.size();
  const size_t scalars_before = e->scalars.size();
  const std::vector<uint8_t> rpn_before = e->rpn_ops;

  InlinedProgram inlined = contains_indirect
                               ? inline_through_indirect(e, in_vec)
                               : inline_directly(e, in_vec);
  if (!inlined.ok) return;

    // MAX_VFUSE_OPS is the size of the generic interpreter's args.pipeline.ops
    // buffer, so it only binds when there is no JIT: a generated kernel has its
    // program baked into C and can be any length.  Without this the interpreter
    // silently truncated the tail of an over-long inlined program.
#if !JIT
  if (inlined.rpn.size() > MAX_VFUSE_OPS) return;
#endif
  if (contains_indirect)
    log_inline_input(e, in_vec, inputs_before, scalars_before, rpn_before,
                     inlined);

  if (inlined.inputs.size() > MAX_COMBINED_INPUTS) return;

  std::vector<uint8_t> normalized_rpn =
      detail::normalize_associative_rpn(inlined.rpn);
#if JIT_PIPELINE_FALLBACK
  if (!detail::pipeline_can_interpret(normalized_rpn)) return;
#endif

  // Clear absorbed state — future ops that read this vector get it from MRAM.
  auto absorbed_vec = std::move(in_vec);
  e->inputs = std::move(inlined.inputs);
  e->rpn_ops = std::move(normalized_rpn);
  e->scalars = std::move(inlined.scalars);
  e->is_scalar = false;
  if (absorbed_vec->last_producer_id != 0)
    e->absorbed_producer_ids.insert(absorbed_vec->last_producer_id);
  absorbed_vec->absorbed_rpn.clear();
  absorbed_vec->absorbed_scalars.clear();
  absorbed_vec->absorbed_inputs.clear();
  absorbed_vec->is_shared_intermediate = false;

  // absorbed_vec's producer may now be redundant: e recomputes it inline.  But
  // whether it is *actually* redundant cannot be known here -- a consumer that
  // needs its MRAM output may not have been submitted yet, and a temporary
  // holding the vector may not have died yet.  So only mark it, and let
  // EventQueue::output_still_needed decide once temporary handles have gone
  // out of scope and every submitted consumer is visible.
  for (auto& op : operations_)
    if (op->output == absorbed_vec && op->extra_outputs.empty()) {
      op->output_was_inlined = true;
      op->inlined_into = e->id;
    }
}

// Vertical fusion: e depends on last's output (on-stack value).
// Merges e's RPN into last so both run in one kernel pass.
bool EventQueue::try_vfuse(std::shared_ptr<Event> last,
                           std::shared_ptr<Event> e) {
  if (!last->rpn_ops.empty() && IS_OP_REDUCTION(last->rpn_ops.back()))
    return false;

  // `e` may be a dpu_jit_foreach kernel using indirect ops (LOAD_INDIRECT /
  // ADD_INDIRECT / PUSH_INDEX).  Those opcodes expect the producer's output
  // to exist in MRAM for random-access loads; vfuse would absorb it
  // on-stack and leave the MRAM slot unwritten, so the indirect access
  // reads garbage (hist segfault on `sweep.py --hist`).  Also, the vfuse
  // rpn rewriter doesn't remap the single-byte operand index that follows
  // LOAD_INDIRECT, so after merging it would still reference slot 0 of a
  // combined inputs list that no longer corresponds to the absorbed vec.
  if (rpn_contains_indirect(e->rpn_ops)) return false;

  // Safety: the on-stack value is the last chain's output.
  detail::VectorDescRef on_stack =
      last->extra_outputs.empty() ? last->output : last->extra_outputs.back();

  // If on_stack is a shared intermediate (e.g. error_shifted consumed by DIM
  // gradient chains), absorbing it on-stack would skip the MRAM write and
  // corrupt subsequent readers.  The linreg error accumulator is the one
  // exception we deliberately support here: each update forms
  //   previous_error + (dx[j] * scalar[j])
  // where the previous_error value is consumed by an ADD chain and replaced by
  // the next accumulator value.  Keeping this case fusable lets the
  // accumulator collapse into one JIT kernel while still rejecting product and
  // reduction chains that read shared materialized intermediates.
  const bool accumulator_tail = consumes_accumulator_chain(e);
  if (on_stack && on_stack->is_shared_intermediate && !accumulator_tail)
    return false;

  if (!accumulator_tail &&
      !absorbing_is_safe(on_stack, count_internal_references(on_stack)))
    return false;

  // There is nothing to fuse unless e actually consumes the stacked value.
  if (!reads_input(e, on_stack)) return false;

  // e must not also read one of the producer's *other* chain outputs; only the
  // stacked one is available without a materialised MRAM buffer.
  for (const auto& in : e->inputs) {
    if (!in || in == on_stack) continue;
    if (in == last->output) return false;
    for (const auto& out : last->extra_outputs)
      if (in == out) return false;
  }

  const detail::FusionOperands ops = detail::build_fusion_operands(last, e);
  const detail::FusionBefore before{
      last->inputs.size(), last->extra_outputs.size(),
      ops.target_scalars.size(), ops.chain_scalars.size()};

  detail::MappedChain mapped =
      map_consumer_onto_producer(ops.chain_rpn, e->inputs, last->inputs,
                                 on_stack, ops.target_scalars.size());
  if (!detail::splice_mapped_chain(last, ops.target_rpn, ops.target_scalars,
                                   ops.chain_scalars, mapped))
    return false;

  if (last->extra_outputs.empty())
    last->output = e->output;
  else
    last->extra_outputs.back() = e->output;

  // Record the merged program on the chain's output so a future consumer can
  // inline it instead of waiting on the pre-fused intermediate.
  //
  // Only for a single-chain event.  Once the event is horizontally fused its
  // rpn_ops describe *every* chain, separated by OP_NEXT_CHAIN, and that is not
  // a recipe for any one output -- splicing it into a consumer's expression
  // yields nonsense.  `(a+b)*c - (d-a)` returned `d-a` because the subtraction
  // inlined a two-chain program as if it were one value.
  const bool single_chain =
      last->extra_outputs.empty() &&
      std::find(last->rpn_ops.begin(), last->rpn_ops.end(),
                (uint8_t)OP_NEXT_CHAIN) == last->rpn_ops.end();
  if (single_chain && last->output && !last->inputs.empty()) {
    last->output->absorbed_rpn = last->rpn_ops;
    last->output->absorbed_scalars = last->scalars;
    last->output->absorbed_inputs = last->inputs;
  }

  detail::adopt_fused_event(last, e);

  last->slice_name = detail::fused_pipeline_label(last->rpn_ops);
  log_vertical_fusion(last, e, before, ops);

  VECTORDPU_NOTE(vertical_fusions);
  trace::event_fused(e, last, "");
  trace::inqueue_end(e);
  return true;
}

#endif  // PIPELINE
