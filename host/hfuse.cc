#include <detail/fusion.h>
#include <perfetto/detail/trace.h>
#include <perfetto/trace.h>

#include "jit.h"
#include "runtime.h"
#include "stats.h"

#if PIPELINE

namespace {
size_t count_reduction_chains(const std::vector<uint8_t>& rpn) {
  size_t count = 0;
  bool chain_has_reduction = false;
  for (size_t i = 0; i < rpn.size(); ++i) {
    uint8_t op = rpn[i];
    if (op == OP_NEXT_CHAIN) {
      if (chain_has_reduction) count++;
      chain_has_reduction = false;
      continue;
    }
    if (IS_OP_REDUCTION(op)) chain_has_reduction = true;
    if (OP_INLINE_BYTES(op) > 0) i += OP_INLINE_BYTES(op);
  }
  if (chain_has_reduction) count++;
  return count;
}

// Rewrites an independent chain onto the target's operand table, prefixed with
// the OP_NEXT_CHAIN separator that starts a new WRAM chain.
//
// Unlike the vertical case there is no value already on the stack, so every
// operand must be reachable through the merged table: a push that cannot be
// mapped fails the whole merge rather than being dropped.
detail::MappedChain map_independent_chain(
    const std::vector<uint8_t>& chain_rpn,
    const std::vector<detail::VectorDescRef>& chain_inputs,
    const std::vector<detail::VectorDescRef>& target_inputs,
    size_t target_scalar_count) {
  detail::MappedChain mapped;
  mapped.inputs = target_inputs;
  mapped.rpn.push_back(OP_NEXT_CHAIN);

  auto push_op_for = [&](const detail::VectorDescRef& vec) -> uint8_t {
    if (!vec) return PUSH_OP_ALREADY_ON_STACK;  // nothing to push: unmappable
    return detail::operand_push_op(mapped.inputs, vec);
  };
  auto unmappable = [](uint8_t push) {
    return push == PUSH_OP_ALREADY_ON_STACK || push == PUSH_OP_BUDGET_EXCEEDED;
  };

  for (size_t k = 0; k < chain_rpn.size(); ++k) {
    uint8_t op = chain_rpn[k];
    if (op == OP_PUSH_INPUT) {
      uint8_t push = push_op_for(chain_inputs[0]);
      if (unmappable(push)) return {};
      mapped.rpn.push_back(push);

    } else if (op >= OP_PUSH_OPERAND_0 &&
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {
      size_t idx = op - OP_PUSH_OPERAND_0 + 1;
      if (idx >= chain_inputs.size()) return {};
      uint8_t push = push_op_for(chain_inputs[idx]);
      if (unmappable(push)) return {};
      mapped.rpn.push_back(push);

    } else if (IS_OP_SCALAR_VAR(op) || op == OP_PUSH_SCALAR_VAR) {
      // Scalar slots shift up past the target's scalar table.
      mapped.rpn.push_back(op);
      if (k + 1 < chain_rpn.size())
        mapped.rpn.push_back((uint8_t)(target_scalar_count + chain_rpn[++k]));

    } else if (OP_INLINE_BYTES(op) > 0) {
      detail::append_token_with_inline_bytes(chain_rpn, k, mapped.rpn);

    } else {
      mapped.rpn.push_back(op);
    }
  }

  mapped.ok = true;
  return mapped;
}

// Not fused_pipeline_label(): the horizontal label always reports the chain
// count and never the reduction count.
std::string horizontal_slice_name(const detail::FusionRpnSummary& summary) {
  return "Horiz-Fused Pipeline (ops=" + std::to_string(summary.decoded_ops) +
         ", bytes=" + std::to_string(summary.bytes) +
         ", chains=" + std::to_string(summary.chains) + ")";
}

void log_horizontal_fusion(const std::shared_ptr<Event>& last,
                           const std::shared_ptr<Event>& e,
                           const detail::FusionBefore& before,
                           const detail::FusionOperands& ops,
                           const detail::FusionRpnSummary& summary) {
#if ENABLE_DPU_LOGGING >= 1
  Logger& logger = DpuRuntime::get().get_logger();
  if (!logger.enabled(2)) return;

  auto log = logger.lock(logcat::FUSION);
  detail::log_fusion_header(log, "horizontal fusion", "independent_same_length",
                            last, e, before);
  log.second() << "existing expr: "
               << detail::fusion_rpn_expr_preview(ops.target_rpn, 1, 90);
  log.second() << "new expr: "
               << detail::fusion_rpn_expr_preview(ops.chain_rpn, 1, 90);
  detail::log_fused_kernel_tail(log, last, summary);
#else
  (void)last;
  (void)e;
  (void)before;
  (void)ops;
  (void)summary;
#endif
}

}  // namespace

// Horizontal fusion: last and e are independent chains over equal-length
// vectors.  Both run in the same kernel pass as separate WRAM chains.
bool EventQueue::try_hfuse(std::shared_ptr<Event> last,
                           std::shared_ptr<Event> e) {
  if (last->extra_outputs.size() >= MAX_HFUSE_CHAINS - 1) return false;

  const detail::FusionOperands ops = detail::build_fusion_operands(last, e);
  const detail::FusionBefore before{
      last->inputs.size(), last->extra_outputs.size(),
      ops.target_scalars.size(), ops.chain_scalars.size()};
  if (count_reduction_chains(ops.target_rpn) +
          count_reduction_chains(ops.chain_rpn) >
      MAX_SAFE_HFUSED_REDUCTION_CHAINS)
    return false;

  detail::MappedChain mapped = map_independent_chain(
      ops.chain_rpn, e->inputs, last->inputs, ops.target_scalars.size());
  if (!detail::splice_mapped_chain(last, ops.target_rpn, ops.target_scalars,
                                   ops.chain_scalars, mapped))
    return false;
  last->extra_outputs.push_back(e->output);
  detail::adopt_fused_event(last, e);

  detail::FusionRpnSummary summary =
      detail::summarize_fusion_rpn(last->rpn_ops);
  last->slice_name = horizontal_slice_name(summary);
  log_horizontal_fusion(last, e, before, ops, summary);

  VECTORDPU_NOTE(horizontal_fusions);
  trace::event_fused(e, last, "");
  trace::inqueue_end(e);
  return true;
}

#endif  // PIPELINE
