#pragma once
// Fusion mechanics at the event level: turning an Event into an RPN program,
// splicing a mapped chain into another event, and the shared fusion logging.
//
// The decisions that use these live in host/vfuse.cc and host/hfuse.cc.

#include <detail/rpn.h>
#include <jit.h>
#include <logger.h>
#include <queue.h>

#include <iterator>
#include <memory>
#include <string>
#include <vector>

#if PIPELINE
namespace detail {

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

struct MappedChain {
  bool ok = false;
  std::vector<uint8_t> rpn;                   // ops in the target's frame
  std::vector<detail::VectorDescRef> inputs;  // merged operand table
};

// Finds `vec` in an operand table, appending it when there is room.  Slot 0 is
// the primary input (OP_PUSH_INPUT); the rest are operand slots.  Returns
// PUSH_OP_BUDGET_EXCEEDED when the table is full.

inline bool splice_mapped_chain(const std::shared_ptr<Event>& last,
                                const std::vector<uint8_t>& last_rpn,
                                const std::vector<uint32_t>& last_scalars,
                                const std::vector<uint32_t>& chain_scalars,
                                const MappedChain& mapped) {
  if (!mapped.ok) return false;
  if (last_rpn.size() + mapped.rpn.size() > MAX_VFUSE_OPS) return false;
  if (last_scalars.size() + chain_scalars.size() > MAX_PIPELINE_SCALARS)
    return false;

  last->rpn_ops = last_rpn;
  last->rpn_ops.insert(last->rpn_ops.end(), mapped.rpn.begin(),
                       mapped.rpn.end());
  last->rpn_ops = normalize_associative_rpn(last->rpn_ops);
  last->scalars = last_scalars;
  last->scalars.insert(last->scalars.end(), chain_scalars.begin(),
                       chain_scalars.end());
  last->inputs = mapped.inputs;
  return true;
}

// Moves `e`'s identity onto the event that absorbed it: its id range, the
// dependencies implied by its inputs, and ownership of its outputs.
inline void adopt_fused_event(const std::shared_ptr<Event>& last,
                              const std::shared_ptr<Event>& e) {
  last->max_id = std::max(last->max_id, e->id);
  last->kid = last->pipeline_kid;

  // Fusion only merges adjacent queued events, so [last->id, last->max_id]
  // covers the events `last` now stands for (including redundant producers
  // retired between them).  A dependency inside that range is on something
  // absorbed, which will never complete on its own -- waiting for it
  // deadlocks.  This happens with in-place ops, where an input and the output
  // are the same vector, so its last_producer_id names an event we swallowed.
  auto absorbed = [&last](size_t id) {
    return id >= last->id && id <= last->max_id;
  };
  for (const auto& in : e->inputs) {
    if (!in || in->last_producer_id == 0) continue;
    if (absorbed(in->last_producer_id)) continue;
    last->dependencies.insert(in->last_producer_id);
  }
  // An earlier merge may have recorded an id that only now became absorbed.
  for (auto it = last->dependencies.begin(); it != last->dependencies.end();)
    it = absorbed(*it) ? last->dependencies.erase(it) : std::next(it);
  if (e->output) e->output->last_producer_id = last->id;
  for (auto& out : e->extra_outputs)
    if (out) out->last_producer_id = last->id;
}

// Perfetto slice name for a fused kernel.
inline std::string fused_pipeline_label(const std::vector<uint8_t>& rpn,
                                        const char* prefix = "Fused") {
  FusionRpnSummary summary = summarize_fusion_rpn(rpn);
  std::string label = std::string(prefix) +
                      " Pipeline (ops=" + std::to_string(summary.decoded_ops) +
                      ", bytes=" + std::to_string(rpn.size());
  if (summary.chains > 1) label += ", chains=" + std::to_string(summary.chains);
  if (summary.reductions > 0)
    label += ", reductions=" + std::to_string(summary.reductions);
  label += ")";
  return label;
}

// The element type an event's kernel is compiled for.
inline const char* event_raw_type_name(const std::shared_ptr<Event>& e) {
  if (e->output && e->output->type_name) return e->output->type_name;
  if (!e->inputs.empty() && e->inputs[0]) return e->inputs[0]->type_name;
  return nullptr;
}

#if JIT
inline Signature event_kernel_signature(const std::shared_ptr<Event>& e) {
  return Signature{e->rpn_ops, jit_canonical_type_name(event_raw_type_name(e))};
}

// Identifies the kernel an RPN program will compile to, for fusion logs.
inline std::string rpn_kernel_hash(const std::vector<uint8_t>& rpn,
                                   const char* raw_type_name) {
  return jit_signature_hash(
      Signature{rpn, jit_canonical_type_name(raw_type_name)});
}

inline std::string fused_kernel_hash(const std::shared_ptr<Event>& e) {
  return rpn_kernel_hash(e->rpn_ops, event_raw_type_name(e));
}
#endif

// The two events' programs in canonical RPN form, with their scalar tables.
struct FusionOperands {
  std::vector<uint8_t> target_rpn;
  std::vector<uint32_t> target_scalars;
  std::vector<uint8_t> chain_rpn;
  std::vector<uint32_t> chain_scalars;
};

inline FusionOperands build_fusion_operands(const std::shared_ptr<Event>& last,
                                            const std::shared_ptr<Event>& e) {
  FusionOperands ops;
  build_default_rpn(last, ops.target_rpn, ops.target_scalars);
  build_default_rpn(e, ops.chain_rpn, ops.chain_scalars);
  return ops;
}

// Shape of the target before a merge, for the fusion log.
struct FusionBefore {
  size_t inputs = 0;
  size_t extra_outputs = 0;
  size_t target_scalars = 0;
  size_t chain_scalars = 0;
};

#if ENABLE_DPU_LOGGING >= 1
// The opening lines shared by both fusion log entries.
inline void log_fusion_header(Logger::Lock& log, const char* title,
                              const char* reason,
                              const std::shared_ptr<Event>& last,
                              const std::shared_ptr<Event>& e,
                              const FusionBefore& before) {
  log.first() << title;
  log.second() << "child #" << e->id << "..#" << e->max_id << " -> fused #"
               << last->id << "..#" << last->max_id
               << "  deps=" << last->dependencies.size();
  log.second() << "reason=" << reason;
  log.second() << "shape inputs=" << before.inputs << "+" << e->inputs.size()
               << "=>" << last->inputs.size()
               << "  extra_outputs=" << before.extra_outputs << "=>"
               << last->extra_outputs.size()
               << "  scalars=" << before.target_scalars << "+"
               << before.chain_scalars << "=>" << last->scalars.size();
}

// The trailing lines shared by every fusion log entry.
inline void log_fused_kernel_tail(Logger::Lock& log,
                                  const std::shared_ptr<Event>& last,
                                  const FusionRpnSummary& summary) {
  log.second() << "fused expr: " << fusion_rpn_expr_preview(last->rpn_ops);
  log.second() << "kernel after: " << fusion_rpn_short(summary)
#if JIT
               << "  kernel_hash=" << fused_kernel_hash(last)
#endif
               << "  opcode mix: " << fusion_op_counts(summary) << std::endl;
}
#endif

}  // namespace detail
#endif  // PIPELINE
