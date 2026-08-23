#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "common.h"
#include "config.h"

namespace detail {
struct VectorSegment {
  uint32_t ptr;
  uint32_t size_bytes;       // Logical bytes (used for count)
  uint32_t allocated_bytes;  // Physical bytes (aligned to 8)
};

struct VectorDesc {
  uint64_t vector_id = 0;
  size_t num_elements = 0;    // total number of elements
  uint32_t element_size = 0;  // sizeof(T)
  uint32_t reserved_bytes = 0;
  size_t allocated_footprint_bytes = 0;
  /// ...

  // Sharded per DPU.
  std::vector<VectorSegment> desc;
  bool needs_layout_materialization = false;

  bool is_reduction_result = false;
  KernelID reduction_rid;

  bool ptr_allocated = false;
  size_t last_producer_id = 0;

  // Live dpu_vector handles pointing here.  Counted explicitly rather than
  // inferred from shared_ptr::use_count(), because absorbed_inputs on *other*
  // descriptors also hold references and would be mistaken for user handles.
  // Zero means no caller can submit a new consumer for this vector.
  size_t handle_count = 0;

  // When this vector is absorbed by vertical fusion (i.e., it is an
  // intermediate result that is consumed inline and never written to MRAM),
  // these fields record the RPN prefix and scalar values that produce it,
  // along with the primary source vector.  Later events that need this vector
  // can inline the prefix rather than reading from (unwritten) MRAM.
  std::vector<uint8_t> absorbed_rpn;
  std::vector<uint32_t> absorbed_scalars;
  std::vector<std::shared_ptr<VectorDesc>>
      absorbed_inputs;  // full input list of absorbed event

  // True when this vector is a shared intermediate written by a standalone
  // kernel (e.g. error_shifted consumed by DIM gradient chains).  Prevents
  // try_vfuse from absorbing it on-stack, which would skip the MRAM write.
  bool is_shared_intermediate = false;

  bool is_local_vector = false;
  uint8_t local_reduce_opcode = OP_SUM;

  const char* type_name = nullptr;
  const char* debug_name = nullptr;
  const char* debug_file = nullptr;
  int debug_line = -1;
  virtual ~VectorDesc();
};

using VectorDescRef = std::shared_ptr<VectorDesc>;

// How a vector's per-DPU shards are laid out for a host transfer.
//
// A shard holds `logical` payload bytes but occupies `align8(...)` in MRAM, and
// dpu_push_xfer applies ONE size to the whole DPU set.  So a transfer must use
// a single stride that covers every shard, and the host buffer must be padded
// to match; the payload is then compacted out of it.  When every shard's
// payload already equals the stride, no padding is needed and the transfer can
// go straight into the caller's buffer.
struct ShardLayout {
  size_t stride = 0;            // uniform transfer size per DPU, in bytes
  size_t total_logical = 0;     // sum of the payload bytes across shards
  std::vector<size_t> logical;  // payload bytes per DPU
  bool needs_padding = false;   // any shard's payload is smaller than stride

  size_t padded_bytes() const { return stride * logical.size(); }
};

inline ShardLayout shard_layout(const VectorDesc& desc) {
  ShardLayout out;
  out.logical.reserve(desc.desc.size());
  for (const auto& segment : desc.desc) {
    const size_t payload = segment.size_bytes > desc.reserved_bytes
                               ? segment.size_bytes - desc.reserved_bytes
                               : 0;
    const size_t occupied = segment.allocated_bytes > desc.reserved_bytes
                                ? segment.allocated_bytes - desc.reserved_bytes
                                : 0;
    out.logical.push_back(payload);
    out.total_logical += payload;
    if (occupied > out.stride) out.stride = occupied;
  }
  for (size_t payload : out.logical)
    if (payload != out.stride) out.needs_padding = true;
  return out;
}

// Implemented in vector.cc.
void vec_xfer_to_dpu(char* cpu, VectorDescRef desc);
void vec_xfer_from_dpu_strided(char* cpu, VectorDescRef desc, size_t stride);
}  // namespace detail
