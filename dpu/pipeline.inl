#include <barrier.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// STACK_DEPTH and MINIMUM_WRITE_SIZE are defined in common.h

#define DEFINE_UNIVERSAL_PIPELINE_KERNEL(TYPE)                                 \
  int universal_##TYPE##_pipeline(void) {                                      \
    unsigned int id = me();                                                    \
    uint32_t n = args.num_elements, n_ops = args.pipeline.num_ops;             \
    __mram_ptr TYPE *in_ptr = (__mram_ptr TYPE *)(args.pipeline.init_offset);  \
    __mram_ptr TYPE *rs_ptr = (__mram_ptr TYPE *)(args.pipeline.res_offset);   \
    __mram_ptr TYPE *res_ptrs[MAX_HFUSE_CHAINS];                               \
    res_ptrs[0] = rs_ptr;                                                      \
    for (int r = 1; r < MAX_HFUSE_CHAINS; r++)                                 \
      res_ptrs[r] = (__mram_ptr TYPE *)(args.pipeline.extra_res_offsets[r - 1]);\
                                                                               \
    /* Workspace Layout: input(0), operands(1-3), scratch(4) */                \
    TYPE *input_blk = (TYPE *)dpu_workspace[id];                               \
    TYPE *op_blks[MAX_VFUSE_INPUTS];                                           \
    for (int k = 0; k < MAX_VFUSE_INPUTS; k++)                                 \
      op_blks[k] = (TYPE *)&dpu_workspace[id][(k + 1) * BLOCK_SIZE *           \
                                              MINIMUM_WRITE_SIZE];             \
    TYPE(*scratch_blks)                                                        \
    [BLOCK_SIZE] = (TYPE(*)[BLOCK_SIZE]) &                                     \
                   dpu_workspace[id][(MAX_VFUSE_INPUTS + 1) * BLOCK_SIZE *     \
                                     MINIMUM_WRITE_SIZE];                      \
                                                                               \
    int64_t acc_64[MAX_HFUSE_CHAINS];                                          \
    TYPE acc[MAX_HFUSE_CHAINS];                                                \
    bool has_r[MAX_HFUSE_CHAINS] = {false};                                    \
    uint8_t r_op[MAX_HFUSE_CHAINS] = {0};                                      \
    uint32_t blk, i, b_e, b_b, oi;                                             \
                                                                               \
    /* Pre-scan for operands and reductions */                                 \
    bool uses_input = false;                                                   \
    bool uses_op[MAX_VFUSE_INPUTS] = {false};                                  \
    int max_used_op = -1;                                                       \
    uint32_t scan_chain = 0;                                                    \
    oi = 0;                                                                    \
    while (oi < n_ops) {                                                       \
      uint8_t op = args.pipeline.ops[oi];                                      \
      if (op == OP_NEXT_CHAIN) {                                               \
        if (scan_chain + 1 < MAX_HFUSE_CHAINS) scan_chain++;                   \
        oi++;                                                                  \
        continue;                                                              \
      }                                                                        \
      if (IS_OP_SCALAR(op)) {                                                  \
        oi += 5; /* Opcode + 4 bytes scalar */                                 \
        continue;                                                              \
      }                                                                        \
      if (IS_OP_SCALAR_VAR(op)) {                                              \
        oi += 2; /* Opcode + 1 byte scalar index */                            \
        continue;                                                              \
      }                                                                        \
      if (op == OP_PUSH_SCALAR_VAR) {                                          \
        oi += 2; /* Opcode + 1 byte scalar index */                            \
        continue;                                                              \
      }                                                                        \
      if (IS_OP_ARG_K(op)) { oi += 2; continue; } /* Opcode + 1 byte k */      \
      if (op == OP_PUSH_INPUT)                                                 \
        uses_input = true;                                                     \
      else if (op >= OP_PUSH_OPERAND_0 &&                                      \
               op < OP_PUSH_OPERAND_0 + MAX_VFUSE_INPUTS) {                    \
        int op_idx = op - OP_PUSH_OPERAND_0;                                   \
        uses_op[op_idx] = true;                                                \
        if (op_idx > max_used_op) max_used_op = op_idx;                        \
      }                                                                        \
      else if (IS_OP_REDUCTION(op)) {                                          \
        r_op[scan_chain] = op;                                                 \
        has_r[scan_chain] = true;                                              \
      }                                                                        \
      oi++;                                                                    \
    }                                                                          \
                                                                               \
    for (uint32_t c = 0; c < MAX_HFUSE_CHAINS; c++) {                          \
      if (!has_r[c]) continue;                                                 \
      switch (r_op[c]) {                                                       \
        case OP_SUM:                                                           \
          acc_64[c] = 0;                                                       \
          acc[c] = (TYPE)0;                                                    \
          break;                                                               \
        case OP_PRODUCT:                                                       \
          acc_64[c] = 1;                                                       \
          acc[c] = (TYPE)1;                                                    \
          break;                                                               \
        case OP_MIN:                                                           \
          acc[c] = (TYPE)INT32_MAX;                                            \
          break;                                                               \
        case OP_MAX:                                                           \
          acc[c] = (TYPE)INT32_MIN;                                            \
          break;                                                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (blk = id << BLOCK_SIZE_LOG2; blk < n;                                 \
         blk += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                            \
      b_e = (blk + BLOCK_SIZE >= n) ? (n - blk) : BLOCK_SIZE;                  \
      b_b = b_e * sizeof(TYPE);                                                \
                                                                               \
      /* 1. Fetch operands (with deduplication) */                             \
      if (uses_input)                                                          \
        mram_read((__mram_ptr void const *)(in_ptr + blk), input_blk, b_b);    \
      for (int k = 0; k <= max_used_op; k++) {                                 \
        if (uses_op[k]) {                                                      \
          __mram_ptr TYPE *p =                                                 \
              (__mram_ptr TYPE *)(args.pipeline.binary_operands[k]);           \
          bool found = false;                                                  \
          if (uses_input && p == in_ptr) {                                     \
            op_blks[k] = input_blk;                                            \
            found = true;                                                      \
          }                                                                    \
          for (int j = 0; j < k; j++) {                                        \
            if (uses_op[j] &&                                                  \
                p == (__mram_ptr TYPE *)(args.pipeline.binary_operands[j])) {  \
              op_blks[k] = op_blks[j];                                         \
              found = true;                                                    \
              break;                                                           \
            }                                                                  \
          }                                                                    \
          if (!found)                                                          \
            mram_read((__mram_ptr void const *)(p + blk), op_blks[k], b_b);    \
        }                                                                      \
      }                                                                        \
                                                                               \
      /* 2. Pointer-based Horizontal Fusion (Loop of Loops) */                 \
      TYPE *st_ptr[MAX_PIPELINE_STACK_DEPTH];                                  \
      bool st_is_temp[MAX_PIPELINE_STACK_DEPTH];                               \
      uint32_t sp = 0;                                                         \
      uint32_t chain_idx = 0;                                                   \
                                                                               \
      oi = 0;                                                                  \
      while (oi < n_ops) {                                                     \
        uint8_t op = args.pipeline.ops[oi];                                    \
        if (op == OP_NEXT_CHAIN) {                                             \
          if (chain_idx < MAX_HFUSE_CHAINS && !has_r[chain_idx] && sp > 0 &&   \
              res_ptrs[chain_idx])                                             \
            mram_write(st_ptr[sp - 1],                                         \
                       (__mram_ptr void *)(res_ptrs[chain_idx] + blk), b_b);   \
          sp = 0;                                                              \
          if (chain_idx + 1 < MAX_HFUSE_CHAINS) chain_idx++;                   \
          oi++;                                                                \
          continue;                                                            \
        }                                                                      \
        if (IS_OP_SCALAR(op)) {                                                \
          TYPE *s1 = st_ptr[sp - 1];                                           \
          int32_t val;                                                         \
          /* Manually copy 4 bytes to avoid alignment issues */                \
          uint8_t b0 = args.pipeline.ops[oi + 1];                              \
          uint8_t b1 = args.pipeline.ops[oi + 2];                              \
          uint8_t b2 = args.pipeline.ops[oi + 3];                              \
          uint8_t b3 = args.pipeline.ops[oi + 4];                              \
          val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));           \
          TYPE scalar;                                                         \
          static_assert(sizeof(TYPE) == 4, "Only 32-bit types supported");     \
          memcpy(&scalar, &val, 4);                                            \
                                                                               \
          if (!st_is_temp[sp - 1]) {                                           \
            TYPE *dest = scratch_blks[sp - 1];                                 \
            switch (op) {                                                      \
              case OP_ADD_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] + scalar;            \
                break;                                                         \
              case OP_SUB_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] - scalar;            \
                break;                                                         \
              case OP_MUL_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] * scalar;            \
                break;                                                         \
              case OP_DIV_SCALAR:                                              \
                for (i = 0; i < b_e; i++)                                      \
                  dest[i] = (scalar != (TYPE)0) ? s1[i] / scalar : (TYPE)0;    \
                break;                                                         \
              case OP_ASR_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] >> scalar;           \
                break;                                                         \
              case OP_EQ_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] == scalar);         \
                break;                                                         \
              case OP_LT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] < scalar);          \
                break;                                                         \
              case OP_GT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] > scalar);          \
                break;                                                         \
              case OP_GE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] >= scalar);         \
                break;                                                         \
              case OP_LE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] <= scalar);         \
                break;                                                         \
            }                                                                  \
            st_ptr[sp - 1] = dest;                                             \
            st_is_temp[sp - 1] = true;                                         \
          } else {                                                             \
            switch (op) {                                                      \
              case OP_ADD_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] += scalar;                     \
                break;                                                         \
              case OP_SUB_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] -= scalar;                     \
                break;                                                         \
              case OP_MUL_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] *= scalar;                     \
                break;                                                         \
              case OP_DIV_SCALAR:                                              \
                for (i = 0; i < b_e; i++)                                      \
                  if (scalar != (TYPE)0) s1[i] /= scalar;                      \
                break;                                                         \
              case OP_ASR_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] >>= scalar;                    \
                break;                                                         \
              case OP_EQ_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] == scalar);           \
                break;                                                         \
              case OP_LT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] < scalar);            \
                break;                                                         \
              case OP_GT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] > scalar);            \
                break;                                                         \
              case OP_GE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] >= scalar);           \
                break;                                                         \
              case OP_LE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] <= scalar);           \
                break;                                                         \
            }                                                                  \
          }                                                                    \
          oi += 5;                                                             \
          continue;                                                            \
        }                                                                      \
        if (IS_OP_SCALAR_VAR(op)) {                                            \
          TYPE *s1 = st_ptr[sp - 1];                                           \
          uint8_t idx = args.pipeline.ops[oi + 1];                             \
          TYPE scalar = (TYPE)args.pipeline.scalars[idx];                      \
          uint8_t base =                                                       \
              op - (OP_ADD_SCALAR_VAR - OP_ADD_SCALAR);                        \
                                                                               \
          if (!st_is_temp[sp - 1]) {                                           \
            TYPE *dest = scratch_blks[sp - 1];                                 \
            switch (base) {                                                    \
              case OP_ADD_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] + scalar;            \
                break;                                                         \
              case OP_SUB_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] - scalar;            \
                break;                                                         \
              case OP_MUL_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] * scalar;            \
                break;                                                         \
              case OP_DIV_SCALAR:                                              \
                for (i = 0; i < b_e; i++)                                      \
                  dest[i] = (scalar != (TYPE)0) ? s1[i] / scalar : (TYPE)0;    \
                break;                                                         \
              case OP_ASR_SCALAR:                                              \
                for (i = 0; i < b_e; i++) dest[i] = s1[i] >> scalar;           \
                break;                                                         \
              case OP_EQ_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] == scalar);         \
                break;                                                         \
              case OP_LT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] < scalar);          \
                break;                                                         \
              case OP_GT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] > scalar);          \
                break;                                                         \
              case OP_GE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] >= scalar);         \
                break;                                                         \
              case OP_LE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) dest[i] = (s1[i] <= scalar);         \
                break;                                                         \
            }                                                                  \
            st_ptr[sp - 1] = dest;                                             \
            st_is_temp[sp - 1] = true;                                         \
          } else {                                                             \
            switch (base) {                                                    \
              case OP_ADD_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] += scalar;                     \
                break;                                                         \
              case OP_SUB_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] -= scalar;                     \
                break;                                                         \
              case OP_MUL_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] *= scalar;                     \
                break;                                                         \
              case OP_DIV_SCALAR:                                              \
                for (i = 0; i < b_e; i++)                                      \
                  if (scalar != (TYPE)0) s1[i] /= scalar;                      \
                break;                                                         \
              case OP_ASR_SCALAR:                                              \
                for (i = 0; i < b_e; i++) s1[i] >>= scalar;                    \
                break;                                                         \
              case OP_EQ_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] == scalar);           \
                break;                                                         \
              case OP_LT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] < scalar);            \
                break;                                                         \
              case OP_GT_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] > scalar);            \
                break;                                                         \
              case OP_GE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] >= scalar);           \
                break;                                                         \
              case OP_LE_SCALAR:                                               \
                for (i = 0; i < b_e; i++) s1[i] = (s1[i] <= scalar);           \
                break;                                                         \
            }                                                                  \
          }                                                                    \
          oi += 2;                                                             \
          continue;                                                            \
        }                                                                      \
        if (op == OP_PUSH_SCALAR) {                                            \
          int32_t val;                                                         \
          uint8_t b0 = args.pipeline.ops[oi + 1];                              \
          uint8_t b1 = args.pipeline.ops[oi + 2];                              \
          uint8_t b2 = args.pipeline.ops[oi + 3];                              \
          uint8_t b3 = args.pipeline.ops[oi + 4];                              \
          val = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));           \
          st_ptr[sp] = scratch_blks[sp];                                       \
          st_is_temp[sp] = true;                                               \
          for (i = 0; i < b_e; i++) st_ptr[sp][i] = (TYPE)val;                 \
          sp++;                                                                \
          oi += 5;                                                             \
          continue;                                                            \
        }                                                                      \
        if (op == OP_PUSH_SCALAR_VAR) {                                        \
          uint8_t idx = args.pipeline.ops[oi + 1];                             \
          st_ptr[sp] = scratch_blks[sp];                                       \
          st_is_temp[sp] = true;                                               \
          TYPE scalar = (TYPE)args.pipeline.scalars[idx];                      \
          for (i = 0; i < b_e; i++) st_ptr[sp][i] = scalar;                    \
          sp++;                                                                \
          oi += 2;                                                             \
          continue;                                                            \
        }                                                                      \
        if (op == OP_DUP) {                                                    \
          st_ptr[sp] = st_ptr[sp - 1];                                         \
          st_is_temp[sp] = false;                                              \
          sp++;                                                                \
        } else if (IS_OP_STACK(op)) {                                          \
          st_ptr[sp] = (op == OP_PUSH_INPUT)                                   \
                           ? input_blk                                         \
                           : op_blks[op - OP_PUSH_OPERAND_0];                  \
          st_is_temp[sp] = false;                                              \
          sp++;                                                                \
        } else if (IS_OP_UNARY(op)) {                                          \
          TYPE *s = st_ptr[sp - 1];                                            \
          if (!st_is_temp[sp - 1]) {                                           \
            TYPE *dest = scratch_blks[sp - 1];                                 \
            if (op == OP_NEGATE)                                               \
              for (i = 0; i < b_e; i++) dest[i] = -s[i];                       \
            else                                                               \
              for (i = 0; i < b_e; i++)                                        \
                dest[i] = (s[i] < (TYPE)0) ? -s[i] : s[i];                     \
            st_ptr[sp - 1] = dest;                                             \
            st_is_temp[sp - 1] = true;                                         \
          } else {                                                             \
            if (op == OP_NEGATE)                                               \
              for (i = 0; i < b_e; i++) s[i] = -s[i];                          \
            else                                                               \
              for (i = 0; i < b_e; i++)                                        \
                s[i] = (s[i] < (TYPE)0) ? -s[i] : s[i];                        \
          }                                                                    \
        } else if (IS_OP_BINARY(op)) {                                         \
          TYPE *s1 = st_ptr[--sp];                                             \
          TYPE *s2 = st_ptr[sp - 1];                                           \
          if (!st_is_temp[sp - 1]) {                                           \
            TYPE *dest = scratch_blks[sp - 1];                                 \
            switch (op) {                                                      \
              case OP_ADD:                                                     \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] + s1[i];             \
                break;                                                         \
              case OP_SUB:                                                     \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] - s1[i];             \
                break;                                                         \
              case OP_MUL:                                                     \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] * s1[i];             \
                break;                                                         \
              case OP_DIV:                                                     \
                for (i = 0; i < b_e; i++)                                      \
                  dest[i] = (s1[i] != (TYPE)0) ? s2[i] / s1[i] : (TYPE)0;      \
                break;                                                         \
              case OP_ASR:                                                     \
                for (i = 0; i < b_e; i++) dest[i] = s2[i] >> s1[i];            \
                break;                                                         \
              case OP_EQ:                                                      \
                for (i = 0; i < b_e; i++) dest[i] = (s2[i] == s1[i]);          \
                break;                                                         \
              case OP_LT:                                                      \
                for (i = 0; i < b_e; i++) dest[i] = (s2[i] < s1[i]);           \
                break;                                                         \
              case OP_GT:                                                      \
                for (i = 0; i < b_e; i++) dest[i] = (s2[i] > s1[i]);           \
                break;                                                         \
              case OP_GE:                                                      \
                for (i = 0; i < b_e; i++) dest[i] = (s2[i] >= s1[i]);          \
                break;                                                         \
              case OP_LE:                                                      \
                for (i = 0; i < b_e; i++) dest[i] = (s2[i] <= s1[i]);          \
                break;                                                         \
            }                                                                  \
            st_ptr[sp - 1] = dest;                                             \
            st_is_temp[sp - 1] = true;                                         \
          } else {                                                             \
            switch (op) {                                                      \
              case OP_ADD:                                                     \
                for (i = 0; i < b_e; i++) s2[i] += s1[i];                      \
                break;                                                         \
              case OP_SUB:                                                     \
                for (i = 0; i < b_e; i++) s2[i] -= s1[i];                      \
                break;                                                         \
              case OP_MUL:                                                     \
                for (i = 0; i < b_e; i++) s2[i] *= s1[i];                      \
                break;                                                         \
              case OP_DIV:                                                     \
                for (i = 0; i < b_e; i++)                                      \
                  if (s1[i] != (TYPE)0) s2[i] /= s1[i];                        \
                break;                                                         \
              case OP_ASR:                                                     \
                for (i = 0; i < b_e; i++) s2[i] >>= s1[i];                     \
                break;                                                         \
              case OP_EQ:                                                      \
                for (i = 0; i < b_e; i++) s2[i] = (s2[i] == s1[i]);            \
                break;                                                         \
              case OP_LT:                                                      \
                for (i = 0; i < b_e; i++) s2[i] = (s2[i] < s1[i]);             \
                break;                                                         \
              case OP_GT:                                                      \
                for (i = 0; i < b_e; i++) s2[i] = (s2[i] > s1[i]);             \
                break;                                                         \
              case OP_GE:                                                      \
                for (i = 0; i < b_e; i++) s2[i] = (s2[i] >= s1[i]);            \
                break;                                                         \
              case OP_LE:                                                      \
                for (i = 0; i < b_e; i++) s2[i] = (s2[i] <= s1[i]);            \
                break;                                                         \
            }                                                                  \
          }                                                                    \
        } else if (IS_OP_TERNARY(op)) {                                        \
          TYPE *s1 = st_ptr[--sp];                                             \
          TYPE *s2 = st_ptr[--sp];                                             \
          TYPE *s3 = st_ptr[sp - 1];                                           \
          if (!st_is_temp[sp - 1]) {                                           \
            TYPE *dest = scratch_blks[sp - 1];                                 \
            if (op == OP_SELECT) {                                             \
              for (i = 0; i < b_e; i++)                                        \
                dest[i] = (s3[i] != (TYPE)0) ? s2[i] : s1[i];                  \
            }                                                                  \
            st_ptr[sp - 1] = dest;                                             \
            st_is_temp[sp - 1] = true;                                         \
          } else {                                                             \
            if (op == OP_SELECT) {                                             \
              for (i = 0; i < b_e; i++)                                        \
                s3[i] = (s3[i] != (TYPE)0) ? s2[i] : s1[i];                    \
            }                                                                  \
          }                                                                    \
        } else if (IS_OP_ARG_K(op)) {                                          \
          uint8_t kk = args.pipeline.ops[oi + 1];                              \
          TYPE *arg_out = scratch_blks[sp - kk];                               \
          for (i = 0; i < b_e; i++) {                                          \
            TYPE arg_best = st_ptr[sp - kk][i];                                \
            TYPE arg_idx = (TYPE)0;                                            \
            for (uint8_t jj = 1; jj < kk; jj++) {                              \
              TYPE arg_v = st_ptr[sp - kk + jj][i];                            \
              if (op == OP_ARGMIN_K ? (arg_v < arg_best)                       \
                                    : (arg_v > arg_best)) {                    \
                arg_best = arg_v;                                              \
                arg_idx = (TYPE)jj;                                            \
              }                                                                \
            }                                                                  \
            arg_out[i] = arg_idx;                                              \
          }                                                                    \
          sp -= (kk - 1);                                                      \
          st_ptr[sp - 1] = arg_out;                                            \
          st_is_temp[sp - 1] = true;                                           \
          oi++; /* consume k byte; trailing oi++ consumes the op */            \
        } else { /* REDUCTION */                                               \
          TYPE *s = st_ptr[--sp];                                              \
          switch (op) {                                                        \
            case OP_SUM:                                                       \
              if (ENABLE_PROMOTION_REDUCTIONS && sizeof(TYPE) == 4) {          \
                for (i = 0; i < b_e; i++) acc_64[chain_idx] += s[i];           \
              } else {                                                         \
                for (i = 0; i < b_e; i++) acc[chain_idx] += s[i];              \
              }                                                                \
              break;                                                           \
            case OP_PRODUCT:                                                   \
              if (ENABLE_PROMOTION_REDUCTIONS && sizeof(TYPE) == 4) {          \
                for (i = 0; i < b_e; i++) acc_64[chain_idx] *= s[i];           \
              } else {                                                         \
                for (i = 0; i < b_e; i++) acc[chain_idx] *= s[i];              \
              }                                                                \
              break;                                                           \
            case OP_MIN:                                                       \
              for (i = 0; i < b_e; i++)                                        \
                if (s[i] < acc[chain_idx]) acc[chain_idx] = s[i];              \
              break;                                                           \
            case OP_MAX:                                                       \
              for (i = 0; i < b_e; i++)                                        \
                if (s[i] > acc[chain_idx]) acc[chain_idx] = s[i];              \
              break;                                                           \
          }                                                                    \
        }                                                                      \
        oi++;                                                                  \
      }                                                                        \
      if (chain_idx < MAX_HFUSE_CHAINS && !has_r[chain_idx] && sp > 0 &&       \
          res_ptrs[chain_idx])                                                 \
        mram_write(st_ptr[sp - 1],                                             \
                   (__mram_ptr void *)(res_ptrs[chain_idx] + blk), b_b);       \
    }                                                                          \
                                                                               \
    for (uint32_t c = 0; c < MAX_HFUSE_CHAINS; c++) {                          \
      if (!has_r[c] || !res_ptrs[c]) continue;                                 \
      bool is_promotable = (r_op[c] == OP_SUM || r_op[c] == OP_PRODUCT);       \
      bool is_sum32 =                                                          \
          (is_promotable && sizeof(TYPE) == 4 && ENABLE_PROMOTION_REDUCTIONS); \
      enum { sd = (MINIMUM_WRITE_SIZE / sizeof(TYPE)) };                       \
      uint64_t bf = 0;                                                         \
      if (is_sum32) {                                                          \
        bf = (uint64_t)acc_64[c];                                              \
      } else {                                                                 \
        memcpy(&bf, &acc[c], sizeof(TYPE));                                    \
      }                                                                        \
      extern uint64_t reduction_scratchpad[];                                  \
      reduction_scratchpad[id] = bf;                                           \
      barrier_wait(&my_barrier);                                               \
      if (id == 0) {                                                           \
        if (is_sum32) {                                                        \
          int64_t tot_64 = (r_op[c] == OP_SUM) ? 0 : 1;                        \
          uint32_t i;                                                          \
          for (i = 0; i < NR_TASKLETS; i++) {                                  \
            if (r_op[c] == OP_SUM)                                             \
              tot_64 += (int64_t)reduction_scratchpad[i];                      \
            else                                                               \
              tot_64 *= (int64_t)reduction_scratchpad[i];                      \
          }                                                                    \
          bf = (uint64_t)tot_64;                                               \
        } else {                                                               \
          TYPE res_block_tot[NR_TASKLETS * sd] __attribute__((aligned(8)));    \
          uint32_t i;                                                          \
          for (i = 0; i < NR_TASKLETS; i++) {                                  \
            res_block_tot[i * sd] = *(TYPE *)&reduction_scratchpad[i];         \
          }                                                                    \
          TYPE total = res_block_tot[0];                                       \
          for (i = 1; i < NR_TASKLETS; i++) {                                  \
            TYPE v = res_block_tot[i * sd];                                    \
            switch (r_op[c]) {                                                 \
              case OP_SUM:                                                     \
                total += v;                                                    \
                break;                                                         \
              case OP_PRODUCT:                                                 \
                total *= v;                                                    \
                break;                                                         \
              case OP_MIN:                                                     \
                if (v < total) total = v;                                      \
                break;                                                         \
              case OP_MAX:                                                     \
                if (v > total) total = v;                                      \
                break;                                                         \
            }                                                                  \
          }                                                                    \
          bf = 0;                                                              \
          memcpy(&bf, &total, sizeof(TYPE));                                   \
        }                                                                      \
        mram_write(&bf, (__mram_ptr void *)res_ptrs[c], MINIMUM_WRITE_SIZE);   \
      }                                                                        \
      barrier_wait(&my_barrier);                                               \
    }                                                                          \
    return 0;                                                                  \
  }
