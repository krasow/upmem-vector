#include <mram.h>

// Writes a constant across the shard.  Without this the host has to stage a
// full-length buffer and DMA it in, which costs one host round trip of the
// whole vector per fill.
#define DEFINE_FILL_KERNEL(TYPE)                                      \
  int fill_##TYPE(void) {                                             \
    unsigned int tasklet_id = me();                                   \
    uint32_t num_elems = args.num_elements;                           \
    TYPE value = (TYPE)args.binary_scalar.rhs_scalar;                 \
                                                                      \
    __mram_ptr TYPE *res_ptr =                                        \
        (__mram_ptr TYPE *)(args.binary_scalar.res_offset);           \
    TYPE *res_block = (TYPE *)dpu_workspace[tasklet_id];              \
                                                                      \
    for (uint32_t i = 0; i < BLOCK_SIZE; i++) res_block[i] = value;   \
                                                                      \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;          \
         block_loc < num_elems;                                       \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {             \
      uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems)    \
                                 ? (num_elems - block_loc)            \
                                 : BLOCK_SIZE;                        \
                                                                      \
      uint32_t block_bytes =                                          \
          ((block_elems * sizeof(TYPE)) + 7) & ~(uint32_t)7;          \
                                                                      \
      mram_write(res_block, (__mram_ptr void *)(res_ptr + block_loc), \
                 block_bytes);                                        \
    }                                                                 \
    return 0;                                                         \
  }
