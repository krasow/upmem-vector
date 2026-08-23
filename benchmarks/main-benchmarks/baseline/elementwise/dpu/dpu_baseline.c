#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#include "../Param.h"

#define NR_TASKLETS 12

#define BLOCK_SIZE_LOG2 6
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)

BARRIER_INIT(my_barrier, NR_TASKLETS);

typedef struct {
  uint32_t lhs_offset;
  uint32_t rhs_offset;
  uint32_t res_offset;
  uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;

__host uint32_t nb_cycles;

int main(void) {
  unsigned int tasklet_id = me();
  uint32_t num_elems = args.num_elements;

  __mram_ptr T *lhs_ptr = (__mram_ptr T *)(args.lhs_offset);
  __mram_ptr T *rhs_ptr = (__mram_ptr T *)(args.rhs_offset);
  __mram_ptr T *res_ptr = (__mram_ptr T *)(args.res_offset);

  __dma_aligned T lhs_block[BLOCK_SIZE];
  __dma_aligned T rhs_block[BLOCK_SIZE];
  __dma_aligned T res_block[BLOCK_SIZE];

  for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;
       block_loc < num_elems; block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {
    uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems)
                               ? (num_elems - block_loc)
                               : BLOCK_SIZE;
    uint32_t block_bytes = block_elems * sizeof(T);

    mram_read((__mram_ptr void const *)(lhs_ptr + block_loc), lhs_block,
              block_bytes);
    mram_read((__mram_ptr void const *)(rhs_ptr + block_loc), rhs_block,
              block_bytes);

    for (uint32_t i = 0; i < block_elems; i++) {
      res_block[i] = OPERATION(lhs_block[i], rhs_block[i]);
    }

    mram_write(res_block, (__mram_ptr void *)(res_ptr + block_loc),
               block_bytes);
  }
  return 0;
}
