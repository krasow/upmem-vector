#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>

#include "../Param.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 12
#endif

#define BLOCK_SIZE 64

typedef struct {
  uint32_t data_offset;     // MRAM offset for (X, y) data
  uint32_t weights_offset;  // MRAM offset for weights
  uint32_t results_offset;  // MRAM offset for partial gradients
  uint32_t num_elements;    // Number of rows (samples) on this DPU
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;

BARRIER_INIT(my_barrier, NR_TASKLETS);

int main(void) {
  unsigned int tasklet_id = me();
  uint32_t num_rows = args.num_elements;
  uint32_t padded_row_size = ((dim + 1) * sizeof(T) + 7) & ~7;
  uint32_t weights_size = dim * sizeof(T);
  uint32_t red_size = dim * sizeof(RED_T);

  __mram_ptr void *data_ptr = (__mram_ptr void *)(uintptr_t)args.data_offset;
  __mram_ptr void *weights_ptr =
      (__mram_ptr void *)(uintptr_t)args.weights_offset;
  __mram_ptr void *results_ptr =
      (__mram_ptr void *)(uintptr_t)args.results_offset;

  // Load weights into WRAM
  __dma_aligned T w_cache[dim];
  mram_read(weights_ptr, w_cache, weights_size);

  // Partial gradients for this tasklet
  __dma_aligned RED_T local_grads[dim];
  for (uint32_t i = 0; i < dim; i++) {
    local_grads[i] = 0;
  }

  __dma_aligned T row_cache[(padded_row_size + sizeof(T) - 1) / sizeof(T)];

  // Work division: each tasklet processes a subset of rows
  for (uint32_t i = tasklet_id; i < num_rows; i += NR_TASKLETS) {
    mram_read(data_ptr + i * padded_row_size, row_cache, padded_row_size);

    RED_T dot_prod = 0;
    for (uint32_t j = 0; j < dim; j++) {
      dot_prod += (RED_T)row_cache[j] * w_cache[j];
    }

    RED_T e =
        (dot_prod >> shift_amount) - ((RED_T)row_cache[dim] << shift_amount);

    for (uint32_t j = 0; j < dim; j++) {
      local_grads[j] +=
          (RED_T)((row_cache[j] >> (prevent_overflow_shift_amount / 2)) *
                  (e >> (prevent_overflow_shift_amount -
                         prevent_overflow_shift_amount / 2)));
    }
  }

  // Write back tasklet-local gradients to MRAM
  // Each tasklet writes to its own slice: results_ptr + tasklet_id * red_size
  mram_write(local_grads, results_ptr + tasklet_id * red_size, red_size);

  return 0;
}
