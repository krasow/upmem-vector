#include <defs.h>
#include <limits.h>
#include <mram.h>
#include <stdint.h>

#include "../Param.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 12
#endif

#define ROW_BYTES (((DIM * sizeof(T)) + 7) & ~7)
#define BLOCK_ROWS 8

typedef struct {
  uint32_t data_offset;
  uint32_t centroids_offset;
  uint32_t result_offset;
  uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;
__dma_aligned T tasklet_centroids[NR_TASKLETS][K * DIM];
__dma_aligned RED_T tasklet_partials[NR_TASKLETS][K][DIM + 1];
__dma_aligned T
    tasklet_row_bufs[NR_TASKLETS][BLOCK_ROWS * (ROW_BYTES / sizeof(T))];

int main(void) {
  unsigned int tid = me();
  uint32_t n = args.num_elements;

  T *centroids = tasklet_centroids[tid];
  mram_read((__mram_ptr void const *)(uintptr_t)args.centroids_offset,
            centroids, (K * DIM * sizeof(T) + 7) & ~7u);

  RED_T(*partial)[DIM + 1] = tasklet_partials[tid];
  for (uint32_t j = 0; j < K; j++) {
    for (uint32_t d = 0; d <= DIM; d++) {
      partial[j][d] = 0;
    }
  }

  T *row_buf = tasklet_row_bufs[tid];

  __mram_ptr void *data_base = (__mram_ptr void *)(uintptr_t)args.data_offset;

  for (uint32_t i = tid * BLOCK_ROWS; i < n; i += NR_TASKLETS * BLOCK_ROWS) {
    uint32_t cnt = (i + BLOCK_ROWS <= n) ? BLOCK_ROWS : (n - i);
    uint32_t bytes = (cnt * ROW_BYTES + 7) & ~7u;
    mram_read((__mram_ptr void const *)((uint8_t *)data_base +
                                        (uint64_t)i * ROW_BYTES),
              row_buf, bytes);

    for (uint32_t r = 0; r < cnt; r++) {
      T *row = &row_buf[r * (ROW_BYTES / sizeof(T))];

      uint32_t best_j = 0;
      RED_T best_dist = INT32_MAX;
      for (uint32_t j = 0; j < K; j++) {
        RED_T sq_dist = 0;
        for (uint32_t d = 0; d < DIM; d++) {
          RED_T diff = (RED_T)row[d] - (RED_T)centroids[j * DIM + d];
          sq_dist += diff * diff;
        }
        if (sq_dist < best_dist) {
          best_dist = sq_dist;
          best_j = j;
        }
      }
      for (uint32_t d = 0; d < DIM; d++) {
        partial[best_j][d] += (RED_T)row[d];
      }
      partial[best_j][DIM]++;
    }
  }

  /* K*(DIM+1)*sizeof(RED_T) = 10*11*4 = 440 bytes, divisible by 8 */
  uint32_t partial_bytes = K * (DIM + 1) * sizeof(RED_T);
  mram_write(
      partial,
      (__mram_ptr void *)(uintptr_t)(args.result_offset + tid * partial_bytes),
      partial_bytes);
  return 0;
}
