#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "../Param.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 12
#endif

#define ROW_BYTES (DIM * sizeof(T))
#define BLOCK_ROWS 16
#define QUERY_WORDS ((DIM + 1) & ~1u)

typedef struct {
  uint32_t data_offset;
  uint32_t query_offset;
  uint32_t result_offset;
  uint32_t num_elements;
  uint32_t base_index;
  uint32_t reserved;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;
BARRIER_INIT(vector_search_barrier, NR_TASKLETS);

__dma_aligned T query_buf[QUERY_WORDS];
__dma_aligned vector_search_result_t tasklet_best[NR_TASKLETS];
__dma_aligned T row_buffers[NR_TASKLETS][BLOCK_ROWS * DIM];

int main(void) {
  const uint32_t tid = me();
  __mram_ptr uint8_t *data = (__mram_ptr uint8_t *)(uintptr_t)args.data_offset;

  if (tid == 0) {
    mram_read((__mram_ptr void const *)(uintptr_t)args.query_offset, query_buf,
              QUERY_WORDS * sizeof(T));
  }
  vector_search_result_init(&tasklet_best[tid]);
  barrier_wait(&vector_search_barrier);

  T *rows = row_buffers[tid];
  for (uint32_t first = tid * BLOCK_ROWS; first < args.num_elements;
       first += NR_TASKLETS * BLOCK_ROWS) {
    uint32_t count = first + BLOCK_ROWS <= args.num_elements
                         ? BLOCK_ROWS
                         : args.num_elements - first;
    mram_read((__mram_ptr void const *)(data + (uint64_t)first * ROW_BYTES),
              rows, count * ROW_BYTES);

    for (uint32_t r = 0; r < count; ++r) {
      int32_t score = 0;
      for (uint32_t d = 0; d < DIM; ++d)
        score += rows[r * DIM + d] + query_buf[d];
      uint32_t global_index = args.base_index + first + r;
      vector_search_result_insert(
          &tasklet_best[tid],
          vector_search_pack_key(score, global_index, N, DIM));
    }
  }

  barrier_wait(&vector_search_barrier);
  if (tid == 0) {
    for (uint32_t t = 1; t < NR_TASKLETS; ++t)
      vector_search_result_merge(&tasklet_best[0], &tasklet_best[t]);
    mram_write(&tasklet_best[0],
               (__mram_ptr void *)(uintptr_t)args.result_offset,
               sizeof(vector_search_result_t));
  }
  return 0;
}
