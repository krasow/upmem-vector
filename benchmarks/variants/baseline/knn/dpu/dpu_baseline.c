#include <defs.h>
#include <limits.h>
#include <mram.h>
#include <stdint.h>

#include "../Param.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 12
#endif

// Row size in bytes (DIM int32_t values, padded to 8-byte multiple)
#define ROW_BYTES (((DIM * 4) + 7) & ~7)
// Process this many rows per MRAM read
#define BLOCK_ROWS 16

typedef struct {
  uint32_t data_offset;
  uint32_t query_offset;
  uint32_t result_offset;
  uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;

int main(void) {
  unsigned int tid = me();
  uint32_t n = args.num_elements;

  __mram_ptr void *data_base = (__mram_ptr void *)(uintptr_t)args.data_offset;
  __mram_ptr T *query_ptr = (__mram_ptr T *)(uintptr_t)args.query_offset;
  __mram_ptr RED_T *result_ptr =
      (__mram_ptr RED_T *)(uintptr_t)(args.result_offset + tid * 8);

  // Load query into WRAM
  __dma_aligned T query[DIM];
  mram_read((__mram_ptr void const *)query_ptr, query,
            ((DIM * sizeof(T)) + 7) & ~7u);

  __dma_aligned T row_buf[BLOCK_ROWS * (ROW_BYTES / 4)];

  RED_T local_min = INT32_MAX;

  for (uint32_t i = tid * BLOCK_ROWS; i < n; i += NR_TASKLETS * BLOCK_ROWS) {
    uint32_t cnt = (i + BLOCK_ROWS <= n) ? BLOCK_ROWS : (n - i);
    uint32_t bytes = cnt * ROW_BYTES;
    mram_read((__mram_ptr void const *)((uint8_t *)data_base +
                                        (uint64_t)i * ROW_BYTES),
              row_buf, bytes);

    for (uint32_t r = 0; r < cnt; r++) {
      T *row = &row_buf[r * (ROW_BYTES / sizeof(T))];
      RED_T sq_dist = 0;
      for (uint32_t d = 0; d < DIM; d++) {
        RED_T diff = (RED_T)row[d] - (RED_T)query[d];
        sq_dist += diff * diff;
      }
      if (sq_dist < local_min) local_min = sq_dist;
    }
  }

  __dma_aligned RED_T write_buf[2] = {local_min, 0};
  mram_write(write_buf, (__mram_ptr void *)result_ptr, 8);
  return 0;
}
