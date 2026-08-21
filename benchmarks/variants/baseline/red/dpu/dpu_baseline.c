#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "../Param.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 12
#endif

#define BLOCK_SIZE 64

typedef struct {
  uint32_t data_offset;
  uint32_t result_offset;
  uint32_t num_elements;
  uint32_t _pad;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;

int main(void) {
  unsigned int tid = me();
  uint32_t n = args.num_elements;

  __mram_ptr T *data = (__mram_ptr T *)(uintptr_t)args.data_offset;
  __mram_ptr RED_T *res = (__mram_ptr RED_T *)(uintptr_t)args.result_offset;

  __dma_aligned T buf[BLOCK_SIZE];
  RED_T local = 0;

  for (uint32_t i = tid * BLOCK_SIZE; i < n; i += NR_TASKLETS * BLOCK_SIZE) {
    uint32_t cnt = (i + BLOCK_SIZE <= n) ? BLOCK_SIZE : (n - i);
    uint32_t bytes = ((cnt * sizeof(T)) + 7) & ~7u;
    mram_read((__mram_ptr void const *)(data + i), buf, bytes);
    for (uint32_t j = 0; j < cnt; j++) local += buf[j];
  }

  res[tid] = local;
  return 0;
}
