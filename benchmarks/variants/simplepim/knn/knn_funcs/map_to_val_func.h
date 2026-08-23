#ifndef MAP_TO_VAL_FUNC_H
#define MAP_TO_VAL_FUNC_H

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <limits.h>
#include <mram.h>
#include <stdint.h>
#include <stdlib.h>

#include "../Param.h"
#include "processing/gen_red/GenRedArgs.h"

__dma_aligned T query_buf[DIM];

BARRIER_INIT(barrier_knn, NR_TASKLETS);

void start_func(gen_red_arguments_t* args) {
  if (me() == 0) {
    mram_read((__mram_ptr void const*)(DPU_MRAM_HEAP_POINTER + args->info),
              query_buf, (DIM * sizeof(T) + 7) & ~7u);
  }
  barrier_wait(&barrier_knn);
}

void map_to_val_func(void* input, void* output, uint32_t* key) {
  *key = 0;
  T* row = (T*)input;
  RED_T sq_dist = 0;
  for (uint32_t d = 0; d < DIM; d++) {
    RED_T diff = (RED_T)row[d] - (RED_T)query_buf[d];
    sq_dist += diff * diff;
  }
  *(RED_T*)output = sq_dist;
}

#endif
