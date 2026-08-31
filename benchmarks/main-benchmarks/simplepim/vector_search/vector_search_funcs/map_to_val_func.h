#ifndef MAP_TO_VAL_FUNC_H
#define MAP_TO_VAL_FUNC_H

#include <barrier.h>
#include <defs.h>
#include <mram.h>

#include "../Param.h"
#include "processing/gen_red/GenRedArgs.h"

#define QUERY_WORDS ((DIM + 1) & ~1u)

__dma_aligned T query_buf[QUERY_WORDS];
BARRIER_INIT(barrier_vector_search, NR_TASKLETS);

void start_func(gen_red_arguments_t *args) {
  if (me() == 0) {
    mram_read((__mram_ptr void const *)(DPU_MRAM_HEAP_POINTER + args->info),
              query_buf, QUERY_WORDS * sizeof(T));
  }
  barrier_wait(&barrier_vector_search);
}

void map_to_val_func(void *input, void *output, uint32_t *key) {
  *key = 0;
  T *record = (T *)input;
  int32_t score = 0;
  for (uint32_t d = 0; d < DIM; ++d) {
    score += record[d] * query_buf[d];
  }

  vector_search_result_t *candidate = (vector_search_result_t *)output;
  vector_search_result_init(candidate);
  candidate->score = score;
  candidate->index = (uint32_t)record[DIM];
}

#endif
