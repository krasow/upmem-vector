#ifndef MAP_TO_VAL_FUNC_H
#define MAP_TO_VAL_FUNC_H

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "../../../multitask_classifier_common.h"
#include "../Param.h"
#include "processing/gen_red/GenRedArgs.h"

__dma_aligned T state_buf[SVM_STATE_WORDS];
BARRIER_INIT(classifier_state_barrier, NR_TASKLETS);

void start_func(gen_red_arguments_t* args) {
  if (me() == 0) {
    uint32_t bytes = (SVM_STATE_WORDS * sizeof(T) + 7u) & ~7u;
    mram_read((__mram_ptr void const*)(DPU_MRAM_HEAP_POINTER + args->info),
              state_buf, bytes);
  }
  barrier_wait(&classifier_state_barrier);
}

void map_to_val_func(void* input, void* output, uint32_t* key) {
  T* row = (T*)input;
  *key = 0;

  if (state_buf[0] == SVM_MODE_EVALUATE) {
    int32_t best_score = 0;
    uint32_t best_class = 0;
    for (uint32_t c = 0; c < CLASSES; c++) {
      int32_t score = 0;
      for (uint32_t d = 0; d < FEATURES; d++) {
        score += state_buf[3 + c * FEATURES + d] * row[d];
      }
      if (c == 0 || score > best_score) {
        best_score = score;
        best_class = c;
      }
    }
    *(RED_T*)output = best_class == (uint32_t)row[FEATURES] ? 1 : 0;
    return;
  }

  uint32_t classifier = (uint32_t)state_buf[1];
  uint32_t statistic = (uint32_t)state_buf[2];
  int32_t score = 0;
  for (uint32_t d = 0; d < FEATURES; d++) {
    score += state_buf[3 + d] * row[d];
  }

  int32_t label = svm_signed_label(row[FEATURES], classifier);
  int active = label * score < SVM_MARGIN;
  if (statistic < FEATURES) {
    *(RED_T*)output = active ? (RED_T)(-label * row[statistic]) : 0;
  } else {
    *(RED_T*)output = active ? 1 : 0;
  }
}

#endif
