#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include "../../../multitask_classifier_common.h"
#include "../Param.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 12
#endif

#define BLOCK_ROWS 8u
#define RESULT_WORDS (FLOW_FEATURES + 2u)

typedef struct {
  uint32_t rows_offset;
  uint32_t weights_offset;
  uint32_t result_offset;
  uint32_t num_elements;
  uint32_t mode;
  uint32_t classifier;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;
__dma_aligned T tasklet_weights[NR_TASKLETS][FLOW_CLASSES * FLOW_FEATURES];
__dma_aligned RED_T tasklet_results[NR_TASKLETS][RESULT_WORDS];
__dma_aligned T tasklet_rows[NR_TASKLETS][BLOCK_ROWS * FLOW_ROW_WORDS];

int main(void) {
  uint32_t tid = me();
  T* weights = tasklet_weights[tid];
  RED_T* result = tasklet_results[tid];
  T* rows = tasklet_rows[tid];

  uint32_t weight_count = args.mode == SVM_MODE_EVALUATE
                              ? FLOW_CLASSES * FLOW_FEATURES
                              : FLOW_FEATURES;
  uint32_t weight_offset =
      args.mode == SVM_MODE_EVALUATE ? 0 : args.classifier * FLOW_FEATURES;
  mram_read((__mram_ptr T*)(uintptr_t)args.weights_offset + weight_offset,
            weights, weight_count * sizeof(T));

  for (uint32_t i = 0; i < RESULT_WORDS; i++) result[i] = 0;

  __mram_ptr T* input = (__mram_ptr T*)(uintptr_t)args.rows_offset;
  for (uint32_t first = tid * BLOCK_ROWS; first < args.num_elements;
       first += NR_TASKLETS * BLOCK_ROWS) {
    uint32_t count = first + BLOCK_ROWS <= args.num_elements
                         ? BLOCK_ROWS
                         : args.num_elements - first;
    uint32_t bytes = count * FLOW_ROW_WORDS * sizeof(T);
    mram_read(input + (uint64_t)first * FLOW_ROW_WORDS, rows, bytes);

    for (uint32_t r = 0; r < count; r++) {
      T* row = &rows[r * FLOW_ROW_WORDS];
      if (args.mode == SVM_MODE_EVALUATE) {
        int32_t best_score = 0;
        uint32_t best_class = 0;
        for (uint32_t c = 0; c < FLOW_CLASSES; c++) {
          int32_t score = 0;
          for (uint32_t d = 0; d < FLOW_FEATURES; d++)
            score += weights[c * FLOW_FEATURES + d] * row[d];
          if (c == 0 || score > best_score) {
            best_score = score;
            best_class = c;
          }
        }
        result[0] += best_class == (uint32_t)row[FLOW_FEATURES];
        continue;
      }

      int32_t score = 0;
      for (uint32_t d = 0; d < FLOW_FEATURES; d++) score += weights[d] * row[d];
      int32_t label = svm_signed_label(row[FLOW_FEATURES], args.classifier);
      if (label * score < SVM_MARGIN) {
        for (uint32_t d = 0; d < FLOW_FEATURES; d++)
          result[d] += (RED_T)(-label * row[d]);
        result[FLOW_FEATURES]++;
      }
    }
  }

  mram_write(result,
             (__mram_ptr void*)(uintptr_t)(args.result_offset +
                                           tid * RESULT_WORDS * sizeof(RED_T)),
             RESULT_WORDS * sizeof(RED_T));
  return 0;
}
