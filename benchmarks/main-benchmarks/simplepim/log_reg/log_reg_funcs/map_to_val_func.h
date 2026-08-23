#ifndef MAP_TO_VAL_FUNC_H
#define MAP_TO_VAL_FUNC_H

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../Param.h"
#include "processing/gen_red/GenRedArgs.h"

__dma_aligned T* weights_data;
BARRIER_INIT(barrier_maptoval, NR_TASKLETS);

void start_func(gen_red_arguments_t* args) {
  uint32_t total_len = args->table_len * args->output_type_size;
  uint32_t aligned_weights_size = total_len + 8 - (total_len % 8);
  if (me() == 0) {
    fsb_allocator_t weights_allocator = fsb_alloc(aligned_weights_size, 1);
    weights_data = (void*)fsb_get(weights_allocator);
    mram_read(DPU_MRAM_HEAP_POINTER + args->info, weights_data,
              aligned_weights_size);
  }
  barrier_wait(&barrier_maptoval);
}

/* A simple integer sigmoid approximation: not used for int version,
   but kept for compatibility.  The log_reg Param.h uses float T so
   this will be compiled as float arithmetic on the DPU. */
static inline T sigmoid_dpu(T x) {
  if (x >= 15.0f) return 1.0f;
  if (x <= -15.0f) return 0.0f;
  if (x == 0.0f) return 0.5f;

  float sum = 1.0f;
  float temp = 1.0f;
  for (uint32_t i = 1; i < 101; ++i) {
    temp = temp * (-x) / (float)i;
    sum = sum + temp;
  }
  return (T)(1.0f / (1.0f + sum));
}

void map_to_val_func(void* input, void* grads, uint32_t* dummy) {
  T* grads_ptr = (T*)grads;
  T* input_ptr = (T*)input;
  T* weights_data_ptr = (T*)weights_data;

  /* compute dot product */
  T dot = 0.0f;
  for (int i = 0; i < dim; i++) {
    dot += input_ptr[i] * weights_data_ptr[i];
  }

  /* gradient of log-loss: (sigmoid(dot) - label) * x_j */
  T e = sigmoid_dpu(dot) - input_ptr[dim];
  for (int i = 0; i < dim; i++) {
    grads_ptr[i] = e * input_ptr[i];
  }
  *dummy = 0;
}

#endif
