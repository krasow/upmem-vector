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
    // initialise weights
    fsb_allocator_t weights_allocator = fsb_alloc(aligned_weights_size, 1);
    weights_data = (void*)fsb_get(weights_allocator);
    mram_read(DPU_MRAM_HEAP_POINTER + args->info, weights_data,
              aligned_weights_size);
  }
  barrier_wait(&barrier_maptoval);
}

void map_to_val_func(void* input, void* grads, uint32_t* dummy) {
  // the data is preserved and later added to corresponding weights
  RED_T* grads_ptr = (RED_T*)grads;
  T* input_ptr = (T*)input;
  T* weights_data_ptr = (T*)weights_data;

  // calculate gradients w.r.t. linear weights
  RED_T dot_prod = 0;
  for (int i = 0; i < dim; i++) {
    dot_prod += (RED_T)input_ptr[i] * weights_data_ptr[i];
  }

  RED_T e = dot_prod - ((RED_T)input_ptr[dim] << shift_amount);
  // printf("error : %f\n", e*e);
  for (int i = 0; i < dim; i++) {
    grads_ptr[i] = (input_ptr[i] >> (prevent_overflow_shift_amount / 2)) *
                   (e >> (prevent_overflow_shift_amount -
                          prevent_overflow_shift_amount / 2));
  }
  // printf("\n");

  // put weight gradients to the 0th entry
  *dummy = 0;
}

#endif
