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

__dma_aligned void* centroids_data;

BARRIER_INIT(barrier_maptoval, NR_TASKLETS);

void start_func(gen_red_arguments_t* args) {
  /* args->table_len == k, args->output_type_size ==
   * dim*sizeof(T)+sizeof(int32_t) */
  uint32_t total_len = args->table_len * args->output_type_size;
  uint32_t aligned_size = total_len + 8 - (total_len % 8);
  if (me() == 0) {
    fsb_allocator_t alloc = fsb_alloc(aligned_size, 1);
    centroids_data = (void*)fsb_get(alloc);
    mram_read(DPU_MRAM_HEAP_POINTER + args->info, centroids_data, aligned_size);
  }
  barrier_wait(&barrier_maptoval);
}

void map_to_val_func(void* input_point, void* intermediate_input,
                     uint32_t* centroid) {
  /* intermediate layout: [int32_t count | T coords[dim]] */
  int32_t* times = (int32_t*)intermediate_input;
  *times = 1;
  T* intermediate_ptr = (T*)((char*)intermediate_input + sizeof(int32_t));
  T* input_point_ptr = (T*)input_point;
  T* centroids_data_ptr = (T*)centroids_data;

  /* copy coordinates into intermediate (will be summed in combine) */
  for (int i = 0; i < dim; i++) {
    intermediate_ptr[i] = input_point_ptr[i];
  }

  /* find nearest centroid */
  uint32_t best = 0;
  uint64_t best_dist = (uint64_t)-1;
  for (int i = 0; i < k; i++) {
    uint64_t dist = 0;
    for (int j = 0; j < dim; j++) {
      T tmp = input_point_ptr[j] - centroids_data_ptr[i * dim + j];
      dist += (uint64_t)(tmp * tmp);
    }
    if (dist < best_dist) {
      best = i;
      best_dist = dist;
    }
  }
  *centroid = best;
}

#endif
