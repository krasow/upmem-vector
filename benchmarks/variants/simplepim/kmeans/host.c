#include <assert.h>
#include <benchmark.h>
#include <dpu.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

void init_input(T* elements) {
  /* Synthetic input: each row has dim features */
  for (uint64_t i = 0; i < num_elements; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      elements[i * dim + j] = (T)((i + j) % 1000);
    }
  }
}

int divRoundClosest(const int n, const int d) {
  return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}

void average_table_entries_to_arr(void* centroid_table, T* centroids) {
  for (int i = 0; i < k; i++) {
    int32_t* times =
        (int32_t*)(centroid_table + i * (dim * sizeof(T) + sizeof(int32_t)));
    T* src = (T*)(centroid_table + i * (dim * sizeof(T) + sizeof(int32_t)) +
                  sizeof(int32_t));
    T* dst = centroids + i * dim;
    for (int j = 0; j < dim; j++) {
      dst[j] = (*times == 0) ? 0 : (T)divRoundClosest((int)src[j], (int)*times);
    }
  }
}

void run() {
  BenchStages stages;       // steady-loop stages (+ one-time setup)
  BenchStages warm_stages;  // cold warmup-loop stages (the cold-start premium)
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t* table_management = table_management_init(dpu_number);
  bench_stage_end(&stages);
  printf("k: %d, dim: %d, num_elem: %lu, iter: %d\n", k, dim, num_elements,
         iter);

  T* elements = NULL;
  T* centroids = NULL;

  // Inputs only; the reference result is loaded/compared under check below.
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  elements = (T*)malloc_scatter_aligned(num_elements, dim * sizeof(T),
                                        table_management);
  centroids =
      (T*)malloc_broadcast_aligned(k, sizeof(T) * dim, table_management);
  bench_stage_end(&stages);
  if (load_ref) {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    char path[1024];
    printf("Loading reference data from %s...\n", ref_path);
    sprintf(path, "%s/AoS/rows.bin", ref_path);
    bench_load_bin(path, elements, num_elements * dim * sizeof(T));
    sprintf(path, "%s/ref_c_init.bin", ref_path);
    bench_load_bin(path, centroids, k * dim * sizeof(T));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    init_input(elements);
    /* Initialise centroids to first k data points */
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < dim; j++)
        centroids[i * dim + j] = elements[i * dim + j];
    }
    bench_stage_end(&stages);
  }

  // create_handle JIT-compiles the DPU kernel at runtime
  // (dpu-upmem-dpurte-clang). That one-time compile is SimplePIM's cold start
  // (PolymerPIM pays its JIT inside the first warmup iteration); time it into
  // warm_stages KERNEL and fold it into the first warmup sample so warmup_ms
  // and kernel_cold_ms agree.
  Timer ch_timer;
  bench_start(&ch_timer, 0);
  bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
  handle_t* va_handle = create_handle("kmeans_funcs", REDUCE);
  bench_stage_end(&warm_stages);
  bench_stop(&ch_timer, 0);
  double create_handle_us = ch_timer.time[0];

  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("t1", elements, num_elements, dim * sizeof(T),
                    table_management);
  bench_stage_end(&stages);
  uint32_t data_offset = lookup_table("t1", table_management)->end;

  // Warmup (timed separately): run throwaway k-means steps so the first
  // table_gen_red — which triggers the cold kernel load — is measured apart
  // from steady state. Save/restore centroids so warmup doesn't advance the
  // real run.
  T* centroids_saved = (T*)calloc(k * dim, sizeof(T));
  for (int i = 0; i < k * dim; i++) centroids_saved[i] = centroids[i];
  Timer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (uint32_t w = 0; w < warmup_iterations; w++) {
    bench_start(&warmup_timer, 0);
    bench_stage_begin(&warm_stages, BENCH_STAGE_WRITE);
    simplepim_broadcast("t2", centroids, k, dim * sizeof(T), table_management);
    bench_stage_end(&warm_stages);
    bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
    void* res = table_gen_red("t1", "t3", dim * sizeof(T) + sizeof(int32_t), k,
                              va_handle, table_management, data_offset);
    bench_stage_end(&warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0] +
                                          (w == 0 ? create_handle_us : 0.0));
    bench_stage_begin(&warm_stages, BENCH_STAGE_MERGE);
    average_table_entries_to_arr(res, centroids);
    bench_stage_end(&warm_stages);
    free_table("t2", table_management);
    free(res);
  }
  if (warmup_iterations > 0)
    bench_stats_print("simplepim_warmup", &warmup_stats);
  for (int i = 0; i < k * dim; i++) centroids[i] = centroids_saved[i];
  free(centroids_saved);

  BenchStats stats;
  bench_stats_init(&stats);
  Timer timer;
  for (int m = 0; m < iter; m++) {
    bench_start(&timer, 0);
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    simplepim_broadcast("t2", centroids, k, dim * sizeof(T), table_management);
    bench_stage_end(&stages);
    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    void* res = table_gen_red("t1", "t3", dim * sizeof(T) + sizeof(int32_t), k,
                              va_handle, table_management, data_offset);
    bench_stage_end(&stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
    bench_stage_begin(&stages, BENCH_STAGE_MERGE);
    average_table_entries_to_arr(res, centroids);
    bench_stage_end(&stages);
    free_table("t2", table_management);
    free(res);
  }
  bench_stats_print("simplepim", &stats);
  bench_stages_report("simplepim", &stages);
  bench_stages_report("simplepim_cold", &warm_stages);

  if (print_info && load_ref) {
    printf("DPU Final centroids sampled:\n");
    for (int i = 0; i < (k < 5 ? k : 5); i++) {
      for (int j = 0; j < (dim < 5 ? dim : 5); j++)
        printf("%d ", centroids[i * dim + j]);
      printf("...\n");
    }
  }

  /* check accuracy */
  if (load_ref && check_correctness) {
    T* cpu_centroids = (T*)calloc(k * dim, sizeof(T));
    char path[1024];
    sprintf(path, "%s/ref_c_final.bin", ref_path);
    bench_load_bin(path, cpu_centroids, k * dim * sizeof(T));

    int correct = 1;
    for (uint64_t i = 0; i < (uint64_t)(k * dim); i++) {
      if (centroids[i] != cpu_centroids[i]) {
        printf("MISMATCH at index %lu: dpu=%d cpu=%d\n", i, centroids[i],
               cpu_centroids[i]);
        correct = 0;
        break;
      }
    }
    if (correct) printf("the result is correct\n");
    free(cpu_centroids);
  }
}

int main(int argc, char** argv) {
  run();
  return 0;
}
