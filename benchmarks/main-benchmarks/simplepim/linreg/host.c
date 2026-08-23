#include <assert.h>
#include <benchmark.h>
#include <dpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

void init_data(T* elements, uint32_t num_elements, uint32_t d) {
  for (size_t i = 0; i < num_elements * d; i++) {
    elements[i] = (T)(i % 256);
  }
}

int main() {
  if (!load_ref) {
    srand(1);
  }
  BenchStages stages;       // steady-loop stages (+ one-time setup)
  BenchStages warm_stages;  // cold warmup-loop stages (the cold-start premium)
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t* table_management = table_management_init(dpu_number);
  bench_stage_end(&stages);
  if (print_info) {
    printf("dim: %d, num_elem: %ld, iter: %d, lr: %f, RED_T size: %zu\n", dim,
           nr_elements, iterations, lr, sizeof(RED_T));
  }

  // inputs
  T* elements = NULL;
  T* weights = NULL;
  uint32_t row_size =
      load_ref ? (((dim + 1) * sizeof(T) + 7) & ~7u) : ((dim + 1) * sizeof(T));

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  elements =
      (T*)malloc_scatter_aligned(nr_elements, row_size, table_management);
  weights = malloc_broadcast_aligned(1, sizeof(T) * dim, table_management);
  for (int i = 0; i < dim; i++) {
    weights[i] = 0;
  }
  bench_stage_end(&stages);

  if (load_ref) {
    printf("Loading reference data from %s...\n", ref_path);
    char path[1024];
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    sprintf(path, "%s/AoS/rows.bin", ref_path);
    bench_load_bin(path, elements, nr_elements * row_size);
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    init_data(elements, nr_elements, dim + 1);
    bench_stage_end(&stages);
  }

  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("t1", elements, nr_elements, row_size, table_management);
  uint32_t data_offset = lookup_table("t1", table_management)->end;
  simplepim_broadcast("t2", weights, 1, dim * sizeof(T), table_management);
  bench_stage_end(&stages);

  // create_handle JIT-compiles the DPU kernel at runtime
  // (dpu-upmem-dpurte-clang). That one-time compile is SimplePIM's cold start
  // (PolymerPIM pays its JIT inside the first warmup iteration); time it into
  // warm_stages KERNEL and fold it into the first warmup sample so warmup_ms
  // and kernel_cold_ms agree.
  BenchTimer ch_timer;
  bench_start(&ch_timer, 0);
  bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
  handle_t* va_handle = create_handle("lin_reg_funcs", REDUCE);
  bench_stage_end(&warm_stages);
  bench_stop(&ch_timer, 0);
  double create_handle_us = ch_timer.time[0];

  // Warmup (timed separately: first map-reduce includes any cold/lazy init)
  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (int l = 0; l < warmup_iterations; l++) {
    bench_start(&warmup_timer, 0);
    bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
    RED_T* res = table_gen_red("t1", "t3", dim * sizeof(RED_T), 1, va_handle,
                               table_management, data_offset);
    bench_stage_end(&warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0] +
                                          (l == 0 ? create_handle_us : 0.0));
    free(res);
    bench_stage_begin(&warm_stages, BENCH_STAGE_WRITE);
    simplepim_broadcast("t2", weights, 1, dim * sizeof(T), table_management);
    bench_stage_end(&warm_stages);
  }
  if (warmup_iterations > 0) {
    bench_stats_print("simplepim_warmup", &warmup_stats);
  }

  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;
  RED_T* final_res = NULL;
  for (int l = 0; l < iterations; l++) {
    if (final_res) {
      free(final_res);
    }
    bench_start(&timer, 0);
    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    final_res = table_gen_red("t1", "t3", dim * sizeof(RED_T), 1, va_handle,
                              table_management, data_offset);
    bench_stage_end(&stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }

  if (final_res) {
    printf("Final gradients: ");
    for (int i = 0; i < dim; i++) {
      printf("%lld ", (long long)final_res[i]);
    }
    printf("\n");
  }

  bench_stats_print("simplepim", &stats);
  bench_stages_report("simplepim", &stages);
  bench_stages_report("simplepim_cold", &warm_stages);

  if (final_res && check_correctness && load_ref) {
    RED_T* expected_grads = (RED_T*)calloc(dim, sizeof(RED_T));
    char path[1024];
    sprintf(path, "%s/ref_grads.bin", ref_path);
    bench_load_bin(path, expected_grads, dim * sizeof(RED_T));

    int match = 1;
    for (uint64_t i = 0; i < (uint64_t)dim; i++) {
      if (final_res[i] != expected_grads[i]) {
        printf("Mismatch at gradient %lu: got %lld, expected %lld\n", i,
               (long long)final_res[i], (long long)expected_grads[i]);
        match = 0;
      }
    }
    if (match) {
      printf("All results match after %d iterations.\n", iterations);
    }
    free(expected_grads);
  }
  if (final_res) {
    free(final_res);
  }

  return 0;
}
