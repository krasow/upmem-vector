#include <benchmark.h>
#include <dpu.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

void run() {
  BenchStages stages;       // steady-loop stages (+ one-time setup)
  BenchStages warm_stages;  // cold warmup-loop stages (the cold-start premium)
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t* mgmt = table_management_init(dpu_number);
  bench_stage_end(&stages);

  /* Allocate row-major data (N rows × DIM columns) */
  T* data = NULL;
  T* query = NULL;

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  data = (T*)malloc_scatter_aligned(nr_elements, DIM * sizeof(T), mgmt);
  query = (T*)malloc_broadcast_aligned(1, DIM * sizeof(T), mgmt);
  bench_stage_end(&stages);

  if (load_ref) {
    char path[512];
    snprintf(path, sizeof(path), "%s/AoS/rows.bin", ref_path);
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    bench_load_bin(path, data, nr_elements * DIM * sizeof(T));
    bench_stage_end(&stages);
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    snprintf(path, sizeof(path), "%s/ref_query.bin", ref_path);
    bench_load_bin(path, query, DIM * sizeof(T));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    for (uint64_t i = 0; i < nr_elements; i++) {
      for (uint32_t d = 0; d < DIM; d++) {
        data[i * DIM + d] = (T)((i * (DIM + 1) + d) % 256);
      }
    }
    for (uint32_t d = 0; d < DIM; d++) {
      query[d] = (T)(d * 17 % 128);
    }
    bench_stage_end(&stages);
  }

  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("t1", data, nr_elements, DIM * sizeof(T), mgmt);
  uint32_t data_offset = lookup_table("t1", mgmt)->end;
  /* Broadcast query once — sits at data_offset in DPU MRAM */
  simplepim_broadcast("query", query, 1, DIM * sizeof(T), mgmt);
  bench_stage_end(&stages);

  // create_handle JIT-compiles the DPU kernel at runtime
  // (dpu-upmem-dpurte-clang). That one-time compile is SimplePIM's cold start
  // (PolymerPIM pays its JIT inside the first warmup iteration); time it into
  // warm_stages KERNEL and fold it into the first warmup sample so warmup_ms
  // and kernel_cold_ms agree.
  BenchTimer ch_timer;
  bench_start(&ch_timer, 0);
  bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
  handle_t* va_handle = create_handle("knn_funcs", REDUCE);
  bench_stage_end(&warm_stages);
  bench_stop(&ch_timer, 0);
  double create_handle_us = ch_timer.time[0];

  /* Warmup */
  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (uint32_t w = 0; w < warmup_iterations; w++) {
    bench_start(&warmup_timer, 0);
    bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
    RED_T* res = (RED_T*)table_gen_red("t1", "res", sizeof(RED_T), 1, va_handle,
                                       mgmt, data_offset);
    bench_stage_end(&warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0] +
                                          (w == 0 ? create_handle_us : 0.0));
    free(res);
  }
  if (warmup_iterations > 0) {
    bench_stats_print("simplepim_warmup", &warmup_stats);
  }

  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;
  RED_T result = INT32_MAX;

  for (uint32_t it = 0; it < iterations; it++) {
    bench_start(&timer, 0);
    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    RED_T* res = (RED_T*)table_gen_red("t1", "res", sizeof(RED_T), 1, va_handle,
                                       mgmt, data_offset);
    bench_stage_end(&stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
    result = *res;
    free(res);
  }

  bench_stats_print("simplepim", &stats);
  bench_stages_report("simplepim", &stages);
  bench_stages_report("simplepim_cold", &warm_stages);

  if (check_correctness && load_ref) {
    char path[512];
    RED_T expected;
    snprintf(path, sizeof(path), "%s/ref_res.bin", ref_path);
    bench_load_bin(path, &expected, sizeof(RED_T));
    if (result == expected) {
      printf("the result is correct\n");
    } else {
      printf("Mismatch: got %d, expected %d\n", (int)result, (int)expected);
    }
  }

  free_table("query", mgmt);
}

int main(void) {
  run();
  return 0;
}
