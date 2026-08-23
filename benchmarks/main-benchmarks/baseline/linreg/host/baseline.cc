#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef DPURT
#define DPURT
#include <dpu>
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include <benchmark.h>

#include "../Param.h"

typedef struct {
  uint32_t data_offset;
  uint32_t weights_offset;
  uint32_t results_offset;
  uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

int main() {
  int nr_of_dpus = dpu_number;
  dpu_set_t dpu_set;

  BenchStages stages;       // steady-loop stages (+ one-time setup)
  BenchStages warm_stages;  // cold warmup-loop stages (the cold-start premium)
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  CHECK_UPMEM(dpu_alloc(
      nr_of_dpus,
      getenv("UPMEM_PROFILE") ? getenv("UPMEM_PROFILE") : "backend=hw",
      &dpu_set));
  CHECK_UPMEM(dpu_load(dpu_set, "./bin/baseline.dpu", nullptr));
  bench_stage_end(&stages);

  uint64_t elements_per_dpu = nr_elements / nr_of_dpus;
  uint32_t padded_row_size = ((dim + 1) * sizeof(T) + 7) & ~7;
  uint32_t weights_size = dim * sizeof(T);
  uint32_t red_size = dim * sizeof(RED_T);
  const int NR_TASKLETS = 12;

  DPU_LAUNCH_ARGS args[nr_of_dpus];
  for (int i = 0; i < nr_of_dpus; i++) {
    args[i].num_elements = elements_per_dpu;
    args[i].data_offset = 0;
    args[i].weights_offset = (elements_per_dpu * padded_row_size + 7) & ~7;
    args[i].results_offset = (args[i].weights_offset + weights_size + 7) & ~7;
  }

  T *all_elements = NULL;
  T *weights = NULL;

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  all_elements =
      (T *)calloc(nr_elements * (padded_row_size / sizeof(T)), sizeof(T));
  weights = (T *)calloc(dim, sizeof(T));
  bench_stage_end(&stages);

  if (load_ref) {
    printf("Loading reference data from %s...\n", ref_path);
    char path[1024];
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    sprintf(path, "%s/AoS/rows.bin", ref_path);
    bench_load_bin(path, all_elements, nr_elements * padded_row_size);
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    for (uint64_t i = 0; i < nr_elements; i++) {
      for (int j = 0; j < dim + 1; j++) {
        all_elements[i * (padded_row_size / sizeof(T)) + j] = (i + j) % 256;
      }
    }
    bench_stage_end(&stages);
  }

  // Transfer inputs
  dpu_set_t dpu;
  uint32_t idx;
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, idx) {
    CHECK_UPMEM(dpu_prepare_xfer(
        dpu,
        &all_elements[idx * elements_per_dpu * (padded_row_size / sizeof(T))]));
  }
  CHECK_UPMEM(dpu_push_xfer(
      dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, args[0].data_offset,
      elements_per_dpu * padded_row_size, DPU_XFER_DEFAULT));

  // Transfer weights (broadcast)
  CHECK_UPMEM(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME,
                               args[0].weights_offset, weights, weights_size,
                               DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  // Warmup (timed separately: the first launch includes any cold kernel load)
  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (int i = 0; i < warmup_iterations; i++) {
    bench_stage_begin(&warm_stages, BENCH_STAGE_WRITE);
    DPU_FOREACH(dpu_set, dpu, idx) {
      CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx]));
    }
    CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                              sizeof(DPU_LAUNCH_ARGS), DPU_XFER_DEFAULT));
    bench_stage_end(&warm_stages);
    bench_start(&warmup_timer, 0);
    bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
    CHECK_UPMEM(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0]);
  }
  if (warmup_iterations > 0)
    bench_stats_print("baseline_warmup", &warmup_stats);

  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  RED_T *final_grads_accum = (RED_T *)calloc(dim, sizeof(RED_T));
  RED_T *dpu_tasklet_grads =
      (RED_T *)calloc(nr_of_dpus * NR_TASKLETS * dim, sizeof(RED_T));
  bench_stage_end(&stages);

  for (int it = 0; it < iterations; it++) {
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    DPU_FOREACH(dpu_set, dpu, idx) {
      CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx]));
    }
    CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                              sizeof(DPU_LAUNCH_ARGS), DPU_XFER_DEFAULT));
    bench_stage_end(&stages);

    bench_start(&timer, 0);
    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    CHECK_UPMEM(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_READ);
    DPU_FOREACH(dpu_set, dpu, idx) {
      CHECK_UPMEM(
          dpu_prepare_xfer(dpu, &dpu_tasklet_grads[idx * NR_TASKLETS * dim]));
    }
    CHECK_UPMEM(dpu_push_xfer(
        dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
        args[0].results_offset, NR_TASKLETS * red_size, DPU_XFER_DEFAULT));
    bench_stage_end(&stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);

    bench_stage_begin(&stages, BENCH_STAGE_MERGE);
    for (int j = 0; j < dim; j++) final_grads_accum[j] = 0;
    for (int d = 0; d < nr_of_dpus; d++) {
      for (int t = 0; t < NR_TASKLETS; t++) {
        for (int j = 0; j < dim; j++) {
          final_grads_accum[j] +=
              dpu_tasklet_grads[(d * NR_TASKLETS + t) * dim + j];
        }
      }
    }
    bench_stage_end(&stages);
  }

  bench_stats_print("baseline", &stats);
  bench_stages_report("baseline", &stages);
  bench_stages_report("baseline_cold", &warm_stages);

  if (final_grads_accum) {
    printf("Final gradients: ");
    for (int i = 0; i < dim; i++) {
      printf("%lld ", (long long)final_grads_accum[i]);
    }
    printf("\n");
  }

  if (check_correctness) {
    if (load_ref) {
      RED_T *expected_grads = (RED_T *)calloc(dim, sizeof(RED_T));
      char path[1024];
      sprintf(path, "%s/ref_grads.bin", ref_path);
      bench_load_bin(path, expected_grads, dim * sizeof(RED_T));
      int match = 1;
      for (int j = 0; j < dim; j++) {
        if (final_grads_accum[j] != expected_grads[j]) {
          printf("Mismatch at gradient %d: got %lld, expected %lld\n", j,
                 (long long)final_grads_accum[j], (long long)expected_grads[j]);
          match = 0;
        }
      }
      if (match) {
        printf("All results match after %d iterations.\n", iterations);
      }
      free(expected_grads);
    }
  }

  free(all_elements);
  free(weights);
  free(final_grads_accum);
  free(dpu_tasklet_grads);
  CHECK_UPMEM(dpu_free(dpu_set));

  return 0;
}
