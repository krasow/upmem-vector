#include <benchmark.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <climits>
#include <dpu>

#include "../Param.h"

#define CHECK(x) DPU_ASSERT(x)
#define NR_TASKLETS 12

static const uint32_t ROW_BYTES = ((DIM * sizeof(T)) + 7) & ~7u;

typedef struct {
  uint32_t data_offset;
  uint32_t centroids_offset;
  uint32_t result_offset;
  uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

static void div_round_closest_centroids(RED_T *sums, RED_T *counts,
                                        T *centroids_out) {
  for (uint32_t j = 0; j < K; j++) {
    if (counts[j] <= 0) continue;
    for (uint32_t d = 0; d < DIM; d++) {
      int s = (int)sums[j * DIM + d];
      int c = (int)counts[j];
      centroids_out[j * DIM + d] =
          (T)(((s < 0) ^ (c < 0)) ? (s - c / 2) / c : (s + c / 2) / c);
    }
  }
}

int main() {
  struct dpu_set_t dpu_set, dpu;
  uint32_t nr_dpus = dpu_number;

  BenchStages stages;       // steady-loop stages (+ one-time setup)
  BenchStages warm_stages;  // cold warmup-loop stages (the cold-start premium)
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  CHECK(dpu_alloc(nr_dpus, NULL, &dpu_set));
  CHECK(dpu_load(dpu_set, "./bin/baseline.dpu", NULL));
  bench_stage_end(&stages);

  uint64_t elems_per_dpu = N / nr_dpus;
  uint32_t data_bytes = (uint32_t)(elems_per_dpu * ROW_BYTES);
  uint32_t centroids_offset = (data_bytes + 7) & ~7u;
  uint32_t centroids_bytes = (K * DIM * sizeof(T) + 7) & ~7u;
  uint32_t result_offset = (centroids_offset + centroids_bytes + 7) & ~7u;
  uint32_t partial_per_tasklet = K * (DIM + 1) * sizeof(RED_T); /* 440 bytes */

  DPU_LAUNCH_ARGS args[nr_dpus];
  for (uint32_t i = 0; i < nr_dpus; i++) {
    args[i].data_offset = 0;
    args[i].centroids_offset = centroids_offset;
    args[i].result_offset = result_offset;
    args[i].num_elements = (uint32_t)elems_per_dpu;
  }

  /* Upload args once */
  uint32_t idx;
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, idx)
  CHECK(dpu_prepare_xfer(dpu, &args[idx]));
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0, sizeof(args[0]),
                      DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  /* Build row-major data */
  T *row_data = NULL;
  T *centroids = NULL;

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  row_data = (T *)calloc((uint64_t)elems_per_dpu * nr_dpus * DIM, sizeof(T));
  centroids = (T *)calloc(K * DIM, sizeof(T));
  bench_stage_end(&stages);
  if (load_ref) {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    char path[512];
    snprintf(path, sizeof(path), "%s/AoS/rows.bin", ref_path);
    bench_load_bin(path, row_data, (uint64_t)N * DIM * sizeof(T));
    snprintf(path, sizeof(path), "%s/ref_c_init.bin", ref_path);
    bench_load_bin(path, centroids, K * DIM * sizeof(T));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    srand(seed);
    for (uint64_t i = 0; i < N; i++)
      for (uint32_t d = 0; d < DIM; d++)
        row_data[i * DIM + d] = (T)((i + d) % 1000);
    for (uint32_t j = 0; j < K; j++)
      for (uint32_t d = 0; d < DIM; d++)
        centroids[j * DIM + d] = row_data[j * DIM + d];
    bench_stage_end(&stages);
  }

  T *centroids_init = (T *)calloc(K * DIM, sizeof(T));
  for (uint32_t i = 0; i < K * DIM; i++) centroids_init[i] = centroids[i];

  /* Upload row-major data once */
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, idx)
  CHECK(dpu_prepare_xfer(dpu, &row_data[(uint64_t)idx * elems_per_dpu * DIM]));
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                      data_bytes, DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  /* Partial result gather buffer */
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  RED_T *partials = (RED_T *)calloc(
      (uint64_t)nr_dpus * NR_TASKLETS * K * (DIM + 1), sizeof(RED_T));
  bench_stage_end(&stages);

  auto run_kmeans_iter = [&](BenchStages &stages) {
    /* Broadcast current centroids */
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    CHECK(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME,
                           centroids_offset, centroids, centroids_bytes,
                           DPU_XFER_DEFAULT));
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    CHECK(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&stages);

    /* Gather partial results */
    bench_stage_begin(&stages, BENCH_STAGE_READ);
    DPU_FOREACH(dpu_set, dpu, idx)
    CHECK(dpu_prepare_xfer(
        dpu, &partials[(uint64_t)idx * NR_TASKLETS * K * (DIM + 1)]));
    CHECK(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                        result_offset, NR_TASKLETS * partial_per_tasklet,
                        DPU_XFER_DEFAULT));
    bench_stage_end(&stages);

    /* CPU reduce: sum all partials into new centroids */
    bench_stage_begin(&stages, BENCH_STAGE_MERGE);
    RED_T global_sum[K * DIM];
    RED_T global_count[K];
    for (uint32_t j = 0; j < K; j++) {
      for (uint32_t d = 0; d < DIM; d++) global_sum[j * DIM + d] = 0;
      global_count[j] = 0;
    }
    for (uint32_t i = 0; i < nr_dpus * NR_TASKLETS; i++) {
      RED_T *p = &partials[(uint64_t)i * K * (DIM + 1)];
      for (uint32_t j = 0; j < K; j++) {
        for (uint32_t d = 0; d < DIM; d++)
          global_sum[j * DIM + d] += p[j * (DIM + 1) + d];
        global_count[j] += p[j * (DIM + 1) + DIM];
      }
    }
    div_round_closest_centroids(global_sum, global_count, centroids);
    bench_stage_end(&stages);
  };

  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (uint32_t w = 0; w < warmup_iterations; w++) {
    for (uint32_t i = 0; i < K * DIM; i++) centroids[i] = centroids_init[i];
    bench_start(&warmup_timer, 0);
    run_kmeans_iter(warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0]);
  }
  if (warmup_iterations > 0)
    bench_stats_print("baseline_warmup", &warmup_stats);

  for (uint32_t i = 0; i < K * DIM; i++) centroids[i] = centroids_init[i];
  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;
  for (uint32_t it = 0; it < iterations; it++) {
    bench_start(&timer, 0);
    run_kmeans_iter(stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }
  bench_stats_print("baseline", &stats);
  bench_stages_report("baseline", &stages);
  bench_stages_report("baseline_cold", &warm_stages);

  free(row_data);
  free(centroids);
  free(centroids_init);
  free(partials);
  CHECK(dpu_free(dpu_set));
  return 0;
}
