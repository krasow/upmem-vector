#include <benchmark.h>
#include <stdio.h>
#include <stdlib.h>

#include <climits>
#include <dpu>

#include "../Param.h"

#define CHECK(x) DPU_ASSERT(x)
#define NR_TASKLETS 12

// Row size padded to 8-byte multiple
static const uint32_t ROW_BYTES = ((DIM * sizeof(T)) + 7) & ~7u;

typedef struct {
  uint32_t data_offset;
  uint32_t query_offset;
  uint32_t result_offset;
  uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

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
  uint32_t data_bytes = elems_per_dpu * ROW_BYTES;
  uint32_t query_offset = (data_bytes + 7) & ~7u;
  uint32_t query_bytes = ((DIM * sizeof(T)) + 7) & ~7u;
  uint32_t result_offset = (query_offset + query_bytes + 7) & ~7u;

  DPU_LAUNCH_ARGS args[nr_dpus];
  for (uint32_t i = 0; i < nr_dpus; i++) {
    args[i].data_offset = 0;
    args[i].query_offset = query_offset;
    args[i].result_offset = result_offset;
    args[i].num_elements = elems_per_dpu;
  }

  T *row_data = NULL;
  T *query = NULL;

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  row_data = (T *)calloc((uint64_t)elems_per_dpu * nr_dpus * DIM, sizeof(T));
  query = (T *)calloc(DIM, sizeof(T));
  bench_stage_end(&stages);

  if (load_ref) {
    char path[512];
    snprintf(path, sizeof(path), "%s/AoS/rows.bin", ref_path);
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    bench_load_bin(path, row_data, N * DIM * sizeof(T));
    bench_stage_end(&stages);
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    snprintf(path, sizeof(path), "%s/ref_query.bin", ref_path);
    bench_load_bin(path, query, DIM * sizeof(T));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    srand(seed);
    for (uint64_t i = 0; i < N; i++)
      for (uint32_t d = 0; d < DIM; d++)
        row_data[i * DIM + d] = (T)((i * (DIM + 1) + d) % 256);
    for (uint32_t d = 0; d < DIM; d++) query[d] = (T)(d * 17 % 128);
    bench_stage_end(&stages);
  }

  // Upload row-major data (each DPU gets its contiguous slice, already
  // row-major)
  uint32_t idx;
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, idx)
  CHECK(dpu_prepare_xfer(dpu, &row_data[(uint64_t)idx * elems_per_dpu * DIM]));
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                      data_bytes, DPU_XFER_DEFAULT));

  // Broadcast query
  CHECK(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, query_offset,
                         query, query_bytes, DPU_XFER_DEFAULT));

  // Broadcast args
  DPU_FOREACH(dpu_set, dpu, idx)
  CHECK(dpu_prepare_xfer(dpu, &args[idx]));
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0, sizeof(args[0]),
                      DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  Timer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (uint32_t w = 0; w < warmup_iterations; w++) {
    bench_start(&warmup_timer, 0);
    bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
    CHECK(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0]);
  }
  if (warmup_iterations > 0)
    bench_stats_print("baseline_warmup", &warmup_stats);

  /* Each tasklet writes 8 bytes (RED_T + 4-byte pad for MRAM alignment) */
  RED_T *tasklet_mins =
      (RED_T *)calloc(nr_dpus * NR_TASKLETS * 2, sizeof(RED_T));
  BenchStats stats;
  bench_stats_init(&stats);
  Timer timer;
  RED_T result = INT32_MAX;

  for (uint32_t it = 0; it < iterations; it++) {
    bench_start(&timer, 0);
    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    CHECK(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&stages);
    bench_stage_begin(&stages, BENCH_STAGE_READ);
    DPU_FOREACH(dpu_set, dpu, idx)
    CHECK(dpu_prepare_xfer(dpu, &tasklet_mins[idx * NR_TASKLETS * 2]));
    CHECK(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                        result_offset, NR_TASKLETS * 8, DPU_XFER_DEFAULT));
    bench_stage_end(&stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);

    bench_stage_begin(&stages, BENCH_STAGE_MERGE);
    result = INT32_MAX;
    for (uint32_t i = 0; i < nr_dpus * NR_TASKLETS; i++)
      if (tasklet_mins[i * 2] < result) result = tasklet_mins[i * 2];
    bench_stage_end(&stages);
  }

  bench_stats_print("baseline", &stats);
  bench_stages_report("baseline", &stages);
  bench_stages_report("baseline_cold", &warm_stages);

  if (check_correctness && load_ref) {
    RED_T expected;
    char path[512];
    snprintf(path, sizeof(path), "%s/ref_res.bin", ref_path);
    bench_load_bin(path, &expected, sizeof(RED_T));
    if (result == expected)
      printf("the result is correct\n");
    else
      printf("Mismatch: got %d, expected %d\n", (int)result, (int)expected);
  }

  free(row_data);
  free(query);
  free(tasklet_mins);
  CHECK(dpu_free(dpu_set));
  return 0;
}
