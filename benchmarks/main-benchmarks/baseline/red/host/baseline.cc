#include <benchmark.h>
#include <stdio.h>
#include <stdlib.h>

#include <dpu>

#include "../Param.h"

#define CHECK(x) DPU_ASSERT(x)
#define NR_TASKLETS 12

typedef struct {
  uint32_t data_offset;
  uint32_t result_offset;
  uint32_t num_elements;
  uint32_t _pad;
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
  uint32_t data_bytes = elems_per_dpu * sizeof(T);
  uint32_t result_offset = (data_bytes + 7) & ~7u;

  DPU_LAUNCH_ARGS args[nr_dpus];
  for (uint32_t i = 0; i < nr_dpus; i++) {
    args[i].data_offset = 0;
    args[i].result_offset = result_offset;
    args[i].num_elements = elems_per_dpu;
    args[i]._pad = 0;
  }

  T* input = NULL;
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  input = (T*)calloc(N, sizeof(T));
  bench_stage_end(&stages);
  if (load_ref) {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    char path[512];
    snprintf(path, sizeof(path), "%s/ref_t1.bin", ref_path);
    bench_load_bin(path, input, N * sizeof(T));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    srand(seed);
    for (uint64_t i = 0; i < N; i++) {
      input[i] = rand() % 10;
    }
    bench_stage_end(&stages);
  }

  // Broadcast args (constant across iterations)
  uint32_t idx;
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, idx)
  CHECK(dpu_prepare_xfer(dpu, &args[idx]));
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0, sizeof(args[0]),
                      DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  RED_T partials[nr_dpus * NR_TASKLETS];
  RED_T result = 0;

  auto run_round_trip = [&](BenchStages& stages) {
    uint32_t idx;
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    DPU_FOREACH(dpu_set, dpu, idx)
    CHECK(dpu_prepare_xfer(dpu, &input[idx * elems_per_dpu]));
    CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                        data_bytes, DPU_XFER_DEFAULT));
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    CHECK(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_READ);
    DPU_FOREACH(dpu_set, dpu, idx)
    CHECK(dpu_prepare_xfer(dpu, &partials[idx * NR_TASKLETS]));
    CHECK(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                        result_offset, NR_TASKLETS * sizeof(RED_T),
                        DPU_XFER_DEFAULT));
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_MERGE);
    result = 0;
    for (uint32_t i = 0; i < nr_dpus * NR_TASKLETS; i++) {
      result += partials[i];
    }
    bench_stage_end(&stages);
  };

  // Warmup
  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (uint32_t w = 0; w < warmup_iterations; w++) {
    bench_start(&warmup_timer, 0);
    run_round_trip(warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0]);
  }
  if (warmup_iterations > 0) {
    bench_stats_print("baseline_warmup", &warmup_stats);
  }

  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;

  for (uint32_t it = 0; it < iterations; it++) {
    bench_start(&timer, 0);
    run_round_trip(stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }

  bench_stats_print("baseline", &stats);
  bench_stages_report("baseline", &stages);
  bench_stages_report("baseline_cold", &warm_stages);

  if (check_correctness && load_ref) {
    RED_T expected;
    char path[512];
    snprintf(path, sizeof(path), "%s/ref_res.bin", ref_path);
    bench_load_bin(path, &expected, sizeof(RED_T));
    if (result == (RED_T)expected) {
      printf("the result is correct\n");
    } else {
      printf("Mismatch: got %d, expected %d\n", (int)result, (int)expected);
    }
  }

  free(input);
  CHECK(dpu_free(dpu_set));
  return 0;
}
