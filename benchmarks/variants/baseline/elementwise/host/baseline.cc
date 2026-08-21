#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include <benchmark.h>

#include "../Param.h"

#ifndef FRESH_RESULT_BUFFER
#define FRESH_RESULT_BUFFER 0
#endif

typedef struct {
  uint32_t lhs_offset;
  uint32_t rhs_offset;
  uint32_t res_offset;
  uint32_t num_elements;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

void vec_xfer_from_dpu(dpu_set_t dpu_set, char *cpu, DPU_LAUNCH_ARGS *args) {
  dpu_set_t dpu;

  uint32_t idx_dpu = 0;
  size_t element = 0;

  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &(cpu[element])));
    element += args[idx_dpu].num_elements * sizeof(int32_t);
  }

  uint32_t mram_location = args[0].res_offset;
  size_t xfer_size = args[0].num_elements * sizeof(int32_t);
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME, mram_location,
                            xfer_size, DPU_XFER_DEFAULT));
}

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

  DPU_LAUNCH_ARGS args[nr_of_dpus];

  int elements_per_dpu = nr_elements / nr_of_dpus;
  uint32_t slice_bytes = elements_per_dpu * sizeof(int32_t);
  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].num_elements = elements_per_dpu;
    args[i].lhs_offset = 0;
    args[i].rhs_offset = slice_bytes;
    args[i].res_offset = slice_bytes * 2;
  }

  int32_t *a_vec = NULL;
  int32_t *b_vec = NULL;
  int32_t *res_vec = NULL;

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  a_vec = (int32_t *)calloc(nr_elements, sizeof(int32_t));
  b_vec = (int32_t *)calloc(nr_elements, sizeof(int32_t));
#if !FRESH_RESULT_BUFFER
  res_vec = (int32_t *)calloc(nr_elements, sizeof(int32_t));
#endif
  bench_stage_end(&stages);
  if (load_ref) {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    char path[1024];
    printf("Loading reference data from %s...\n", ref_path);
    sprintf(path, "%s/ref_a.bin", ref_path);
    bench_load_bin(path, a_vec, nr_elements * sizeof(int32_t));
    sprintf(path, "%s/ref_b.bin", ref_path);
    bench_load_bin(path, b_vec, nr_elements * sizeof(int32_t));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    for (uint64_t i = 0; i < nr_elements; i++) {
      a_vec[i] = rand() % 10;
      b_vec[i] = rand() % 10;
    }
    bench_stage_end(&stages);
  }

  dpu_set_t dpu;
  uint32_t idx_dpu = 0;
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
  }
  CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
                            sizeof(args[0]), DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  auto run_round_trip = [&](BenchStages &stages) {
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    uint32_t idx_dpu = 0;
    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
      CHECK_UPMEM(dpu_prepare_xfer(dpu, &a_vec[idx_dpu * elements_per_dpu]));
    }
    CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                              DPU_MRAM_HEAP_POINTER_NAME, 0, slice_bytes,
                              DPU_XFER_DEFAULT));

    idx_dpu = 0;
    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
      CHECK_UPMEM(dpu_prepare_xfer(dpu, &b_vec[idx_dpu * elements_per_dpu]));
    }
    CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                              DPU_MRAM_HEAP_POINTER_NAME, slice_bytes,
                              slice_bytes, DPU_XFER_DEFAULT));
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    CHECK_UPMEM(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_READ);
#if FRESH_RESULT_BUFFER
    int32_t *round_res_vec = (int32_t *)calloc(nr_elements, sizeof(int32_t));
    if (!round_res_vec) {
      fprintf(stderr, "failed to allocate fresh result buffer\n");
      exit(1);
    }
    vec_xfer_from_dpu(dpu_set, (char *)round_res_vec, args);
    res_vec = round_res_vec;
#else
    vec_xfer_from_dpu(dpu_set, (char *)res_vec, args);
#endif
    bench_stage_end(&stages);
  };

  Timer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (int i = 0; i < warmup_iterations; i++) {
#if FRESH_RESULT_BUFFER
    if (res_vec) {
      free(res_vec);
      res_vec = NULL;
    }
#endif
    bench_start(&warmup_timer, 0);
    run_round_trip(warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0]);
  }
  if (warmup_iterations > 0)
    bench_stats_print("baseline_warmup", &warmup_stats);

  BenchStats stats;
  bench_stats_init(&stats);
  Timer timer;

  for (int i = 0; i < iterations; i++) {
#if FRESH_RESULT_BUFFER
    if (res_vec) {
      free(res_vec);
      res_vec = NULL;
    }
#endif
    bench_start(&timer, 0);
    run_round_trip(stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }

  bench_stats_print("baseline", &stats);
  bench_stages_report("baseline", &stages);
  bench_stages_report("baseline_cold", &warm_stages);

  if (check_correctness) {
    int32_t *correct_res = (int32_t *)calloc(nr_elements, sizeof(int32_t));
    if (load_ref) {
      char path[1024];
      sprintf(path, "%s/ref_res.bin", ref_path);
      bench_load_bin(path, correct_res, nr_elements * sizeof(int32_t));
    } else {
      for (uint64_t i = 0; i < nr_elements; i++) {
        correct_res[i] = OPERATION(a_vec[i], b_vec[i]);
      }
    }

    int is_correct = 1;
    for (uint64_t i = 0; i < nr_elements; i++) {
      if (res_vec[i] != correct_res[i]) {
        is_correct = 0;
        printf("result mismatch at position %lu, got %d, expected %d \n", i,
               res_vec[i], correct_res[i]);
        break;
      }
    }
    if (is_correct) {
      printf("All results match after %d iterations.\n", iterations);
    }
    free(correct_res);
  }

  free(a_vec);
  free(b_vec);
  free(res_vec);

  CHECK_UPMEM(dpu_free(dpu_set));

  return 0;
}
