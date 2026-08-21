#include <benchmark.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dpu>
#include <vector>

#include "../Param.h"

#define CHECK(x) DPU_ASSERT(x)

typedef struct {
  uint32_t data_offset;
  uint32_t query_offset;
  uint32_t result_offset;
  uint32_t num_elements;
  uint32_t base_index;
  uint32_t reserved;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

static vector_search_result_t cpu_best(const T *data, const T *query) {
  vector_search_result_t result;
  vector_search_result_init(&result);
#pragma omp parallel
  {
    vector_search_result_t local;
    vector_search_result_init(&local);
#pragma omp for nowait
    for (uint64_t i = 0; i < N; ++i) {
      int32_t score = 0;
      for (uint32_t d = 0; d < DIM; ++d) score += data[i * DIM + d] + query[d];
      vector_search_result_insert(&local,
                                  vector_search_pack_key(score, i, N, DIM));
    }
#pragma omp critical
    vector_search_result_merge(&result, &local);
  }
  return result;
}

int main() {
  if (N % dpu_number != 0 || !vector_search_key_range_is_valid(N, DIM)) {
    std::fprintf(
        stderr,
        "Invalid Vector search configuration: require N divisible by DPUs "
        "and (4*DIM+1)*N <= INT32_MAX\n");
    return 2;
  }

  struct dpu_set_t dpu_set, dpu;
  const uint32_t nr_dpus = dpu_number;
  const uint32_t elems_per_dpu = N / nr_dpus;
  const uint32_t data_bytes = elems_per_dpu * DIM * sizeof(T);
  const uint32_t query_offset = (data_bytes + 7) & ~7u;
  const uint32_t query_bytes = ((DIM * sizeof(T)) + 7) & ~7u;
  const uint32_t result_offset = query_offset + query_bytes;

  BenchStages stages, warm_stages;
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  CHECK(dpu_alloc(nr_dpus, NULL, &dpu_set));
  CHECK(dpu_load(dpu_set, "./bin/baseline.dpu", NULL));
  bench_stage_end(&stages);

  std::vector<T> data;
  std::vector<T> query((query_bytes / sizeof(T)), 0);
  std::vector<vector_search_result_t> local_results(nr_dpus);
  std::vector<DPU_LAUNCH_ARGS> args(nr_dpus);

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  data.resize(N * DIM);
  bench_stage_end(&stages);

  bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
  for (uint64_t i = 0; i < N; ++i)
    for (uint32_t d = 0; d < DIM; ++d)
      data[i * DIM + d] = vector_search_dataset_value(seed, i, d, DIM);
  bench_stage_end(&stages);

  for (uint32_t i = 0; i < nr_dpus; ++i) {
    args[i] = {0, query_offset, result_offset, elems_per_dpu, i * elems_per_dpu,
               0};
  }

  /* The disjoint dataset partitions and launch metadata are uploaded once. */
  uint32_t dpu_index;
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  DPU_FOREACH(dpu_set, dpu, dpu_index)
  CHECK(
      dpu_prepare_xfer(dpu, &data[(uint64_t)dpu_index * elems_per_dpu * DIM]));
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                      data_bytes, DPU_XFER_DEFAULT));
  DPU_FOREACH(dpu_set, dpu, dpu_index)
  CHECK(dpu_prepare_xfer(dpu, &args[dpu_index]));
  CHECK(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0, sizeof(args[0]),
                      DPU_XFER_DEFAULT));
  bench_stage_end(&stages);

  auto run_query = [&](uint64_t query_id, BenchStages &query_stages) {
    for (uint32_t d = 0; d < DIM; ++d)
      query[d] = vector_search_query_value(seed, query_id, d);

    bench_stage_begin(&query_stages, BENCH_STAGE_WRITE);
    CHECK(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, query_offset,
                           query.data(), query_bytes, DPU_XFER_DEFAULT));
    bench_stage_end(&query_stages);

    bench_stage_begin(&query_stages, BENCH_STAGE_KERNEL);
    CHECK(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    bench_stage_end(&query_stages);

    bench_stage_begin(&query_stages, BENCH_STAGE_READ);
    DPU_FOREACH(dpu_set, dpu, dpu_index)
    CHECK(dpu_prepare_xfer(dpu, &local_results[dpu_index]));
    CHECK(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                        result_offset, sizeof(vector_search_result_t),
                        DPU_XFER_DEFAULT));
    bench_stage_end(&query_stages);

    bench_stage_begin(&query_stages, BENCH_STAGE_MERGE);
    vector_search_result_t global;
    vector_search_result_init(&global);
    for (const auto &local : local_results)
      vector_search_result_merge(&global, &local);
    bench_stage_end(&query_stages);
    return global;
  };

  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  Timer timer;
  for (uint32_t w = 0; w < warmup_iterations; ++w) {
    bench_start(&timer, 0);
    (void)run_query(w, warm_stages);
    bench_stop(&timer, 0);
    bench_stats_update(&warmup_stats, timer.time[0]);
  }
  if (warmup_iterations) bench_stats_print("baseline_warmup", &warmup_stats);

  BenchStats stats;
  bench_stats_init(&stats);
  vector_search_result_t result;
  vector_search_result_init(&result);
  for (uint32_t it = 0; it < iterations; ++it) {
    bench_start(&timer, 0);
    result = run_query((uint64_t)warmup_iterations + it, stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }

  bench_stats_print("baseline", &stats);
  bench_stages_report("baseline", &stages);
  bench_stages_report("baseline_cold", &warm_stages);

  if (check_correctness && iterations) {
    vector_search_result_t expected = cpu_best(data.data(), query.data());
    bool ok = result.key == expected.key;
    if (ok)
      std::printf("the result is correct\n");
    else
      std::printf("Mismatch: got key %d, expected %d\n", result.key,
                  expected.key);
  }

  CHECK(dpu_free(dpu_set));
  return 0;
}
