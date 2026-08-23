#include <benchmark.h>
#include <dpu.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

static vector_search_result_t cpu_best(const T *records, const T *query) {
  vector_search_result_t result;
  vector_search_result_init(&result);
#pragma omp parallel
  {
    vector_search_result_t local;
    vector_search_result_init(&local);
#pragma omp for nowait
    for (uint64_t i = 0; i < nr_elements; ++i) {
      const T *row = records + i * (DIM + 1);
      int32_t score = 0;
      for (uint32_t d = 0; d < DIM; ++d) score += row[d] + query[d];
      vector_search_result_insert(
          &local,
          (int32_t)(((int64_t)score + 2 * DIM) * nr_elements + row[DIM]));
    }
#pragma omp critical
    vector_search_result_merge(&result, &local);
  }
  return result;
}

int main(void) {
  if (nr_elements % dpu_number != 0 ||
      !vector_search_key_range_is_valid(nr_elements, DIM)) {
    fprintf(stderr,
            "Invalid Vector search configuration: require N divisible by DPUs "
            "and (4*DIM+1)*N <= INT32_MAX\n");
    return 2;
  }

  BenchStages stages, warm_stages;
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t *mgmt = table_management_init(dpu_number);
  bench_stage_end(&stages);

  T *records;
  T *query;
  const uint32_t query_bytes = ((DIM * sizeof(T)) + 7) & ~7u;
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  records =
      (T *)malloc_scatter_aligned(nr_elements, (DIM + 1) * sizeof(T), mgmt);
  /* SimplePIM's initial broadcast helper rounds an already-aligned size up
   * by another eight bytes, so retain one guard word beyond the DMA size. */
  query = (T *)calloc(1, query_bytes + 8);
  bench_stage_end(&stages);

  bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
  for (uint64_t i = 0; i < nr_elements; ++i) {
    for (uint32_t d = 0; d < DIM; ++d)
      records[i * (DIM + 1) + d] = vector_search_dataset_value(seed, i, d, DIM);
    records[i * (DIM + 1) + DIM] = (T)(nr_elements - 1 - i);
  }
  bench_stage_end(&stages);

  /* Scatter the disjoint partitions once, then reserve a broadcast slot. */
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("dataset", records, nr_elements, (DIM + 1) * sizeof(T),
                    mgmt);
  simplepim_broadcast("query", query, 1, DIM * sizeof(T), mgmt);
  const uint32_t query_offset = lookup_table("query", mgmt)->start;
  bench_stage_end(&stages);

  BenchTimer create_timer;
  bench_start(&create_timer, 0);
  bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
  handle_t *handle = create_handle("vector_search_funcs", REDUCE);
  bench_stage_end(&warm_stages);
  bench_stop(&create_timer, 0);
  const double create_handle_us = create_timer.time[0];

  vector_search_result_t last_result;
  vector_search_result_init(&last_result);
  uint64_t query_id = 0;
  double pending_cold_us = create_handle_us;

/* SimplePIM's reduction primitive performs local and host maxima. */
#define RUN_QUERY(STAGE_SET, RESULT_LVALUE)                               \
  do {                                                                    \
    for (uint32_t d = 0; d < DIM; ++d)                                    \
      query[d] = vector_search_query_value(seed, query_id, d);            \
    ++query_id;                                                           \
    bench_stage_begin(&(STAGE_SET), BENCH_STAGE_WRITE);                   \
    DPU_ASSERT(dpu_broadcast_to(mgmt->set, DPU_MRAM_HEAP_POINTER_NAME,    \
                                query_offset, query, query_bytes,         \
                                DPU_XFER_DEFAULT));                       \
    bench_stage_end(&(STAGE_SET));                                        \
    bench_stage_begin(&(STAGE_SET), BENCH_STAGE_KERNEL);                  \
    vector_search_result_t *query_result =                                \
        (vector_search_result_t *)table_gen_red(                          \
            "dataset", "best", sizeof(vector_search_result_t), 1, handle, \
            mgmt, query_offset);                                          \
    bench_stage_end(&(STAGE_SET));                                        \
    (RESULT_LVALUE) = *query_result;                                      \
    free(query_result);                                                   \
  } while (0)

  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  BenchTimer timer;
  for (uint32_t w = 0; w < warmup_iterations; ++w) {
    bench_start(&timer, 0);
    RUN_QUERY(warm_stages, last_result);
    bench_stop(&timer, 0);
    bench_stats_update(&warmup_stats,
                       timer.time[0] + (w == 0 ? pending_cold_us : 0.0));
  }
  if (warmup_iterations) bench_stats_print("simplepim_warmup", &warmup_stats);

  BenchStats stats;
  bench_stats_init(&stats);
  for (uint32_t it = 0; it < iterations; ++it) {
    bench_start(&timer, 0);
    RUN_QUERY(stages, last_result);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }
#undef RUN_QUERY

  bench_stats_print("simplepim", &stats);
  bench_stages_report("simplepim", &stages);
  bench_stages_report("simplepim_cold", &warm_stages);

  if (check_correctness && iterations) {
    vector_search_result_t expected = cpu_best(records, query);
    int ok = last_result.key == expected.key;
    if (ok)
      printf("the result is correct\n");
    else
      printf("Mismatch: got key %d, expected %d\n", last_result.key,
             expected.key);
  }

  free(records);
  free(query);
  return 0;
}
