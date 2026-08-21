#include <benchmark.h>
#include <dpu.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../multitask_classifier_common.h"
#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

static void fill_rows(T* rows) {
  for (uint64_t i = 0; i < nr_elements; i++) {
    for (uint32_t d = 0; d < FEATURES; d++)
      rows[i * FLOW_ROW_WORDS + d] = (T)flow_feature_value(i, d);
    rows[i * FLOW_ROW_WORDS + FEATURES] = (T)flow_class_for_row(i);
    rows[i * FLOW_ROW_WORDS + FEATURES + 1] = 0;
  }
}

static void run_epoch(BenchStages* stages, simplepim_management_t* management,
                      handle_t* handle, uint32_t state_offset, T* state,
                      T* weights, int64_t* gradients, svm_metrics_t* metrics) {
  memset(gradients, 0, CLASSES * FEATURES * sizeof(*gradients));
  metrics->margin_violations = 0;

  for (uint32_t c = 0; c < CLASSES; c++) {
    for (uint32_t statistic = 0; statistic <= FEATURES; statistic++) {
      memset(state, 0, SVM_STATE_WORDS * sizeof(T));
      state[0] = SVM_MODE_TRAIN;
      state[1] = (T)c;
      state[2] = (T)statistic;
      for (uint32_t d = 0; d < FEATURES; d++)
        state[3 + d] = weights[c * FEATURES + d];

      bench_stage_begin(stages, BENCH_STAGE_WRITE);
      simplepim_broadcast("state", state, 1, SVM_STATE_WORDS * sizeof(T),
                          management);
      bench_stage_end(stages);

      bench_stage_begin(stages, BENCH_STAGE_KERNEL);
      RED_T* result =
          (RED_T*)table_gen_red("flows", "statistic", sizeof(RED_T), 1, handle,
                                management, state_offset);
      bench_stage_end(stages);

      if (statistic < FEATURES)
        gradients[c * FEATURES + statistic] = (int64_t)result[0];
      else
        metrics->margin_violations += (uint64_t)result[0];
      free(result);
    }
  }

  bench_stage_begin(stages, BENCH_STAGE_MERGE);
  for (uint32_t i = 0; i < CLASSES * FEATURES; i++)
    weights[i] = (T)svm_update_weight(weights[i], gradients[i], nr_elements);
  bench_stage_end(stages);

  memset(state, 0, SVM_STATE_WORDS * sizeof(T));
  state[0] = SVM_MODE_EVALUATE;
  for (uint32_t i = 0; i < CLASSES * FEATURES; i++) state[3 + i] = weights[i];

  bench_stage_begin(stages, BENCH_STAGE_WRITE);
  simplepim_broadcast("state", state, 1, SVM_STATE_WORDS * sizeof(T),
                      management);
  bench_stage_end(stages);

  bench_stage_begin(stages, BENCH_STAGE_KERNEL);
  RED_T* correct = (RED_T*)table_gen_red("flows", "correct", sizeof(RED_T), 1,
                                         handle, management, state_offset);
  bench_stage_end(stages);
  metrics->correct_predictions = (uint64_t)correct[0];
  free(correct);
}

int main(void) {
  BenchStages stages;
  BenchStages warm_stages;
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t* management = table_management_init(dpu_number);
  bench_stage_end(&stages);

  T* rows;
  T* state;
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  rows = (T*)malloc_scatter_aligned(nr_elements, FLOW_ROW_WORDS * sizeof(T),
                                    management);
  state =
      (T*)malloc_broadcast_aligned(1, SVM_STATE_WORDS * sizeof(T), management);
  bench_stage_end(&stages);

  bench_stage_begin(&stages, BENCH_STAGE_LOAD);
  if (load_ref) {
    char path[512];
    snprintf(path, sizeof(path), "%s/AoS/rows.bin", ref_path);
    bench_load_bin(path, rows, nr_elements * FLOW_ROW_WORDS * sizeof(T));
  } else {
    fill_rows(rows);
  }
  bench_stage_end(&stages);

  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("flows", rows, nr_elements, FLOW_ROW_WORDS * sizeof(T),
                    management);
  memset(state, 0, SVM_STATE_WORDS * sizeof(T));
  simplepim_broadcast("state", state, 1, SVM_STATE_WORDS * sizeof(T),
                      management);
  uint32_t state_offset = lookup_table("state", management)->start;
  bench_stage_end(&stages);

  BenchTimer handle_timer;
  bench_start(&handle_timer, 0);
  bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
  handle_t* handle = create_handle("multitask_classifier_funcs", REDUCE);
  bench_stage_end(&warm_stages);
  bench_stop(&handle_timer, 0);
  double create_handle_us = handle_timer.time[0];

  T* weights = (T*)calloc(CLASSES * FEATURES, sizeof(T));
  int64_t* gradients = (int64_t*)calloc(CLASSES * FEATURES, sizeof(int64_t));
  svm_metrics_t metrics = {0, 0};

  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (uint32_t i = 0; i < warmup_iterations; i++) {
    memset(weights, 0, CLASSES * FEATURES * sizeof(T));
    bench_start(&warmup_timer, 0);
    run_epoch(&warm_stages, management, handle, state_offset, state, weights,
              gradients, &metrics);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats,
                       warmup_timer.time[0] + (i == 0 ? create_handle_us : 0));
  }
  if (warmup_iterations > 0)
    bench_stats_print("simplepim_warmup", &warmup_stats);

  memset(weights, 0, CLASSES * FEATURES * sizeof(T));
  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;
  for (uint32_t i = 0; i < iterations; i++) {
    bench_start(&timer, 0);
    run_epoch(&stages, management, handle, state_offset, state, weights,
              gradients, &metrics);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }

  bench_stats_print("simplepim", &stats);
  bench_stages_report("simplepim", &stages);
  bench_stages_report("simplepim_cold", &warm_stages);
  printf("simplepim_result margin_violations=%llu accuracy=%llu/%llu\n",
         (unsigned long long)metrics.margin_violations,
         (unsigned long long)metrics.correct_predictions,
         (unsigned long long)nr_elements);

  if (check_correctness && load_ref) {
    T expected[FLOW_CLASSES * FLOW_FEATURES];
    svm_metrics_t expected_metrics;
    char path[512];
    snprintf(path, sizeof(path), "%s/ref_weights.bin", ref_path);
    bench_load_bin(path, expected, sizeof(expected));
    snprintf(path, sizeof(path), "%s/ref_metrics.bin", ref_path);
    bench_load_bin(path, &expected_metrics, sizeof(expected_metrics));
    int ok =
        memcmp(weights, expected, sizeof(expected)) == 0 &&
        metrics.margin_violations == expected_metrics.margin_violations &&
        metrics.correct_predictions == expected_metrics.correct_predictions;
    if (ok)
      printf("the result is correct\n");
    else
      printf("Mismatch: got violations=%llu accuracy=%llu\n",
             (unsigned long long)metrics.margin_violations,
             (unsigned long long)metrics.correct_predictions);
  }

  free(weights);
  free(gradients);
  table_management_free(management);
  return 0;
}
