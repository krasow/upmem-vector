#include <benchmark.h>
#include <dpu.h>
#include <fcntl.h>
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

typedef enum {
  QUERY_ADD,
  QUERY_ABS_DIFF,
  QUERY_AVERAGE,
  QUERY_CENTER
} query_op_t;

typedef struct {
  query_op_t op;
  uint32_t column;
} query_step_t;

typedef struct {
  query_step_t* steps;
  uint32_t count;
} query_plan_t;

static T query_abs(T value) { return value < 0 ? -value : value; }

static T column_value(uint64_t index, uint32_t column) {
  return (T)(1 + ((index * 17 + column * 53 + seed * 11) % 251));
}

static query_op_t parse_op(char code) {
  switch (code) {
    case 'A':
      return QUERY_ADD;
    case 'D':
      return QUERY_ABS_DIFF;
    case 'V':
      return QUERY_AVERAGE;
    case 'C':
      return QUERY_CENTER;
    default:
      fprintf(stderr, "unknown query operation: %c\n", code);
      exit(2);
  }
}

static query_plan_t* load_queries(uint32_t* count) {
  FILE* file = fopen(query_trace, "r");
  if (!file) {
    fprintf(stderr, "cannot open %s\n", query_trace);
    exit(2);
  }
  query_plan_t* queries = calloc(64, sizeof(*queries));
  char line[256];
  *count = 0;
  while (fgets(line, sizeof(line), file)) {
    if (line[0] == '#' || line[0] == '\n') {
      continue;
    }
    query_plan_t* plan = &queries[(*count)++];
    plan->steps = calloc(query_ops, sizeof(*plan->steps));
    char* token = strtok(line, ",\n");
    while (token && plan->count < query_ops) {
      plan->steps[plan->count].op = parse_op(token[0]);
      plan->steps[plan->count].column = (uint32_t)strtoul(token + 1, NULL, 10);
      if (plan->steps[plan->count].column >= columns) {
        fprintf(stderr, "query column exceeds configured columns\n");
        exit(2);
      }
      ++plan->count;
      token = strtok(NULL, ",\n");
    }
    if (plan->count != query_ops) {
      fprintf(stderr, "query has fewer operations than query_ops\n");
      exit(2);
    }
  }
  fclose(file);
  return queries;
}

static T project_cpu(const query_plan_t* plan, uint32_t projection,
                     uint64_t index) {
  T value = column_value(index, projection % columns);
  for (uint32_t i = 0; i < plan->count; ++i) {
    T operand =
        column_value(index, (plan->steps[i].column + projection) % columns);
    switch (plan->steps[i].op) {
      case QUERY_ADD:
        value += operand;
        break;
      case QUERY_ABS_DIFF:
        value = query_abs(value - operand);
        break;
      case QUERY_AVERAGE:
        value = (query_abs(value) + operand) >> 1;
        break;
      case QUERY_CENTER:
        value = query_abs(value + operand - 251);
        break;
    }
  }
  return column_value(index, projection % columns) <
                 column_value(index, (projection + 1) % columns)
             ? value
             : 0;
}

static T expected_max(const query_plan_t* plan, uint32_t projection) {
  T result = INT32_MIN;
  uint64_t period = nr_elements < 251 ? nr_elements : 251;
#pragma omp parallel for reduction(max : result)
  for (uint64_t i = 0; i < period; ++i) {
    T value = project_cpu(plan, projection, i);
    if (value > result) {
      result = value;
    }
  }
  return result;
}

static void free_handle_local(handle_t* handle) {
  if (!handle) {
    return;
  }
  free(handle->bin_location);
  free(handle->so_bin_location);
  free(handle);
}

int main(void) {
  if (warmup_iterations != 0) {
    fprintf(stderr, "dynamic_query requires warmup=0\n");
    return 2;
  }
  if (columns != QUERY_COLUMNS || projections == 0 || projections > columns ||
      batches_per_query == 0 || query_ops == 0 || iterations == 0) {
    fprintf(stderr, "invalid dynamic_query parameters\n");
    return 2;
  }

  uint32_t trace_queries = 0;
  query_plan_t* queries = load_queries(&trace_queries);
  if (iterations > trace_queries) {
    fprintf(stderr,
            "dynamic_query requested %u queries, but trace contains %u\n",
            iterations, trace_queries);
    return 2;
  }

  BenchStages stages;
  bench_stages_init(&stages);
  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t* management = table_management_init(dpu_number);
  bench_stage_end(&stages);

  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  query_row_t* rows =
      malloc_scatter_aligned(nr_elements, sizeof(*rows), management);
  bench_stage_end(&stages);
  bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
  for (uint64_t i = 0; i < nr_elements; ++i) {
    for (uint32_t column = 0; column < columns; ++column) {
      rows[i].values[column] = column_value(i, column);
    }
  }
  bench_stage_end(&stages);
  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("query_rows", rows, nr_elements, sizeof(*rows), management);
  bench_stage_end(&stages);

  BenchStats stats, first_stats, reuse_stats;
  BenchTimer query_timer, batch_timer;
  bench_stats_init(&stats);
  bench_stats_init(&first_stats);
  bench_stats_init(&reuse_stats);
  uint64_t checksum = seed;

  fflush(stdout);
  int saved_stdout = dup(STDOUT_FILENO);
  int quiet_stdout = open("/dev/null", O_WRONLY);
  if (saved_stdout < 0 || quiet_stdout < 0 ||
      dup2(quiet_stdout, STDOUT_FILENO) < 0) {
    fprintf(stderr, "failed to silence SimplePIM reduction logging\n");
    return 2;
  }
  close(quiet_stdout);

  for (uint32_t query = 0; query < iterations; ++query) {
    char functions[64];
    snprintf(functions, sizeof(functions), "query_%03u_funcs", query);
    handle_t* handle = NULL;
    T checked_results[batches_per_query * projections];
    bench_start(&query_timer, 0);
    for (uint32_t batch = 0; batch < batches_per_query; ++batch) {
      bench_start(&batch_timer, 0);
      bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
      if (batch == 0) {
        handle = create_handle(functions, REDUCE);
      }
      T results[QUERY_COLUMNS];
      for (uint32_t projection = 0; projection < projections; ++projection) {
        T* result = table_gen_red("query_rows", "query_result", sizeof(T), 1,
                                  handle, management, projection);
        results[projection] = result[0];
        free(result);
      }
      bench_stage_end(&stages);

      if (check_correctness) {
        memcpy(&checked_results[batch * projections], results,
               projections * sizeof(T));
      }
      for (uint32_t projection = 0; projection < projections; ++projection) {
        checksum =
            checksum * UINT64_C(1099511628211) ^ (uint64_t)results[projection];
      }
      bench_stop(&batch_timer, 0);
      bench_stats_update(batch == 0 ? &first_stats : &reuse_stats,
                         batch_timer.time[0]);
    }
    bench_stop(&query_timer, 0);
    bench_stats_update(&stats, query_timer.time[0]);

    if (check_correctness) {
      for (uint32_t projection = 0; projection < projections; ++projection) {
        T expected = expected_max(&queries[query], projection);
        for (uint32_t batch = 0; batch < batches_per_query; ++batch) {
          T actual = checked_results[batch * projections + projection];
          if (actual != expected) {
            fprintf(stderr,
                    "Mismatch in query %u, batch %u, projection %u: got %d, "
                    "expected %d\n",
                    query, batch, projection, actual, expected);
            return 1;
          }
        }
      }
    }
    free_handle_local(handle);
  }

  fflush(stdout);
  if (dup2(saved_stdout, STDOUT_FILENO) < 0) {
    fprintf(stderr, "failed to restore stdout\n");
    return 2;
  }
  close(saved_stdout);

  bench_stats_print("simplepim", &stats);
  bench_stats_print("dynamic_query_first_batch", &first_stats);
  if (reuse_stats.count > 0) {
    bench_stats_print("dynamic_query_reuse_batch", &reuse_stats);
  }
  bench_stages_report("simplepim", &stages);
  printf("simplepim_stage_query_first (ms): %.6f\n", first_stats.mean / 1000.0);
  printf("simplepim_stage_query_reuse (ms): %.6f\n",
         reuse_stats.count ? reuse_stats.mean / 1000.0 : 0.0);
  printf(
      "dynamic_query: queries=%u trace_queries=%u batches_per_query=%u "
      "query_ops=%u projections=%u checksum=%llu\n",
      iterations, trace_queries, batches_per_query, query_ops, projections,
      (unsigned long long)checksum);
  if (check_correctness) {
    printf("All results match after %u queries.\n", iterations);
  }

  for (uint32_t query = 0; query < trace_queries; ++query) {
    free(queries[query].steps);
  }
  free(queries);
  free(rows);
  table_management_free(management);
  return 0;
}
