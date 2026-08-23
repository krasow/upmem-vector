#include <benchmark.h>
#include <omp.h>
#include <runtime.h>
#include <stats.h>
#include <vectordpu.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Param.h"

enum class QueryOp : uint8_t { ADD, ABS_DIFF, AVERAGE, CENTER };

struct QueryStep {
  QueryOp op;
  uint32_t column;
};

using QueryPlan = std::vector<QueryStep>;
using QueryResult = typename dpu_vector<T>::reduction_result_t;

static T column_value(uint64_t index, uint32_t column) {
  return (T)(1 + ((index * 17 + column * 53 + seed * 11) % 251));
}

static QueryOp parse_op(char code) {
  switch (code) {
    case 'A':
      return QueryOp::ADD;
    case 'D':
      return QueryOp::ABS_DIFF;
    case 'V':
      return QueryOp::AVERAGE;
    case 'C':
      return QueryOp::CENTER;
    default:
      throw std::runtime_error(std::string("unknown query operation: ") + code);
  }
}

static std::vector<QueryPlan> load_queries(const char* path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error(std::string("cannot open ") + path);

  std::vector<QueryPlan> queries;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    QueryPlan plan;
    std::stringstream fields(line);
    std::string token;
    while (std::getline(fields, token, ',')) {
      if (token.size() < 2)
        throw std::runtime_error("invalid query token: " + token);
      uint32_t column = (uint32_t)std::stoul(token.substr(1));
      if (column >= columns)
        throw std::runtime_error("query column exceeds configured columns");
      plan.push_back({parse_op(token[0]), column});
    }
    if (query_ops == 0 || plan.size() < query_ops)
      throw std::runtime_error("query has fewer operations than query_ops");
    plan.resize(query_ops);
    queries.push_back(std::move(plan));
  }
  return queries;
}

static dpu_vector<T> project(const QueryPlan& plan, uint32_t projection,
                             const std::vector<dpu_vector<T>>& input) {
  dpu_vector<T> value = input[projection % columns];
  for (const QueryStep& step : plan) {
    const auto& operand = input[(step.column + projection) % columns];
    switch (step.op) {
      case QueryOp::ADD:
        value = value + operand;
        break;
      case QueryOp::ABS_DIFF:
        value = abs(value - operand);
        break;
      case QueryOp::AVERAGE:
        value = (abs(value) + operand) >> (T)1;
        break;
      case QueryOp::CENTER:
        value = abs(value + operand - (T)251);
        break;
    }
  }
  const auto& lhs = input[projection % columns];
  const auto& rhs = input[(projection + 1) % columns];
  return value * (lhs < rhs);
}

static T project_cpu(const QueryPlan& plan, uint32_t projection,
                     uint64_t index) {
  T value = column_value(index, projection % columns);
  for (const QueryStep& step : plan) {
    T operand = column_value(index, (step.column + projection) % columns);
    switch (step.op) {
      case QueryOp::ADD:
        value += operand;
        break;
      case QueryOp::ABS_DIFF:
        value = std::abs(value - operand);
        break;
      case QueryOp::AVERAGE:
        value = (std::abs(value) + operand) >> 1;
        break;
      case QueryOp::CENTER:
        value = std::abs(value + operand - 251);
        break;
    }
  }
  T lhs = column_value(index, projection % columns);
  T rhs = column_value(index, (projection + 1) % columns);
  return lhs < rhs ? value : 0;
}

static QueryResult expected_max(const QueryPlan& plan, uint32_t projection) {
  T result = std::numeric_limits<T>::lowest();
  const uint64_t period = std::min<uint64_t>(N, 251);
#pragma omp parallel for reduction(max : result)
  for (uint64_t i = 0; i < period; ++i)
    result = std::max(result, project_cpu(plan, projection, i));
  return (QueryResult)result;
}

int main() {
  try {
    if (warmup_iterations != 0) {
      std::cerr << "dynamic_query requires warmup=0" << std::endl;
      return 2;
    }
    if (columns < 2 || projections == 0 || projections > columns ||
        batches_per_query == 0 || iterations == 0 || query_ops == 0) {
      std::cerr
          << "dynamic_query requires columns>=2, 1<=projections<=columns, "
             "batches_per_query>0, iterations>0, and query_ops>0"
          << std::endl;
      return 2;
    }

    const char* nr_dpus_env = std::getenv("NR_DPUS");
    const uint32_t nr_dpus = nr_dpus_env ? std::stoi(nr_dpus_env) : 64;
    BenchStages stages;
    bench_stages_init(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_INIT);
    DpuRuntime::get().init(nr_dpus);
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
    std::vector<T> host_column(N);
    std::vector<dpu_vector<T>> input;
    input.reserve(columns);
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    std::vector<QueryPlan> queries = load_queries(query_trace);
    bench_stage_end(&stages);
    if (iterations > queries.size()) {
      std::cerr << "dynamic_query requested " << iterations << " queries, but "
                << query_trace << " contains " << queries.size() << std::endl;
      return 2;
    }

    for (uint32_t column = 0; column < columns; ++column) {
      bench_stage_begin(&stages, BENCH_STAGE_LOAD);
#pragma omp parallel for
      for (uint64_t i = 0; i < N; ++i) host_column[i] = column_value(i, column);
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_WRITE);
      input.push_back(dpu_vector<T>::from_cpu(host_column, "query_column"));
      dpu_fence();
      bench_stage_end(&stages);
    }

    BenchStats stats;
    BenchStats first_batch_stats;
    BenchStats reuse_batch_stats;
    BenchTimer timer;
    BenchTimer batch_timer;
    bench_stats_init(&stats);
    bench_stats_init(&first_batch_stats);
    bench_stats_init(&reuse_batch_stats);
    StatsSnapshot runtime_before = RuntimeStats::get().snapshot();

    uint64_t checksum = seed;

    for (uint32_t query = 0; query < iterations; ++query) {
      const QueryPlan& plan = queries[query];
      std::vector<QueryResult> checked_results;
      if (check_correctness)
        checked_results.reserve(batches_per_query * projections);
      bench_start(&timer, 0);

      for (uint32_t batch = 0; batch < batches_per_query; ++batch) {
        bench_start(&batch_timer, 0);
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        std::vector<dpu_future<T>> pending;
        pending.reserve(projections);
        for (uint32_t projection = 0; projection < projections; ++projection)
          pending.push_back(max(project(plan, projection, input)));
        dpu_fence();
        bench_stage_end(&stages);

        bench_stage_begin(&stages, BENCH_STAGE_READ);
        std::vector<QueryResult> results;
        results.reserve(projections);
        for (auto& future : pending) results.push_back(future.get());
        bench_stage_end(&stages);

        if (check_correctness)
          checked_results.insert(checked_results.end(), results.begin(),
                                 results.end());

        for (QueryResult result : results)
          checksum = checksum * UINT64_C(1099511628211) ^ (uint64_t)result;

        bench_stop(&batch_timer, 0);
        bench_stats_update(batch == 0 ? &first_batch_stats : &reuse_batch_stats,
                           batch_timer.time[0]);
      }

      bench_stop(&timer, 0);
      bench_stats_update(&stats, timer.time[0]);

      if (check_correctness) {
        for (uint32_t projection = 0; projection < projections; ++projection) {
          QueryResult expected = expected_max(plan, projection);
          for (uint32_t batch = 0; batch < batches_per_query; ++batch) {
            QueryResult actual =
                checked_results[batch * projections + projection];
            if (actual != expected) {
              std::cerr << "Mismatch in query " << query << ", batch " << batch
                        << ", projection " << projection << ": got " << actual
                        << ", expected " << expected << std::endl;
              return 1;
            }
          }
        }
      }
    }

    StatsSnapshot runtime_stats =
        RuntimeStats::get().snapshot() - runtime_before;
#if JIT_PIPELINE_FALLBACK
    size_t jit_pipeline_fallbacks = runtime_stats.jit_pipeline_fallbacks;
    size_t jit_eager_fallbacks = runtime_stats.jit_eager_fallbacks;
#else
    size_t jit_pipeline_fallbacks = 0;
    size_t jit_eager_fallbacks = 0;
#endif

    bench_stats_print("polymerpim", &stats);
    bench_stats_print("dynamic_query_first_batch", &first_batch_stats);
    if (reuse_batch_stats.count > 0)
      bench_stats_print("dynamic_query_reuse_batch", &reuse_batch_stats);
    bench_stages_report("polymerpim", &stages);
    std::cout << "polymerpim_stage_query_first (ms): "
              << first_batch_stats.mean / 1000.0 << '\n';
    std::cout << "polymerpim_stage_query_reuse (ms): "
              << (reuse_batch_stats.count ? reuse_batch_stats.mean / 1000.0
                                          : 0.0)
              << '\n';
    std::cout << "dynamic_query: queries=" << iterations
              << " trace_queries=" << queries.size()
              << " batches_per_query=" << batches_per_query
              << " query_ops=" << query_ops << " projections=" << projections
              << " checksum=" << checksum
              << " compute_launches=" << runtime_stats.compute_launches
              << " vertical_fusions=" << runtime_stats.vertical_fusions
              << " horizontal_fusions=" << runtime_stats.horizontal_fusions
              << " jit_kernel_compiles=" << runtime_stats.jit_kernel_compiles
              << " jit_kernel_cache_hits="
              << runtime_stats.jit_kernel_cache_hits
              << " binary_switches=" << runtime_stats.binary_switches
              << " jit_pipeline_fallbacks=" << jit_pipeline_fallbacks
              << " jit_eager_fallbacks=" << jit_eager_fallbacks << std::endl;

    if (check_correctness)
      std::cout << "All results match after " << iterations << " queries."
                << std::endl;

    DpuRuntime::get().shutdown();
    return 0;
  } catch (const DpuOOMException& error) {
    std::cerr << "DPU OOM: " << error.what() << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Exception: " << error.what() << std::endl;
    return 1;
  }
}
