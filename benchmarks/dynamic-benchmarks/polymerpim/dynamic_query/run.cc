#include <benchmark.h>
#include <omp.h>
#include <polymerpim.h>

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

using namespace polymerpim;

enum class QueryOp : uint8_t { ADD, ABS_DIFF, AVERAGE, CENTER };

struct QueryStep {
  QueryOp op;
  uint32_t column;
};

using QueryPlan = std::vector<QueryStep>;
using QueryResult = typename DPUVector<T>::reduction_result_t;

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
  if (!file) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }

  std::vector<QueryPlan> queries;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    QueryPlan plan;
    std::stringstream fields(line);
    std::string token;
    while (std::getline(fields, token, ',')) {
      if (token.size() < 2) {
        throw std::runtime_error("invalid query token: " + token);
      }
      uint32_t column = (uint32_t)std::stoul(token.substr(1));
      if (column >= columns) {
        throw std::runtime_error("query column exceeds configured columns");
      }
      plan.push_back({parse_op(token[0]), column});
    }
    if (query_ops == 0 || plan.size() < query_ops) {
      throw std::runtime_error("query has fewer operations than query_ops");
    }
    plan.resize(query_ops);
    queries.push_back(std::move(plan));
  }
  return queries;
}

static auto project(const QueryPlan& plan,
                    const std::vector<DPUVector<T>>& input) {
  using Expression = decltype(input[0] + (T)0);
  Expression value = input[0];
  for (const QueryStep& step : plan) {
    const auto& operand = input[step.column % columns];
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
  return value * (input[0] < input[1]);
}

static T project_cpu(const QueryPlan& plan, uint64_t index) {
  T value = column_value(index, 0);
  for (const QueryStep& step : plan) {
    T operand = column_value(index, step.column % columns);
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
  return column_value(index, 0) < column_value(index, 1) ? value : 0;
}

static QueryResult expected_max(const QueryPlan& plan) {
  T result = std::numeric_limits<T>::lowest();
  const uint64_t period = std::min<uint64_t>(N, 251);
#pragma omp parallel for reduction(max : result)
  for (uint64_t i = 0; i < period; ++i) {
    result = std::max(result, project_cpu(plan, i));
  }
  return (QueryResult)result;
}

int main() {
  try {
    if (warmup_iterations != 0) {
      std::cerr << "dynamic_query requires warmup=0" << std::endl;
      return 2;
    }
    if (columns < 2 || batches_per_query == 0 || iterations == 0 ||
        query_ops == 0) {
      std::cerr << "dynamic_query requires columns>=2, batches_per_query>0, "
                   "iterations>0, and query_ops>0"
                << std::endl;
      return 2;
    }

    const char* nr_dpus_env = std::getenv("NR_DPUS");
    const uint32_t nr_dpus = nr_dpus_env ? std::stoi(nr_dpus_env) : 64;
    BenchStages stages;
    bench_stages_init(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_INIT);
    init(nr_dpus);
    bench_stage_end(&stages);

    bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
    std::vector<T> host_column(N);
    std::vector<DPUVector<T>> input;
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
      for (uint64_t i = 0; i < N; ++i) {
        host_column[i] = column_value(i, column);
      }
      bench_stage_end(&stages);

      bench_stage_begin(&stages, BENCH_STAGE_WRITE);
      input.push_back(DPUVector<T>(host_column, "query_column"));
      sync();
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
    RuntimeStatistics runtime_before = statistics();

    uint64_t checksum = seed;

    for (uint32_t query = 0; query < iterations; ++query) {
      const QueryPlan& plan = queries[query];
      std::vector<QueryResult> checked_results;
      if (check_correctness) {
        checked_results.reserve(batches_per_query);
      }
      bench_start(&timer, 0);

      for (uint32_t batch = 0; batch < batches_per_query; ++batch) {
        bench_start(&batch_timer, 0);
        bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
        DpuFuture<T> pending = maximum(project(plan, input));
        sync();
        bench_stage_end(&stages);

        bench_stage_begin(&stages, BENCH_STAGE_READ);
        QueryResult result = pending.get();
        bench_stage_end(&stages);

        if (check_correctness) {
          checked_results.push_back(result);
        }

        checksum = checksum * UINT64_C(1099511628211) ^ (uint64_t)result;

        bench_stop(&batch_timer, 0);
        bench_stats_update(batch == 0 ? &first_batch_stats : &reuse_batch_stats,
                           batch_timer.time[0]);
      }

      bench_stop(&timer, 0);
      bench_stats_update(&stats, timer.time[0]);

      if (check_correctness) {
        QueryResult expected = expected_max(plan);
        for (uint32_t batch = 0; batch < batches_per_query; ++batch) {
          if (checked_results[batch] != expected) {
            std::cerr << "Mismatch in query " << query << ", batch " << batch
                      << ": got " << checked_results[batch] << ", expected "
                      << expected << std::endl;
            return 1;
          }
        }
      }
    }

    RuntimeStatistics runtime_stats = statistics() - runtime_before;
#if JIT_PIPELINE_FALLBACK
    size_t jit_pipeline_fallbacks = runtime_stats.jit_pipeline_fallbacks;
    size_t jit_eager_fallbacks = runtime_stats.jit_eager_fallbacks;
#else
    size_t jit_pipeline_fallbacks = 0;
    size_t jit_eager_fallbacks = 0;
#endif

    bench_stats_print("polymerpim", &stats);
    bench_stats_print("dynamic_query_first_batch", &first_batch_stats);
    if (reuse_batch_stats.count > 0) {
      bench_stats_print("dynamic_query_reuse_batch", &reuse_batch_stats);
    }
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
              << " query_ops=" << query_ops << " checksum=" << checksum
              << " compute_launches=" << runtime_stats.compute_launches
              << " vertical_fusions=" << runtime_stats.vertical_fusions
              << " horizontal_fusions=" << runtime_stats.horizontal_fusions
              << " jit_kernel_compiles=" << runtime_stats.jit_kernel_compiles
              << " jit_kernel_cache_hits="
              << runtime_stats.jit_kernel_cache_hits
              << " binary_switches=" << runtime_stats.binary_switches
              << " jit_pipeline_fallbacks=" << jit_pipeline_fallbacks
              << " jit_eager_fallbacks=" << jit_eager_fallbacks << std::endl;

    if (check_correctness) {
      std::cout << "All results match after " << iterations << " queries."
                << std::endl;
    }

    shutdown();
    return 0;
  } catch (const OutOfMemory& error) {
    std::cerr << "DPU OOM: " << error.what() << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Exception: " << error.what() << std::endl;
    return 1;
  }
}
