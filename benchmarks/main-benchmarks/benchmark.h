#pragma once
// Single source of truth for benchmark timing and binary file I/O.
// C and C++ compatible (static inline, no C++ features).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct {
  struct timeval startTime[6];
  struct timeval stopTime[6];
  double time[6];
} BenchTimer;

// New API —————————————————————————————————————————————————————————————

// Reset accumulator and start timing slot i.
static inline void bench_start(BenchTimer *t, int i) {
  t->time[i] = 0.0;
  gettimeofday(&t->startTime[i], NULL);
}

// Stop and accumulate into slot i.
static inline void bench_stop(BenchTimer *t, int i) {
  gettimeofday(&t->stopTime[i], NULL);
  t->time[i] += (t->stopTime[i].tv_sec - t->startTime[i].tv_sec) * 1000000.0 +
                (t->stopTime[i].tv_usec - t->startTime[i].tv_usec);
}

// Always prints per-iteration time: total_us / (1000 * iterations).
static inline void bench_print(const char *label, BenchTimer *t, int i,
                               int iterations) {
  printf("%s (ms): %f\n", label, t->time[i] / (1000.0 * iterations));
}

// Per-iteration stats using Welford's online algorithm.
typedef struct {
  double mean;
  double M2;
  double min_us;
  double max_us;
  int count;
} BenchStats;

static inline void bench_stats_init(BenchStats *s) {
  s->mean = 0.0;
  s->M2 = 0.0;
  s->min_us = 0.0;
  s->max_us = 0.0;
  s->count = 0;
}

// Feed one elapsed_us sample (from timer.time[i] after bench_start/bench_stop).
static inline void bench_stats_update(BenchStats *s, double elapsed_us) {
  s->count++;
  double delta = elapsed_us - s->mean;
  s->mean += delta / s->count;
  s->M2 += delta * (elapsed_us - s->mean);
  if (s->count == 1) {
    s->min_us = elapsed_us;
    s->max_us = elapsed_us;
  } else {
    if (elapsed_us < s->min_us) s->min_us = elapsed_us;
    if (elapsed_us > s->max_us) s->max_us = elapsed_us;
  }
}

static inline void bench_stats_print(const char *label, BenchStats *s) {
  double mean_ms = s->mean / 1000.0;
  double stddev_ms =
      (s->count > 1) ? sqrt(s->M2 / (s->count - 1)) / 1000.0 : 0.0;
  double min_ms = s->min_us / 1000.0;
  double max_ms = s->max_us / 1000.0;
  printf("%s (ms): mean=%.3f stddev=%.3f min=%.3f max=%.3f n=%d\n", label,
         mean_ms, stddev_ms, min_ms, max_ms, s->count);
}

// Load a binary file; exits on error.
static inline void bench_load_bin(const char *filename, void *data,
                                  size_t size) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open %s for reading\n", filename);
    exit(1);
  }
  size_t got = fread(data, 1, size, f);
  if (got != size) {
    fprintf(stderr, "Short read from %s: got %zu, expected %zu\n", filename,
            got, size);
    exit(1);
  }
  fclose(f);
}

// Whole-process stage breakdown ————————————————————————————————————————
//
// A fixed, model-agnostic vocabulary so every benchmark/programming model
// reports the same stages and the sweep can reconstruct where wall-clock goes:
//   Sum(stage ms) ~= /usr/bin/time real (minus teardown / unattributed).
// Stages accumulate across the whole run (one-time load/init plus every
// iteration's write/kernel/read/merge), so wrap each region and they add up.
typedef enum {
  BENCH_STAGE_ALLOC = 0,  // host allocation / zero-fill / page touching
  BENCH_STAGE_LOAD,       // host input acquisition (bench_load_bin from file)
  BENCH_STAGE_TRANSPOSE,  // host-side layout shift/materialization
  BENCH_STAGE_INIT,   // DPU set alloc + kernel-binary load + runtime/table init
  BENCH_STAGE_WRITE,  // host -> DPU input transfer (scatter/push)
  BENCH_STAGE_KERNEL,  // on-DPU compute (launch)
  BENCH_STAGE_READ,    // DPU -> host result transfer (gather)
  BENCH_STAGE_MERGE,   // host-side post-processing / final reduction
  BENCH_STAGE_COUNT
} bench_stage_t;

static const char *const BENCH_STAGE_NAMES[BENCH_STAGE_COUNT] = {
    "alloc", "load", "transpose", "init", "write", "kernel", "read", "merge"};

// Per-stage microsecond accumulators plus a single in-flight region. Stages are
// sequential (no nesting); call bench_stage_begin/end around each region.
typedef struct {
  double us[BENCH_STAGE_COUNT];
  struct timeval _t0;
  int _active;
} BenchStages;

static inline void bench_stages_init(BenchStages *s) {
  for (int i = 0; i < BENCH_STAGE_COUNT; i++) s->us[i] = 0.0;
  s->_active = -1;
}
static inline void bench_stage_begin(BenchStages *s, bench_stage_t stage) {
  s->_active = (int)stage;
  gettimeofday(&s->_t0, NULL);
}
static inline void bench_stage_end(BenchStages *s) {
  struct timeval t1;
  gettimeofday(&t1, NULL);
  if (s->_active >= 0) {
    s->us[s->_active] +=
        (t1.tv_sec - s->_t0.tv_sec) * 1000000.0 + (t1.tv_usec - s->_t0.tv_usec);
    s->_active = -1;
  }
}
// Prints one "<label>_stage_<name> (ms): <total>" line per stage. The "_stage_"
// infix keeps these from colliding with the steady "<label> (ms):" parse.
static inline void bench_stages_report(const char *label, BenchStages *s) {
  for (int i = 0; i < BENCH_STAGE_COUNT; i++)
    printf("%s_stage_%s (ms): %f\n", label, BENCH_STAGE_NAMES[i],
           s->us[i] / 1000.0);
}
