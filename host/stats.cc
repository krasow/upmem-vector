#include "stats.h"

#include <sstream>

RuntimeStats& RuntimeStats::get() {
  static RuntimeStats instance;
  return instance;
}

std::string StatsSnapshot::to_string() const {
  std::ostringstream out;
  bool first = true;
#define VECTORDPU_STAT_PRINT(name, desc) \
  if (name != 0) {                       \
    if (!first) out << " ";              \
    out << #name << "=" << name;         \
    first = false;                       \
  }
  VECTORDPU_STAT_LIST(VECTORDPU_STAT_PRINT)
#undef VECTORDPU_STAT_PRINT
  if (first) out << "<all zero>";
  return out.str();
}
