ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

SRCS := $(wildcard ./*.cc)
BENCHMARK_INCLUDE := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../../../main-benchmarks)
INCLUDES := -I../../../../install/include/vectordpu -I$(BENCHMARK_INCLUDE)
LDFLAGS := -L../../../../install/lib
CXXFLAGS := -std=c++17 -O3 -fopenmp $(INCLUDES) $(EXTRA_FLAGS)

.PHONY: all clean

all: run

run: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -fopenmp \
		-Wl,-rpath,../../../../install/lib -lvectordpu

clean:
	rm -f run
