ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

SRCS := $(wildcard ./*.cc)

TESTS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../..)
INCLUDES := -I../../../../install/include/vectordpu -I$(TESTS_DIR)
LDFLAGS := -L../../../../install/lib
CXXFLAGS := -std=c++17 -O3 $(INCLUDES) $(EXTRA_FLAGS)

.PHONY: all clean

all: run

run: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -Wl,-rpath,../../../../install/lib -lvectordpu

clean:
	rm -f run
