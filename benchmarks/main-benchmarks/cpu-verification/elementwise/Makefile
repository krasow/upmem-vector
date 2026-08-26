CXX ?= g++
CXXFLAGS := -std=c++17 -O3 -fopenmp -I. -I../.. $(EXTRA_FLAGS)

.PHONY: all clean

all: generate_ref

generate_ref: generate_ref.cc Param.h
	$(CXX) $(CXXFLAGS) generate_ref.cc -o generate_ref

clean:
	rm -f generate_ref
	rm -rf data
