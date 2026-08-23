CXX = g++
CXXFLAGS = -O3 -fopenmp -I.

all: generate_ref

generate_ref: generate_ref.cc Param.h
	$(CXX) $(CXXFLAGS) generate_ref.cc -o generate_ref

clean:
	rm -f generate_ref data/*.bin
