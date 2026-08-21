all: generate_ref

generate_ref: generate_ref.cc Param.h
	g++ -O3 -fopenmp $(EXTRA_FLAGS) generate_ref.cc -o generate_ref

clean:
	rm -f generate_ref
	rm -rf data
