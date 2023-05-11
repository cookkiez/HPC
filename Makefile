SHELL:=/bin/bash

build-cuda: src/cuda/sparseMV_template.cu
	module load CUDA/10.1.243-GCC-8.3.0
	nvcc src/cuda/sparseMV_template.cu -Xcompiler -O2 src/mtx_sparse.c -o build/sparseMV --expt-relaxed-constexpr

run-cuda:
	srun --reservation=fri --time=00:05:00 -G1 -n1 build/sparseMV

build-seq: src/sequential/sequential.c
	gcc src/sequential/sequential.c src/mtx_sparse.c -lm -Wall -Wno-unused-variable -o build/sequential

build-openmp: src/openmp/openmp.c
	gcc src/openmp/openmp.c src/mtx_sparse.c --openmp -lm -o build/openmp

build: build-cuda build-seq
