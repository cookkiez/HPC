SHELL:=/bin/bash

build-cuda: src/cuda/cuda.cu
	module load CUDA/10.1.243-GCC-8.3.0
	nvcc src/cuda/cuda.cu -Xcompiler -O2 src/mtx_sparse.c -o build/cuda --expt-relaxed-constexpr

build-seq: src/sequential/sequential.c
	gcc src/sequential/sequential.c src/mtx_sparse.c -lm -O2 -Wall -Wno-unused-variable -o build/sequential

build-openmp: src/openmp/openmp.c
	gcc src/openmp/openmp.c src/mtx_sparse.c --openmp -lm -O2 -o build/openmp

build-openmpi: src/openmp/openmp.c
	module load OpenMPI/4.1.0-GCC-10.2.0 
	mpicc src/openmpi/openmpi.c src/mtx_sparse.c --openmp -lm -O2 -o build/openmpi

build: build-cuda build-seq build-openmp build-openmpi
