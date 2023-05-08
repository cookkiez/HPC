SHELL:=/bin/bash

src/cuda/sparseMV_template.cu:
	module load CUDA/10.1.243-GCC-8.3.0
	nvcc src/cuda/sparseMV_template.cu -Xcompiler -O2 src/mtx_sparse.c -o build/sparseMV --expt-relaxed-constexpr

build-cuda: src/cuda/sparseMV_template.cu

run-cuda:
	srun --reservation=fri --time=00:05:00 -G1 -n1 build/sparseMV
