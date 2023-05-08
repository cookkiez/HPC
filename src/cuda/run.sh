#!/bin/sh
module load CUDA/10.1.243-GCC-8.3.0
nvcc sparseMV_template.cu -Xcompiler -O2 mtx_sparse.c -o sparseMV
#srun --reservation=fri -G1 -n1 sparseMV data/dw8192.mtx
#srun --reservation=fri -G1 -n1 sparseMV data/pdb1HYS.mtx
#srun --reservation=fri -G1 -n1 sparseMV data/scircuit.mtx
srun --reservation=fri -G1 -n1 sparseMV data/test.mtx