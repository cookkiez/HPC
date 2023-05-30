#!/bin/bash

module load CUDA/10.1.243-GCC-8.3.0

srun \
  --reservation=fri \
  -G1 \
  -n1 \
  --time=00:10:00 \
  --output=vzr-task1-cuda.txt \
  --job-name=vzr-task1-cuda \
  ./build/cuda $@
