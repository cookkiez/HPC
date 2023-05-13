#!/bin/sh
make build-seq
srun --reservation=fri -G1 -n1 build/sequential -n 1000 data/3diag100.mtx