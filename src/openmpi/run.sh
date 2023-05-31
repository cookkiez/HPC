#!/bin/sh
export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=64
cd /ceph/grid/home/aj8977/HPC
make build-openmpi
module load OpenMPI/4.1.0-GCC-10.2.0 
srun --reservation=fri --time=02:00:00 --mpi=pmix --nodelist=nsc-msv007,nsc-msv011 -N2 -n2 build/openmpi data/scircuit.mtx
wait

