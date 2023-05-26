#!/bin/sh
cd /ceph/grid/home/aj8977/HPC
make build-openmpi
module load OpenMPI/4.1.0-GCC-10.2.0 
srun --reservation=fri --time=02:00:00 --mpi=pmix -N2 -n128 build/openmpi data/wikipedia-20051105.mtx
wait

