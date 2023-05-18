#!/bin/bash
#SBATCH --job-name=vzr-task1-openmpi
#SBATCH --output=vzr-task1-openmpi.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1            # število nalog v poslu, ki se izvajajo hkrati
#SBATCH --cpus-per-task=1
#SBATCH --reservation=fri     # rezervacija, če jo imamo; drugače vrstico zbrišemo

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=1

cd /ceph/grid/home/aj8977/HPC
make build-openmpi
srun mpirun build/openmpi --mpi=pmix $@
wait

