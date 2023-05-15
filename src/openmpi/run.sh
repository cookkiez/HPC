#!/bin/bash
#SBATCH --job-name=vzr-task1-openmpi
#SBATCH --output=vzr-task1-openmpi.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=2            # število nalog v poslu, ki se izvajajo hkrati
#SBATCH --cpus-per-task=8
#SBATCH --reservation=fri     # rezervacija, če jo imamo; drugače vrstico zbrišemo
#SBATCH --mpi=pmix

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=8

srun ../../build/openmpi $@
wait
