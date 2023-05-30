#!/bin/bash
#SBATCH --job-name=vzr-task1-openmp
#SBATCH --output=vzr-task1-openmp.txt
#SBATCH --time=00:10:00
#SBATCH --ntasks=1            # število nalog v poslu, ki se izvajajo hkrati
#SBATCH --cpus-per-task=32
#SBATCH --reservation=fri     # rezervacija, če jo imamo; drugače vrstico zbrišemo

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=32

srun ./build/openmp $@
wait