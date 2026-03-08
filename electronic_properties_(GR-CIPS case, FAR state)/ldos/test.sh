#!/bin/bash
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --partition             amd128c
#SBATCH --nodes                 1
#SBATCH -w n03
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   500G

#singularity run -B $HOME /opt/singularity/cp2k-2024.1_openmpi_znver3_psmp.sif mpiexec cp2k -i ./celloptpbe.inp

#module load vasp/6.3.2/aocc_aocl4.2-ompi4.1
#module load aocl/4.2.0
#module load aocc/4.2.0
#module load openmpi/5.0.3
#module load gcc/14.1.0
#module load vasp/6.4.3/aocc_aocl4.2-ompi5.0

ulimit -s unlimited
(echo 125;echo 3; echo -3.5 1.5 301 0.1)|vaspkit
##mpiexec -n 20 vasp_std
#mpiexec -n 64 vasp_std #std
