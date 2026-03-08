#!/bin/bash -l

#SBATCH --job-name              process1
#SBATCH --partition             serial-short
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  24:00:00
#SBATCH --mem                   30G

# asks SLURM to send the USR1 signal 120 seconds before end of the time limit
#SBATCH --signal=B:USR1@120

source /etc/profile
source /etc/profile.d/modules.sh


# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
your_cleanup_function()
{
    echo "function your_cleanup_function called at $(date)"
    # do whatever cleanup you want here
    pkill -u hejinyan
}

# call your_cleanup_function once we receive USR1 signal
#module load gcc/7.5.0
#module load gcc/12.2.0
#module load intel/18
#module load impi/18
#module load vasp/5.4.4-huipan/intel/18
#module load vasp/6.1.1-huipan/intel/18
#module load cuda/10.0.130
#module load cmake/3.13.4
#source /data/home/yc17806/code/cp2k-7.1/tools/toolchain/install/setup

ulimit -s unlimited
NP=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
cd $SLURM_SUBMIT_DIR

#phonopy --qe --nac -c chdafrozen.in -p band.conf
#phonopy-bandplot --gnuplot >> band1.dat

conda activate mace
#python checkir.py
#python modecharge.py
#python finedraw_checkir.py
#python check.py
#python ft_draw3.py
#python vasp2pdb_modified.py
#python vasp2pdb.py

#python vasp2pdb_modified.py
##python dynamictest4.py
#python check.py
#python out_plane.py

for i in 290 310 330 350 370 390 410 450 500 550
do
  echo step$i
  mkdir $i
  cd $i
  cp ../sequent_generate_dos.py ./
  python sequent_generate_dos.py $i up_init
  cd ..
done


#./DENSITYTOOL.X < DENSITYTOOL.IN1 > out1.dat
#srun -n 1 /data/home/yc17806/code/constantc_vasp.5.4.4/bin/vasp_constant
