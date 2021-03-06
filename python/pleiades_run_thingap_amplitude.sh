#PBS -S /bin/bash
#PBS -N ThingapAmp256
#PBS -l select=2:ncpus=10:mpiprocs=10:model=bro_ele
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q devel

dedalus_script=run_amplitude

cd $PBS_O_WORKDIR

source /u/sclark9/dedalus/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/eigentools
export MPLBACKEND='Agg'
date
mpiexec_mpt -np 20 python3 $dedalus_script.py
date
