#PBS -S /bin/bash
#PBS -N Thingap_Amplitude
#PBS -l select=2:ncpus=10:mpiprocs=10:model=has
#PBS -l walltime=20:00:00
#PBS -j oe
#PBS -q long

dedalus_script=run_amplitude

cd $PBS_O_WORKDIR

source /u/sclark9/dedalus/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/eigentools
export MPLBACKEND='Agg'
date
mpiexec_mpt -np 20 python3 $dedalus_script.py
date
