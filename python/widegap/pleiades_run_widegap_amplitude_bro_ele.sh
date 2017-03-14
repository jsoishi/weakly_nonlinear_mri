#PBS -S /bin/bash
#PBS -N WidegapFinalLongLongBroEle
#PBS -l select=2:ncpus=28:mpiprocs=28:model=bro_ele
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -q long

dedalus_script=run_widegap_amplitude

cd $PBS_O_WORKDIR

source /u/sclark9/dedalus/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/eigentools
export MPLBACKEND='Agg'
date
mpiexec_mpt -np 20 python3 $dedalus_script.py
date
