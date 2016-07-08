#PBS -S /bin/bash
#PBS -N Widegap_Rm_Q_search_nr50
#PBS -l select=1:ncpus=24:mpiprocs=24:model=has
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q devel

dedalus_script=run_pm_sweep

cd $PBS_O_WORKDIR

source /u/joishi1/dedalus/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/hg-projects/eigentools
export MPLBACKEND='Agg'
date
mpiexec_mpt -np 20 python3 $dedalus_script.py
date
