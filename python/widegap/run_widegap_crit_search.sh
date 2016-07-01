#PBS -S /bin/bash
#PBS -N Widegap_Rm_Q_search_nr50
#PBS -l select=20:ncpus=24:mpiprocs=24:model=has
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q devel

dedalus_script=find_widegap_crit

cd $PBS_O_WORKDIR

source /u/joishi1/dedalus/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/hg-projects/eigentools
date
mpiexec_mpt -np 400 python3 $dedalus_script.py --k_max=0.3 --k_min=0.03
date
