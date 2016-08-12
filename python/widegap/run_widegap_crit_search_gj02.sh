#PBS -S /bin/bash
#PBS -N Widegap_Rm_Q_search_nr256_gj
#PBS -l select=20:ncpus=24:mpiprocs=24:model=has
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q devel

dedalus_script=find_single_widegap_crit

cd $PBS_O_WORKDIR

source /u/joishi1/dedalus/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/hg-projects/eigentools
export MPLBACKEND='Agg'
date
mpiexec_mpt -np 400 python3 $dedalus_script.py --k_max=1.5 --k_min=0.6 --Rm_max=5. --Rm_min=0.1 --R1=1. --R2=3. --Omega1=1. --Omega2=0.12087 --beta=41.2 --Pm=1.6e-6 --n_r=100
date
