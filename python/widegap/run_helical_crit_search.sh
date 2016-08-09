#PBS -S /bin/bash
#PBS -N Helical_Rm_Q_search_nr100
#PBS -l select=20:ncpus=24:mpiprocs=24:model=has
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q devel

dedalus_script=find_widegap_crit

cd $PBS_O_WORKDIR

source /u/sclark9/dedalus/bin/activate
export PYTHONPATH=$PYTHONPATH:$HOME/eigentools
export MPLBACKEND='Agg'
date
mpiexec_mpt -np 400 python3 $dedalus_script.py --R1=1 --R2=2 --Omega1=313.55 --Omega2=87.794 --Pm=1.0E-6 --beta=0.0174 --xi=4.0 --Rm_min=1.0E-4 --Rm_max=1.0E1 --k_min=1.0E-1 --k_max=50.0 --insulate=1
date
