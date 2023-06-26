#!/bin/bash
cd ${PBS_O_WORKDIR}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=5
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

module load conda
conda activate
source /home/epautsch/GitRepos/UncertaintyANL/adv2it/my_env/bin/activate
time mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth /home/jneprz/UncertaintyANL/adv2it_dynamic/set_affinity_gpu_polaris.sh python3 /home/jneprz/UncertaintyANL/adv2it_dynamic/test.py
