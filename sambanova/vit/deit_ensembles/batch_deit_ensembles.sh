#!/bin/sh
sbatch --nodelist=sn30-r1-h1 ./job0.sh
sbatch --nodelist=sn30-r1-h2 ./job1.sh
sbatch --nodelist=sn30-r2-h1 ./job2.sh
sbatch --nodelist=sn30-r2-h2 ./job3.sh
sbatch --nodelist=sn30-r3-h1 ./job4.sh
sbatch --nodelist=sn30-r3-h2 ./job5.sh
