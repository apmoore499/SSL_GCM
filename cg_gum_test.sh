#!/bin/bash




use_single_si=True
n_iterations=100
lr=1e-3
lab_bsize=4
tot_bsize=2048
n_trials=10
metric=val_acc
estop_patience=10
use_tuned_hpms=False
min_epochs=10
d_n=n36_gaussian_mixture_d7_100000
model_name=CGAN_GUMBEL
min_si=0
max_si=99
SLURM_ARRAY_TASK_ID=0








python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$SLURM_ARRAY_TASK_ID --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=300 --algo_variant=gumbel --n_trials=10

