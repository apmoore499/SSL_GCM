#!/bin/bash

s_i=1
n_iterations=100
lr=1e-3
use_single_si=False
tot_bsize=256
lab_bsize=4
use_bernoulli=False
use_benchmark_generators=False
estop_patience=5
estop_mmd_type=val_trans
plot_synthetic_dist=False
precision=16	
n_trials=10
use_gpu=True
metric=val_acc
estop_patience=10
min_epochs=10
si_max=1
nsamps=-1
ulimit -n 4096     





d_n=n36_gaussian_mixture_d7_5000


python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10
