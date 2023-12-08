#!/bin/bash

s_i=1
n_trials=10
n_iterations=100
estop_patience=10
min_epochs=10
lr=0.0001
use_single_si=False
use_tuned_hpms=False
tot_bsize=256
lab_bsize=4
use_bernoulli=False
use_benchmark_generators=False
estop_mmd_type=val_trans
plot_synthetic_dist=False
precision=32
metric=val_acc
si_max=1

d_n=n36_gaussian_mixture_d5_5000

python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 
