#!/bin/bash




use_single_si=True
n_iterations=100
lr=1e-2
lab_bsize=4
tot_bsize=2048
n_trials=10
metric=val_acc
estop_patience=10
use_tuned_hpms=False
min_epochs=10
d_n=n36_gaussian_mixture_d7_100000
model_name=CGAN_GUMBEL
min_si=1
max_si=3




#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=300 --algo_variant=gumbel --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=1 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=2 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=3 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=4 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=5 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=6 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=7 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=8 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=9 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=10 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10




# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=1 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=300 --algo_variant=gumbel --n_trials=10

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=2 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=300 --algo_variant=gumbel --n_trials=10

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=3 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=300 --algo_variant=gumbel --n_trials=10

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=4 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=300 --algo_variant=gumbel --n_trials=10

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=5 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=300 --algo_variant=gumbel --n_trials=10


ulab=torch.load('/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_n36_gaussian_mixture_d7_100000/d_n_n36_gaussian_mixture_d7_100000_s_i_0_unlabel_y.pt')