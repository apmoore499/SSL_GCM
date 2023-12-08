#!/bin/bash







#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=1 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=2 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=3 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=4 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=5 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=6 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=7 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=8 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=9 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
#python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=10 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10



d_n=n36_gaussian_mixture_d2_10000


# for si in 10,13

# for si in 35,70,77,52,75,53,24 <- d3



for si in {1..5}
do
python src/benchmarks_CGAN_disjoint.py --d_n=n36_gaussian_mixture_d2_10000 --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 --plot_synthetic_dist=True;
done




d_n=n36_gaussian_mixture_d3_10000


for si in {1..5}
do
python src/benchmarks_CGAN_disjoint.py --d_n=n36_gaussian_mixture_d2_10000 --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 --plot_synthetic_dist=True;
done


