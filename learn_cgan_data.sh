#!/bin/bash

si=1

#python src/benchmarks_CGAN_disjoint.py --d_n=n36_gaussian_mixture_d7_100000 --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=2048 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;


python src/benchmarks_CGAN_gumbel_disjoint.py --d_n=n36_gaussian_mixture_d7_10000 --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;


# for si in {2..10}
# do
# 	python src/benchmarks_CGAN_disjoint.py --d_n=n36_gaussian_mixture_d7_100000 --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=2048 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# done


# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d3_10000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end


# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d5_10000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end


# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d6_10000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end


# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d7_10000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end



# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d3_100000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end


# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d5_100000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end


# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d6_100000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end


# for si in (seq 0 99)
# 	python src/benchmarks_CGAN_gumbel.py --d_n=n36_gaussian_mixture_d7_100000 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# end

