#!/bin/bash








use_single_si=True
n_iterations=100
lr=1e-3
lab_bsize=4
tot_bsize=256 #change from 256!!
n_trials=10
metric=val_acc
estop_patience=10
use_tuned_hpms=False
min_epochs=10
model_name=CGAN_BASIC_DJ_CLASSIFIER
min_si=1
max_si=3

nsamps=-1 #300



d_n=n36_gaussian_mixture_d3_10000
model_name=CGAN_GUMBEL_DJ_CLASSIFIER

s_i=1
n_iterations=100
lr=1e-3
precision=16
use_single_si=True
tot_bsize=256
lab_bsize=4
use_bernoulli=False
use_benchmark_generators=False
estop_patience=10
estop_mmd_type=val_trans
plot_synthetic_dist=False


d_n=n36_gaussian_mixture_d3_10000



# for s_i in {0..4}
# do
# 	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=True --estop_mmd_type=$estop_mmd_type;

# done


#python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=24 --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=True --estop_mmd_type=$estop_mmd_type;


for s_i in {47..99}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

done


for s_i in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=100 --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10 --estop_patience=$estop_patience;
done



d_n=n36_gaussian_mixture_d5_10000



for s_i in {0..4}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=True --estop_mmd_type=$estop_mmd_type;

done

for s_i in {5..99}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

done


for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=100 --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10 --estop_patience=$estop_patience;
done

exit

d_n=n36_gaussian_mixture_d6_10000


for s_i in {0..4}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=True --estop_mmd_type=$estop_mmd_type;

done

for s_i in {5..99}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

done

for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=100 --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10 --estop_patience=$estop_patience;
done

d_n=n36_gaussian_mixture_d7_10000

for s_i in {0..4}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=True --estop_mmd_type=$estop_mmd_type;

done

for s_i in {5..99}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

done


for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=100 --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10 --estop_patience=$estop_patience;
done

# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=1 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=2 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=3 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=4 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=5 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=6 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=7 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=8 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=9 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10
# #python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --s_i=10 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --n_trials=10



# d_n=n36_gaussian_mixture_d2_10000


# # for si in 10,13

# # for si in 35,70,77,52,75,53,24 <- d3

# # si_range=100

# # python src/benchmarks_CGAN_disjoint.py --d_n=n36_gaussian_mixture_d2_10000 --lr=1e-3 --s_i=$si_range --n_iterations=100 --precision=16 --use_single_si=False --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 --plot_synthetic_dist=False --estop_mmd_type=val_trans;

# # exit

# # for si in {95..99}
# # do
# # 	python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1e-3 --s_i=$si --n_iterations=200 --precision=16 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=50 --plot_synthetic_dist=False --estop_mmd_type=val_trans;
	
# # done





# # python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1e-3 --s_i=94 --n_iterations=10 --precision=16 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 --plot_synthetic_dist=False --estop_mmd_type=val_trans;
# # exit







# #for si in {71,82,24,15,83}
# for si in {47..99}
for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10

done




# d_n=n36_gaussian_mixture_d3_10000


# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1e-3 --s_i=$si --n_iterations=200 --precision=16 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=50 --plot_synthetic_dist=False --estop_mmd_type=val_trans;
# done





# use_single_si=True
# n_iterations=100
# lr=1e-3
# lab_bsize=4
# tot_bsize=256
# n_trials=10
# metric=val_acc
# estop_patience=10
# use_tuned_hpms=False
# min_epochs=10
# model_name=CGAN_BASIC_DJ_CLASSIFIER
# min_si=1
# max_si=3




# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done




# d_n=n36_gaussian_mixture_d4_10000


# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1e-3 --s_i=$si --n_iterations=200 --precision=16 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=50 --plot_synthetic_dist=False --estop_mmd_type=val_trans;

# done



# use_single_si=True
# n_iterations=100
# lr=1e-3
# lab_bsize=4
# tot_bsize=256
# n_trials=10
# metric=val_acc
# estop_patience=10
# use_tuned_hpms=False
# min_epochs=10
# model_name=CGAN_BASIC_DJ_CLASSIFIER
# min_si=1
# max_si=3




# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done


d_n=n36_gaussian_mixture_d5_10000


# for si in {0..99}
# do
# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# done



use_single_si=True
n_iterations=100
lr=1e-3
lab_bsize=4
tot_bsize=256
n_trials=10
metric=val_acc
estop_patience=10
use_tuned_hpms=False
min_epochs=10
model_name=CGAN_BASIC_DJ_CLASSIFIER
min_si=1
max_si=3




for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
done


d_n=n36_gaussian_mixture_d6_10000


# for si in {0..99}
# do
# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# done



use_single_si=True
n_iterations=100
lr=1e-3
lab_bsize=4
tot_bsize=256
n_trials=10
metric=val_acc
estop_patience=10
use_tuned_hpms=False
min_epochs=10
model_name=CGAN_BASIC_DJ_CLASSIFIER
min_si=1
max_si=3




for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
done


d_n=n36_gaussian_mixture_d7_10000


# for si in {0..99}
# do
# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;
# done



use_single_si=True
n_iterations=100
lr=1e-3
lab_bsize=4
tot_bsize=256
n_trials=10
metric=val_acc
estop_patience=10
use_tuned_hpms=False
min_epochs=10
model_name=CGAN_BASIC_DJ_CLASSIFIER
min_si=1
max_si=3




for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
done




# d_n=n36_gaussian_mixture_d2_10000

# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done

# d_n=n36_gaussian_mixture_d3_10000

# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done

# d_n=n36_gaussian_mixture_d4_10000


# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done


# d_n=n36_gaussian_mixture_d5_10000

# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done

# d_n=n36_gaussian_mixture_d6_10000

# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done

# d_n=n36_gaussian_mixture_d7_10000

# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# done

# 	python src/benchmarks_CGAN_disjoint.py --d_n=n36_gaussian_mixture_d7_100000 --lr=1e-3 --s_i=$si --n_iterations=100 --lr=1e-3 --use_single_si=True --tot_bsize=2048 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 ;


#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=1 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=2 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=3 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=4 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=5 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=6 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=7 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=8 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=9 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10
#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=10 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=3 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=4 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=5 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10


#ulab=torch.load('/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_n36_gaussian_mixture_d7_100000/d_n_n36_gaussian_mixture_d7_100000_s_i_0_unlabel_y.pt')