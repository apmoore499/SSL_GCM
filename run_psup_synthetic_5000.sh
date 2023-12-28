#!/bin/bash

s_i=0
n_iterations=100
lr=1e-3
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
estop_patience=5
min_epochs=10
si_max=1
nsamps=-1
ulimit -n 4096     

use_single_si=False


d_n=n36_gaussian_mixture_d1_5000

python src/benchmarks_LABEL_PROPAGATION.py  --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 




d_n=n36_gaussian_mixture_d2_5000

python src/benchmarks_LABEL_PROPAGATION.py  --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 

d_n=n36_gaussian_mixture_d3_5000

python src/benchmarks_LABEL_PROPAGATION.py  --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 

d_n=n36_gaussian_mixture_d4_5000

python src/benchmarks_LABEL_PROPAGATION.py  --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


d_n=n36_gaussian_mixture_d5_5000

python src/benchmarks_LABEL_PROPAGATION.py  --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


d_n=n36_gaussian_mixture_d6_5000

python src/benchmarks_LABEL_PROPAGATION.py  --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


d_n=n36_gaussian_mixture_d7_5000


python src/benchmarks_LABEL_PROPAGATION.py  --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


#python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


#python src/--s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10



exit


# d_n=n36_gaussian_mixture_d5_5000

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10

# d_n=n36_gaussian_mixture_d6_5000


# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10



# # d_n=n36_gaussian_mixture_d5_10000


# # python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

# # python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;




# d_n=n36_gaussian_mixture_d5_10000


# python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10



# d_n=n36_gaussian_mixture_d6_10000


# python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200  --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;



# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10


# d_n=n36_gaussian_mixture_d7_10000


# python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;




# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10














# d_n=n36_gaussian_mixture_d1_5000


# #python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1$lr --s_i=0 --n_iterations=$n_iterations --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10




# d_n=n36_gaussian_mixture_d2_5000


# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1$lr --s_i=0 --n_iterations=$n_iterations --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;


# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10



# d_n=n36_gaussian_mixture_d3_5000


# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1$lr --s_i=0 --n_iterations=$n_iterations --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;





# d_n=n36_gaussian_mixture_d4_5000


# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1$lr --s_i=0 --n_iterations=$n_iterations --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;




# d_n=n36_gaussian_mixture_d5_5000

# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1$lr --s_i=0 --n_iterations=$n_iterations --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;





# d_n=n36_gaussian_mixture_d6_5000

# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1$lr --s_i=0 --n_iterations=$n_iterations --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;




d_n=n36_gaussian_mixture_d7_5000

python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=1$lr --s_i=0 --n_iterations=$n_iterations --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;




d_n=n36_gaussian_mixture_d3_5000


python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10


d_n=n36_gaussian_mixture_d4_5000

python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10




python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10




d_n=n36_gaussian_mixture_d6_5000

python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10



d_n=n36_gaussian_mixture_d7_5000



python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10


















#xit



# #source gumbel_disjoint_50000_d3.sh

# source gumbel_disjoint_5000_d5.sh
# source gumbel_disjoint_5000_d6.sh
# source gumbel_disjoint_5000_d7.sh














# d_n=n36_gaussian_mixture_d1_5000


# python src/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 


# d_n=n36_gaussian_mixture_d2_5000


# python src/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# d_n=n36_gaussian_mixture_d3_5000

# python src/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 


# d_n=n36_gaussian_mixture_d4_5000

# python src/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# d_n=n36_gaussian_mixture_d5_5000

# python src/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 


# d_n=n36_gaussian_mixture_d6_5000

# python src/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# d_n=n36_gaussian_mixture_d7_5000

# python src/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_gpu=$use_gpu --metric=$metric --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 






# d_n=n36_gaussian_mixture_d1_5000


# #python src/benchmarks_SSL_VAE.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# #python src/benchmarks_TRIPLE_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# #python src/benchmarks_ENTROPY_MINIMISATION.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# #python src/benchmarks_LABEL_PROPAGATION.py --d_n=$d_n --s_i=0 --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=True --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 





# # d_n=n36_gaussian_mixture_d2_5000


# # #python src/benchmarks_SSL_VAE.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# # #python src/benchmarks_TRIPLE_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# # python src/benchmarks_ENTROPY_MINIMISATION.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# # python src/benchmarks_LABEL_PROPAGATION.py --d_n=$d_n --s_i=0 --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=True --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 




# d_n=n36_gaussian_mixture_d3_5000



# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 







# #python src/benchmarks_SSL_VAE.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_TRIPLE_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_ENTROPY_MINIMISATION.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_LABEL_PROPAGATION.py --d_n=$d_n --s_i=0 --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=True --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 




# d_n=n36_gaussian_mixture_d4_5000
# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_SSL_VAE.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_TRIPLE_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_ENTROPY_MINIMISATION.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_LABEL_PROPAGATION.py --d_n=$d_n --s_i=0 --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=True --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 





# d_n=n36_gaussian_mixture_d5_5000
# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_SSL_VAE.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_TRIPLE_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_ENTROPY_MINIMISATION.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_LABEL_PROPAGATION.py --d_n=$d_n --s_i=0 --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=True --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 




# d_n=n36_gaussian_mixture_d6_5000
# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_SSL_VAE.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_TRIPLE_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_ENTROPY_MINIMISATION.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_LABEL_PROPAGATION.py --d_n=$d_n --s_i=0 --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=True --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 




# d_n=n36_gaussian_mixture_d7_5000
# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 

# python src/benchmarks_SSL_VAE.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_TRIPLE_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_ENTROPY_MINIMISATION.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 
# python src/benchmarks_LABEL_PROPAGATION.py --d_n=$d_n --s_i=0 --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=True --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 






# d_n=n36_gaussian_mixture_d1_5000




# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 


# d_n=n36_gaussian_mixture_d2_5000


# python src/benchmarks_VAT.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=False --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=False --estop_patience=$estop_patience --min_epochs=$min_epochs 





# for si in {0..99}
# do
# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10
# done







for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10
done





d_n=n36_gaussian_mixture_d6_5000





for s_i in {0..99}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

done



for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10
done




d_n=n36_gaussian_mixture_d7_5000







for s_i in {0..99}
do
	python src/benchmarks_CGAN_gumbel_disjoint_ptl.py --d_n=$d_n --lr=1$lr --s_i=$s_i --n_iterations=200 --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=30 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type;

done




for si in {0..99}
do
	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$si --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10
done