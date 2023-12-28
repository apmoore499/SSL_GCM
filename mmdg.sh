
s_i=0
n_iterations=100
lr=1e-3
tot_bsize=32
lab_bsize=4
use_bernoulli=False
use_benchmark_generators=False
#estop_patience=5
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

d_n=n36_gaussian_mixture_d6_10000
use_single_si=False




# python src/benchmarks_MMD_GAN.py --d_n=n36_gaussian_mixture_d6_10000 --s_i=1 --n_iterations=100 --lr=1e-3 --precision=16 --use_single_si=False --tot_bsize=256 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 --estop_mmd_type=val_trans --precision=16 --plot_synthetic_dist=False --compile_mmd_mode=max-autotune

# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=mmd_gan --n_trials=10


d_n=real_bcancer_diagnosis_zscore



# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si   --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=True;


# python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si   --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=10 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=True;



# s_i=10
# s_i=44


# s_i=33


# python src/benchmarks_CGAN_joint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si   --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=20 --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=True --compile_mmd_mode='reduce-overhead' 



# #python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10


# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10 --estop_patience=$estop_patience


tot_bsize=256




si=36

si=44

si=74

use_single_si=False

plot_synthetic_dist=False


estop_patience=10



d_n=real_sachs_raf_log






# python src/benchmarks_CGAN_gumbel_joint_ptl.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si  --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=$estop_patience --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=False --compile_mmd_mode='reduce-overhead' 



n_pretrain_epo=5
estop_patience=5

python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel_disjoint --n_trials=10 --estop_patience=$estop_patience --n_pretrain_epo=$n_pretrain_epo

exit






d_n=real_sachs_bcancer_zscore

estop_patience=10





python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si  --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=$estop_patience --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=False --compile_mmd_mode='reduce-overhead' 










n_pretrain_epo=5

estop_patience=5

python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10 --estop_patience=$estop_patience --n_pretrain_epo=$n_pretrain_epo







d_n=real_sachs_raf_log
estop_patience=10






python src/benchmarks_CGAN_disjoint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si  --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=$estop_patience --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=False --compile_mmd_mode='reduce-overhead' 




n_pretrain_epo=5

estop_patience=5

python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10 --estop_patience=$estop_patience --n_pretrain_epo=$n_pretrain_epo


exit






# python src/benchmarks_CGAN.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si  --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=$estop_patience  --estop_mmd_type=$estop_mmd_type --ignore_plot_5=True --compile_mmd_mode='reduce-overhead'  --plot_synthetic_dist=$plot_synthetic_dist



n_pretrain_epo=5

n_pretrain_epo=5

estop_patience=5


python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10 --estop_patience=$estop_patience --n_pretrain_epo=$n_pretrain_epo






exit





#--plot_synthetic_dist=$plot_synthetic_dist --ignore_plot_5=False --compile_mmd_mode='reduce-overhead' 


#python src/benchmarks_CGAN_joint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si  --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=$estop_patience --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=False --compile_mmd_mode='reduce-overhead' 



#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10






exit




estop_patience=10

python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic --n_trials=10 --estop_patience=$estop_patience









# python src/benchmarks_CGAN_joint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si   --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=$estop_patience --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=False --compile_mmd_mode='reduce-overhead' 



# #python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# estop_patience=10


# python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10 --estop_patience=$estop_patience




# tot_bsize=32




# d_n=real_bcancer_diagnosis_zscore



# use_single_si=True

# plot_synthetic_dist=False


# # 36


# for s_i in 44 74; do



# 	python src/benchmarks_CGAN_joint.py --d_n=$d_n --lr=$lr --s_i=$s_i --n_iterations=$n_iterations --use_single_si=$use_single_si   --precision=$precision --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --use_bernoulli=False --use_benchmark_generators=False --estop_patience=$estop_patience --plot_synthetic_dist=$plot_synthetic_dist --estop_mmd_type=$estop_mmd_type --ignore_plot_5=False --compile_mmd_mode='reduce-overhead'  --synthesise_w_cs=True

# 	exit


# 	#python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=0 --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=basic_disjoint --n_trials=10
# 	estop_patience=10


# 	python src/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --s_i=$s_i --d_n=$d_n --n_iterations=$n_iterations --lr=$lr --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --nsamps=$nsamps --algo_variant=gumbel --n_trials=10 --estop_patience=$estop_patience


# done







#dataset_real_sachs_raf_log