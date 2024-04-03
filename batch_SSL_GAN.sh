#!/bin/bash



conda activate ssl_gcm
conda deactivate

conda activate ssl_gcm



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


# d_n=n36_gaussian_mixture_d1

# python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


# d_n=n36_gaussian_mixture_d2
# python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 

# d_n=n36_gaussian_mixture_d3
# python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 

d_n=n36_gaussian_mixture_d4
python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 

d_n=n36_gaussian_mixture_d5
python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 

d_n=n36_gaussian_mixture_d6
python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 

d_n=n36_gaussian_mixture_d7

python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


d_n=dataset_real_bcancer_diagnosis_zscore
python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


d_n=dataset_real_sachs_mek_log
python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


d_n=dataset_real_sachs_raf_log
python src/benchmarks_SSL_GAN.py --d_n=$d_n --s_i=$s_i --n_trials=$n_trials --lr=$lr --precision=$precision --n_iterations=$n_iterations --use_single_si=$use_single_si --tot_bsize=$tot_bsize --lab_bsize=$lab_bsize --metric=val_acc --use_tuned_hpms=$use_tuned_hpms --estop_patience=$estop_patience --min_epochs=$min_epochs 


conda deactivate


#train transformer lmks model


cd /media/krillman/1TB_DATA/codes/eg3d_rlhf/eg3d/RLHF_nbooks/RLHF_Codebase/src_rlhf



conda activate pointface_env
python train_rwd_model.py experiment=aw98_3d_lmks_transformer.yaml         
conda deactivate


#run rest of inversions ok

conda activate pointface_env

cd /media/krillman/1TB_DATA/codes/HFGI3D/inversion/scripts

python run_pti.py
python run_pti.py
python run_pti.py
python run_pti.py
python run_pti.py
python run_pti.py
python run_pti.py
python run_pti.py


conda deactivate