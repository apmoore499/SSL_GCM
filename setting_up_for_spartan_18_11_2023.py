





dataset='n36_gaussian_mixture_d6'



def create_cmd_fsup(s_i):
        out_string=f'python py/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --lab_bsize=4 --tot_bsize=32 --n_trials=10 --metric=val_acc --estop_patience=>
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 -->
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsiz>
        return(out_string)


with open('batch_fsup_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_fsup(s_i))


def create_cmd_psup(s_i):
        out_string=f'python py/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --lab_bsize=4 --tot_bsize=32 --n_trials=10 --metric=val_acc --estop_patienc>
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 -->
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsiz>
        return(out_string)


with open('batch_psup_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_psup(s_i))






def create_cmd_sslgan(s_i):
        out_string=f'python py/benchmarks_SSL_GAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --lab_bsize=4 --tot_bsize=32 --n_trials=10 --metric=val_acc --estop_patience=5 --min_epochs=10 --use_tuned_hpms=False\n'
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 --estop_mmd_type=val\n'
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)

with open('batch_sslgan_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_sslgan(s_i))

def create_cmd_sslvae(s_i):
        out_string=f'python py/benchmarks_SSL_VAE.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --lab_bsize=4 --tot_bsize=32 --n_trials=10 --metric=val_acc --estop_patience=5 --min_epochs=10 --use_tuned_hpms=False\n'
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 --estop_mmd_type=val\n'
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)

with open('batch_sslvae_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_sslvae(s_i))


def create_cmd_cgan(s_i):
        out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=False --use_benchmark_generators=False --estop_patience=5 --estop_mmd_type=trans\n'
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 --estop_mmd_type=val\n'
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)

with open('batch_cgan_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_cgan(s_i))


def create_cmd_cgan_classifier(s_i):
        out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=basic --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)


with open('batch_cgan_classifier_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_cgan_classifier(s_i))







def create_cmd_cgan_ybp(s_i):
        out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --estop_patience=5 --estop_mmd_type=trans\n'
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 --estop_mmd_type=val\n'
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)

with open('batch_cgan_ybp_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_cgan_ybp(s_i))



def create_cmd_cgan_gumbel(s_i):
        out_string=f'python py/benchmarks_CGAN_GUMBEL.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --min_temp=0.1 --init_temp=1.0 --n_trials=1 --val_loss_criterion=labelled_bce_and_all_feat_mmd_with_individual --lab_bsiz>
        return(out_string)

with open('batch_cgan_gumbel_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                print(s_i)
                print(str(s_i))
                f.write(create_cmd_cgan_gumbel(s_i))



def create_cmd_cgan_gumbel_classifier(s_i):
        out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=gumbel --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)


with open('batch_cgan_gumbel_classifier_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_cgan_gumbel_classifier(s_i))













def create_cmd_tgan(s_i):
        out_string=f'python py/benchmarks_TRIPLE_GAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --lab_bsize=4 --tot_bsize=32 --n_trials=10 --metric=val_acc --estop_patience=5 --min_epochs=10 --use_tuned_hpms=False\n'
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 --estop_mmd_type=val\n'
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)

with open('batch_tgan_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_tgan(s_i))


def create_cmd_emin(s_i):
        out_string=f'python py/benchmarks_ENTROPY_MINIMISATION.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --lab_bsize=4 --tot_bsize=32 --n_trials=10 --metric=val_acc --estop_patience=5 --min_epochs=10 --use_tuned_hpms=>
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 --estop_mmd_type=val\n'
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)

with open('batch_emin_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_emin(s_i))





def create_cmd_vat(s_i):
        out_string=f'python py/benchmarks_VAT.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --lab_bsize=4 --tot_bsize=32 --n_trials=10 --metric=val_acc --estop_patience=5 --min_epochs=10 --use_tuned_hpms=False\n'
        #out_string=f'python py/benchmarks_CGAN.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --use_single_si=True --tot_bsize=32 --lab_bsize=4 --use_bernoulli=True --use_benchmark_generators=False --patience=10 --estop_mmd_type=val\n'
        #out_string=f'python py/benchmarks_CGAN_SUPERVISED_CLASSIFIER.py --d_n={dataset} --s_i={s_i} --n_iterations=100 --lr=1e-2 --nsamps=300 --algo_variant=y_given_bp --use_single_si=True --n_trials=10 --lab_bsize=4 --tot_bsize=32\n'
        return(out_string)

with open('batch_vat_{0}.sh'.format(dataset),'w') as f:
        for s_i in range(100):
                f.write(create_cmd_vat(s_i))


benchmark_cmd=[

'source batch_sslvae.sh\n',
'source batch_sslgan.sh\n',
'source batch_emin.sh\n',
'source batch_tgan.sh\n',
]
with open('batch_bmrks_{0}.sh'.format(dataset),'w') as f:
        for cmd_line in benchmark_cmd:
                f.write(cmd_line)



# in file on spartan

/apps/examples/Python/virtualenv.md





#ssvae

# SPARTAN NOODLING 18_11_2023



rsync -abviuzPI /media/krillman/240GB_DATA/codes2/SSL_GCM/lightning_logs/combined_spec.xls apmoore@spartan.hpc.unimelb.edu.au::/data/projects/punim1573/amoore/causal_ssl_gan/combined_spec.xls 




## Load a python module and check that virtualenv is available
```
$ module load GCCcore/11.3.0 
$ module load Python/3.10.4
$ which virtualenv
/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.10.4/bin/virtualenv
$ virtualenv --version
virtualenv 20.14.1 from /apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.10.4/lib/python3.10/site-packages/virtualenv/__init__.py
```









module load Python/3.7.1-GCC-6.2.0;
module load web_proxy;
source ~/pytorch_env_for_jupyter/bin/activate;






module load Python/3.10.4;
module load web_proxy;
source ~/pytorch_env_for_jupyter/bin/activate;


virtualenv ~/venvs/venv-3.10.4



module load GCCcore/11.3.0 
module load Python/3.10.4

sinteractive -p gpu-a100-short --time=01:00:00 --cpus-per-task=1 --gres=gpu:1


source ~/venvs/venv-3.10.4/bin/activate

module load GCCcore/11.3.0 
module load Python/3.10.4
module load CUDA/12.2.0


pip install google-api-python-client


pip install coloraide 





GNU nano 5.6.1          PARTIAL_SUPERVISED_dn_n36_gaussian_mixture_d7_10000_si_0.out                     
Lmod has detected the following error: The following module(s) are unknown:
"GCCore/11.3.0"

Please check the spelling or version number. Also try "module spider ..."
It is also possible your cache file is out-of-date; it may help to try:
  $ module --ignore_cache load "GCCore/11.3.0"

Also make sure that all modulefiles written in TCL start with the string
#%Module



Traceback (most recent call last):
srun: error: spartan-gpgpu128: task 0: Exited with exit code 1
  File "/data/gpfs/projects/punim1573/amoore/causal_ssl_gan/google_api/log_json_job.py", line 3, in <modul>
    from googleapiclient.http import MediaFileUpload
ModuleNotFoundError: No module named 'googleapiclient'
Traceback (most recent call last):
  File "/data/gpfs/projects/punim1573/amoore/causal_ssl_gan/py/benchmarks_PARTIAL_SUPERVISED_CLASSIFIER.py>
    from benchmarks_utils import *
  File "/data/gpfs/projects/punim1573/amoore/causal_ssl_gan/py/benchmarks_utils.py", line 22, in <module>
    from coloraide import Color
ModuleNotFoundError: No module named 'coloraide'
srun: error: spartan-gpgpu128: task 0: Exited with exit code 1





pip3 install torch torchvision torchaudio







nano slurm/PARTIAL_SUPERVISED_dn_n36_gaussian_mixture_d7_10000.slurm



home_dir="/data/projects/punim1573/amoore/causal_ssl_gan/"
cd $home_dir
module load GCCore/11.3.0
module load Python/3.10.4
source ~/venvs/venv-3.10.4/bin/activate;
module load CUDA/12.2.0

if [ "$SLURM_ARRAY_TASK_ID" == "$min_si" ]; then
   srun python google_api/log_json_job.py --model_name=$model_name --d_n=$d_n --status=start --job_id=$SLURM_JOB_ID
fi
srun 





wait
if [ "$SLURM_ARRAY_TASK_ID" == "$max_si" ]; then
   srun python google_api/log_json_job.py --model_name=$model_name --d_n=$d_n --status=finish --job_id=$SLURM_JOB_ID
fi











local env libs


dict_keys(['CGAN_BASIC', 'CGAN_BASIC_DJ', 'CGAN_GUMBEL', 'CGAN_GUMBEL_DJ', 'CGAN_GUMBEL_DJ_XCES', 
        'CGAN_MARGINAL', 'CGAN_MMD_GAN', 'FULLY_SUPERVISED', 'PARTIAL_SUPERVISED', 'ENTROPY_MINIMISATION', 
        'TRIPLE_GAN', 'SSL_GAN', 'VAT', 'SSVAE', 'LABEL_PROPAGATION', 'CGAN_BASIC_CLASSIFIER', 'CGAN_BASIC_DJ_CLASSIFIER', 
        'CGAN_GUMBEL_CLASSIFIER', 'CGAN_MARGINAL_CLASSIFIER', 'CGAN_MMD_GAN_CLASSIFIER', 'CGAN_GUMBEL_DJ_CLASSIFIER', 'CGAN_GUMBEL_DJ_XCES_CLASSIFIER'])


slurm/SSL_GAN_dn_n36_gaussian_mixture_d3_10000_serial_batch_all_si.slurm


aiohttp                                                                                    
aiosignal               
asttokens               
async-timeout           
attrs                   
cdt                     
certifi                 
charset-normalizer      
contourpy               
cycler                  
decorator               
exceptiongroup          
executing               
filelock                
fonttools               
frozenlist              
fsspec                  
GPUtil                  
idna                    
igraph                  
importlib-resources     
intel-openmp            
ipython                 
jedi                    
Jinja2                  
joblib                  
kaleido                 
kiwisolver              
lightning-utilities     
MarkupSafe              
matplotlib              
matplotlib-inline       
mkl                     
mpmath                  
multidict               
networkx                
numpy                        
packaging               
pandas                  
parso                   
patsy                   
pexpect                 
Pillow                  
plotly                  
prompt-toolkit          
protobuf                
ptyprocess              
pure-eval               
pycairo                 
Pygments                
pyparsing               
python-dateutil         
pytorch-lightning       
pytz                    
PyYAML                  
requests                
scikit-learn            
scipy                   
six                     
skrebate                
stack-data              
statsmodels             
sympy                   
tbb                     
tenacity                
tensorboardX            
texttable               
threadpoolctl           
torch                   
torchmetrics            
tqdm                    
traitlets               
triton                  
typing_extensions       
tzdata                  
urllib3                 
wcwidth                 
xlrd                    
yarl                    
zipp                    












module load PyTorch/1.12.1-CUDA-11.7.0




sinteractive -p gpu-a100-short --time=01:00:00 --cpus-per-task=1 --gres=gpu:1




home_dir="/data/projects/punim1573/amoore/causal_ssl_gan/"
cd $home_dir


module load GCCore/11.3.0
module load Python/3.10.4

source ~/venvs/venv-3.10.4/bin/activate;
        

module load CUDA/12.2.0












source ~/venvs/venv-3.10.4/bin/activate










use_single_si=True
n_iterations=100
lr=1e-3
lab_bsize=4
tot_bsize=32
n_trials=10
metric=val_acc
estop_patience=10
use_tuned_hpms=False
min_epochs=10
d_n=n36_gaussian_mixture_d7  #_10000
model_name=PARTIAL_SUPERVISED
min_si=0
max_si=99


tot_bsize=256



SLURM_ARRAY_TASK_ID=1


rsync -abviuzPI /media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_n36_gaussian_mixture_d7_100000 apmoore@spartan.hpc.unimelb.edu.au:/data/projects/punim1573/amoore/causal_ssl_gan/data/dataset_n36_gaussian_mixture_d7_100000

d_n=n36_gaussian_mixture_d7_100000





```
unlabel prediction acc: 
Traceback (most recent call last):
  File "/data/gpfs/projects/punim1573/amoore/causal_ssl_gan/py/benchmarks_FULLY_SUPERVISED_CLASSIFIER.py", line 322, in <module>
    print((optimal_model.predict(orig_data['unlabel_features']) == orig_data['unlabel_y'].argmax(1).numpy()).mean())
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
(venv-3.10.4) [apmoore@spartan-gpgpu128 causal_ssl_gan]$ 
```














FileNotFoundError: [Errno 2] No such file or directory: './data/dataset_n36_gaussian_mixture_d7_10000/d_n_n36_gaussian_mixture_d7_10000_s_i_1_label_features.pt'
