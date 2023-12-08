











#eq23...
import math

import numpy as np

    
import scipy
import tqdm

import os

import time


import glob


median_x2=3.57544
median_x1=1.6569125



import torch


dtype='n36_gaussian_mixture_d3_10000'


si='0'

U_X_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{dtype}/d_n_{dtype}_s_i_{si}_unlabel_features.pt'
U_Y_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{dtype}/d_n_{dtype}_s_i_{si}_unlabel_y.pt'


L_X_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{dtype}/d_n_{dtype}_s_i_{si}_label_features.pt'
L_Y_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{dtype}/d_n_{dtype}_s_i_{si}_label_y.pt'


V_X_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{dtype}/d_n_{dtype}_s_i_{si}_val_features.pt'
V_Y_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{dtype}/d_n_{dtype}_s_i_{si}_val_y.pt'



#U_Y_fn='/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_n36_gaussian_mixture_d3_10000/d_n_n36_gaussian_mixture_d3_10000_s_i_0_unlabel_y.pt'
#U_Y_gt=torch.load(U_Y_fn)

fns_d3={'U_X_fn':U_X_fn,
        'U_Y_fn':U_Y_fn,
        'L_X_fn':L_X_fn,
        'L_Y_fn':L_Y_fn,
        'V_X_fn':V_X_fn,
        'V_Y_fn':V_Y_fn}



for k in fns_d3.keys():
    fns_d3[k]=torch.load(fns_d3[k])



#split out....

U_Y=fns_d3['U_Y_fn']
U_X=fns_d3['U_X_fn']

L_X=fns_d3['L_X_fn']
L_Y=fns_d3['L_Y_fn']

V_X=fns_d3['V_X_fn']
V_Y=fns_d3['V_Y_fn']


first=[0,1]
second=[2,3]

third=[4,5]

X1_idx=second
X2_idx=first


#initialisging....
U_X1=U_X[:,X1_idx]
print(U_X1.shape)
U_X2=U_X[:,X2_idx]
print(U_X2.shape)


L_X1=L_X[:,X1_idx]
print(L_X1.shape)
L_X2=L_X[:,X2_idx]
print(L_X2.shape)



V_X1=V_X[:,X1_idx]
print(V_X1.shape)
V_X2=V_X[:,X2_idx]
print(V_X2.shape)





N_lab=L_X1.shape[0]
N_ulab=U_X1.shape[0]
N_val=V_X1.shape[0]


V_Y_gt=V_Y
U_Y_gt=U_Y

print('n lab ulab val')
print(f'{N_lab}\t {N_ulab}\t {N_val}')







X1_file=torch.cat((L_X1,U_X1,V_X1),dim=0)
X2_file=torch.cat((L_X2,U_X2,V_X2),dim=0)



#do for task I2T, ie X1 -> T -> X2



#construct for R-th modality the KNN graph!!!!
device_str='cuda'


#initalising Laplacian matrix for q'th modality
def compute_L_q_matrix(X,N=None,n_neighbours=10,device_str='cuda',median=None):
    



    Xi=X
    
    

    import scipy
    import sklearn



    if median==None:

        pwd=scipy.spatial.distance.cdist(X.cpu().numpy(), X1_file.cpu().numpy(), metric='euclidean')

        I_mat=np.eye(pwd.shape[0])
        idx_m=np.ones_like(pwd)
        idx_m=idx_m-I_mat

        idm=idx_m.astype(np.bool_)


        mdist=np.median(pwd[idm])

    else:
        mdist=median
        

        
    


    if N is not None:
        Xi=X[:N,:]

    Xi=Xi.to(torch.device(device_str))
    N=Xi.shape[0]


    orders=torch.argsort(torch.linalg.norm(Xi.unsqueeze(0).repeat(N,1,1)-Xi.unsqueeze(1).repeat(1,N,1),2,2),dim=1,descending=False)




    neighbour_idx=orders[:,1:n_neighbours+1]


    ni=neighbour_idx.unsqueeze(2)



    idx_cat=torch.arange(ni.shape[0],device=torch.device(device_str))[:, None, None].expand(-1,n_neighbours,1)
    idx_for_rep=torch.cat((idx_cat,ni),dim=-1)




    #sparse coo initialisation for speed
    multi_indices=idx_for_rep.flatten(0).reshape(-1,2)
    values = torch.ones(multi_indices.shape[0],device=torch.device(device_str))

    # Convert to tensor
    indices_tensor = multi_indices.t()#torch.tensor(multi_indices.clone().detach()).t()  # Transpose to get 2 x N shape
    values_tensor = values#torch.tensor(values)

    # Size of the sparse matrix
    size = (Xi.shape[0], Xi.shape[0])  # Assuming a 3x3 matrix

    # Create a sparse tensor
    sparse_matrix = torch.sparse_coo_tensor(indices_tensor, values_tensor, size).clone().detach()
    smc=sparse_matrix.to_dense()
    x_row=smc.clone()[:,:,None].expand(-1,-1,2)*Xi.unsqueeze(1).expand(-1,N,-1)
    neighbouring_points=smc[:,:,None].expand(-1,-1,2)*Xi
    diff=x_row-neighbouring_points
    weights=torch.linalg.norm(diff,ord=2,dim=2,keepdim=True)**2
    SIGMA=mdist

    weights=weights/2*(SIGMA**2)
    weights=weights[:,:,-1]#.shape

    diag_terms=torch.sum(weights,dim=1,keepdim=True)
    D_q=torch.eye(N,device=torch.device(device_str))*diag_terms
    LP_mat=torch.linalg.inv(D_q) @ ( D_q - weights) @ torch.linalg.inv(D_q) 


    return(LP_mat)





def get_accuracy_U(ulab_y_k,U_Y_gt,N_ulab):
    #compile results...
    ul_final=torch.from_numpy(ulab_y_k[:N_ulab,:])
    sm=torch.nn.functional.softmax(ul_final,dim=1)
    pred_labels_u=torch.argmax(sm,1)
    pred_oh=torch.nn.functional.one_hot(pred_labels_u)


    correct = torch.sum(pred_oh[:,1]==U_Y_gt[:,1]) #55pc acc)
    return(correct/ul_final.shape[0])



def get_accuracy_V(ulab_y_k,V_Y_gt,N_ulab):
    #compile results...
    ul_final=torch.from_numpy(ulab_y_k[N_ulab:,:])
    sm=torch.nn.functional.softmax(ul_final,dim=1)
    pred_labels_u=torch.argmax(sm,1)
    pred_oh=torch.nn.functional.one_hot(pred_labels_u)


    correct = torch.sum(pred_oh[:,1]==V_Y_gt[:,1]) #55pc acc)
    return(correct/ul_final.shape[0])




# output after ctrl-c


'''

^C^C^C^C^C^C^C^C^C^CTraceback (most recent call last):
  File "/media/krillman/240GB_DATA/codes2/SSL_GCM/src/meth1_py_ray.py", line 513, in <module>
    tune.with_resources(
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/tune/tuner.py", line 347, in fit
    return self._local_tuner.fit()
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/tune/impl/tuner_internal.py", line 588, in fit
    analysis = self._fit_internal(trainable, param_space)
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/tune/impl/tuner_internal.py", line 703, in _fit_internal
    analysis = run(
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/tune/tune.py", line 1133, in run
    runner.cleanup()
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/tune/execution/trial_runner.py", line 1143, in cleanup
    self._cleanup_trials()
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/tune/execution/tune_controller.py", line 435, in _cleanup_trials
    self._actor_manager.next(timeout=1)
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/air/execution/_internal/actor_manager.py", line 214, in next
    ready, _ = ray.wait(all_futures, num_returns=1, timeout=timeout)
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/_private/auto_init_hook.py", line 24, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/media/krillman/1TB_DATA/mc3_envs_2/ssl_gcm/lib/python3.9/site-packages/ray/_private/worker.py", line 2701, in wait
    ready_ids, remaining_ids = worker.core_worker.wait(
  File "python/ray/_raylet.pyx", line 2998, in ray._raylet.CoreWorker.wait
  File "python/ray/_raylet.pyx", line 400, in ray._raylet.check_status
KeyboardInterrupt

'''


def train_SFSM(config,data):
    
    # if freezing maybe have to coall this to freee up gpu memory beforenext training run!
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.utils.wait_for_gpu.html
    tune.utils.util.wait_for_gpu(target_util=0.8) #5pc utilisation
    
    
    # extra (unhelpful / haven't looked at them closely)
    
    # https://github.com/ray-project/ray/issues/14156
    # https://github.com/ray-project/ray/issues/17359
    # https://discuss.ray.io/t/ray-train-hangs-for-long-time/6333/10
    
    L_Y=data['L_Y']
    #L_X=data['L_X']
    #X1=data['X1']
    #X2=data['X2']
    #U_X=data['U_X']
    L_Y=data['L_Y']
    V_Y_gt=data['V_Y_gt']
    U_Y_gt=data['U_Y_gt']
    N_ulab=data['N_ulab']
    N_lab=data['N_lab']
    
    X1_file=data['X1_file']
    X2_file=data['X2_file']
    
    gamma=config['gamma']
    lambda_1=config['lambda_1']
    lambda_2=config['lambda_2']
    beta=config['beta']
    nn=config['nn']
    eps=config['eps']
    
    n_iterations=100

    print(gamma)
    print(lambda_1)

    #get data
    ulab_accs=[]
    val_accs=[]
    st=time.time()
    max_val_acc=0.0
                        
    L_k=compute_L_q_matrix(X1_file,n_neighbours=nn,device_str=device_str,median=median_x2)
    L_ll,L_lu,L_ul,L_uu = L_k[:N_lab,:N_lab],   L_k[:N_lab,N_lab:], L_k[N_lab:,:N_lab], L_k[N_lab:,N_lab:]
    Y_hat=-torch.linalg.inv(L_uu) @ L_ul @ L_Y.to(torch.device(device_str)).float()
    X1=X1_file.transpose(1,0).to(torch.device(device_str))
    X2=X2_file.transpose(1,0).to(torch.device(device_str))

    X1_dim=X1.shape[0]
    X2_dim=X2.shape[0]


    assert n_class==X1_dim,'error not square for init U matrices!'
    assert n_class==X2_dim,'error not square for init U matrices!'

    U_11=torch.eye(X1_dim,device=torch.device(device_str))
    U_12=torch.eye(X2_dim,device=torch.device(device_str))

    Y=torch.cat((L_Y.to(torch.device(device_str)),Y_hat),dim=0)


    R11=(torch.eye(2)*(1/2*math.sqrt(1+eps))).to(torch.device(device_str))
    R22=R11

    best_ulab=0.0
    best_val=0.0
    
    for kkk in tqdm.tqdm(range(n_iterations)):
        #EQ 23

        first_term=torch.linalg.inv(X1 @ X1.transpose(1,0)+gamma * X1 @ L_k @ X1.transpose(1,0)+lambda_1 * R11)
        second_term=beta*X1 @ Y + (1-beta)*X1 @ X2.transpose(1,0) @ U_12
        update_U_11= first_term @ second_term
        update_U_11
        
        U_11=update_U_11


        #EQ 24
        first_term=torch.linalg.inv((1-beta)*X2 @ X2.transpose(1,0) + lambda_2*R22)
        second_term = (1-beta)*X2 @ X1.transpose(1,0) @ U_11
        update_U_12=first_term @ second_term
        update_U_12

        U_12=update_U_12

        #EQ 27 #this can double as prediction step
        L_Y=L_Y.to(torch.device(device_str)).float()
        first_term=torch.linalg.inv(beta-gamma*L_uu)
        second_term=X1[:,N_lab:].transpose(1,0) @ U_11 + gamma * L_ul @ L_Y
        U_Y_hat=first_term @ second_term
        
        U_Y_f=U_Y_hat[1].cpu().mean()
        
        
        has_nan=False
        if torch.isnan(U_Y_f):
            #tune.report({'U_Y_f':U_Y_f.numpy()})
            
            has_nan=True
            tune.report({'val_acc':val_acc,'ulab_acc':ulab_acc,'U_Y_f':U_Y_f.numpy(),'best_ulab':best_ulab,'best_val':best_val,'has_nan':has_nan})

        
        #has_nan=False
        #if any(torch.isnan(U_Y_hat.flatten())):
            #has_nan=True
            #tune.report(has_nan=has_nan)
            

        Y=torch.cat((L_Y,U_Y_hat),0)
        
        #ulab_y.append(U_Y_hat.detach().cpu().numpy())

        val_acc=get_accuracy_V(U_Y_hat.detach().cpu().numpy(),V_Y_gt=V_Y_gt,N_ulab=N_ulab).item()
        ulab_acc=get_accuracy_U(U_Y_hat.detach().cpu().numpy(),U_Y_gt=U_Y_gt,N_ulab=N_ulab).item()

        ulab_accs.append(ulab_acc)
        val_accs.append(val_acc)
        
        if val_acc>max_val_acc:
            max_val_acc=val_acc
            best_ulab=ulab_acc
            best_val=val_acc
        
       
        
        
        tune.report({'val_acc':val_acc,'ulab_acc':ulab_acc,'U_Y_f':U_Y_f.numpy(),'best_ulab':best_ulab,'best_val':best_val,'has_nan':has_nan})
        
        


import torch
import torch.optim as optim
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet#,# train, test
import ray
import math


from ray import air
from ray.tune import Tuner
from ray import train#, tune

#https://github.com/ray-project/ray/issues/8671 custom stopper on nann...

# analysis = tune.run(
#     train_SFSM,
#     resources_per_trial={'gpu': 1,'cpu':1},
#     max_concurrent_trials=1,
#     resume=None,
#     config=rdict)



import time
from ray import train, tune
from ray.tune import Stopper

    
import numpy as np
from ray import train, tune
from ray.tune.stopper import CombinedStopper


class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        self._deadline = 60  # Stop all trials after 60 seconds
        
    def __call__(self, trial_id, result):
        return False
    
    def stop_all(self):
        return time.time() - self._start > self._deadline



if __name__=='__main__':





 
    ray.init(num_cpus=4,num_gpus=1)
    #ray.init()

    print(ray.cluster_resources())

    # ray.init(
    #     _memory=MEMORY_LIMIT_BYTES,
    #     num_cpus=joblib.cpu_count(),
    #     num_gpus=N_GPU,
    #     _temp_dir=os.environ["TMPDIR"],
    #     _system_config={
    #         "automatic_object_spilling_enabled": True,
    #         "object_spilling_config": json.dumps(
    #             {"type": "filesystem", "params": {"directory_path": SPILL_DIR}},
    #         )
    #     },
    # )
    #Logical resource usage: 1.0/0 CPUs, 1.0/0 GPUs


    # rdict={'gamma':tune.grid_search([0.001, 0.01, 0.1]),
    #         'lambda_1':tune.grid_search([0.001, 0.1, 0.1]),
    #         'lambda_2':tune.grid_search([0.001, 0.01, 0.1]),
    #         'beta':tune.grid_search([0.001, 0.01, 0.1]),
    #         'eps':tune.grid_search([0.001, 0.01, 0.1]),
    #         'nn':tune.randint(5, 200)}
            #np.linspace(start=0.01,end=10.0,num=10)

    # rdict={'gamma':tune.grid_search([0.001, 10, 0.1]),
    #         'lambda_1':tune.grid_search([0.1, 10, 0.5]),
    #         'lambda_2':tune.grid_search([0.1, 10, 0.5]),
    #         'beta':tune.grid_search([0.001, 10, 0.1]),
    #         'eps':tune.choice([0.001, 0.01, 0.1]),
    #         'nn':tune.choice([5,10,50,100,150,200])}

    rdict={'gamma':tune.loguniform(1e-3,10),
            'lambda_1':tune.loguniform(1e-1,10),
            'lambda_2':tune.loguniform(1e-1,10),
            'beta':tune.loguniform(1e-3,10),
            'eps':tune.choice([0.001, 0.01, 0.1]),
            'nn':tune.choice([5,10,50,100,150,200])}



tune.choice




    # from ray.tune.search.optuna import OptunaSearch
    # import optuna

    # space = {
    #     "gamma": optuna.distributions.LogUniformDistribution(1e-4,1e2),
    #     "lambda_1": optuna.distributions.LogUniformDistribution(1e-4,1e2),
    #     "lambda_2": optuna.distributions.LogUniformDistribution(1e-4,1e2),
    #     "beta": optuna.distributions.LogUniformDistribution(1e-4,1e2),
    #     'eps':optuna.distributions.CategoricalDistribution([1.0,1e-3,1e-6]),
    #     'nn':optuna.distributions.CategoricalDistribution([5,10,50,100,150,200]),
    # }
    # optuna_search = OptunaSearch(
    #     space,
    #     metric="val_acc",
    #     mode="max")


    n_class=2

    data={'L_Y':L_Y,
        'V_Y_gt':V_Y_gt,
        'U_Y_gt':U_Y_gt,
        'N_ulab':N_ulab,
        'N_lab':N_lab,
        'n_class':n_class,
        'X1_file':X1_file,
        'X2_file':X2_file}


    def stopnanloss(trial_id, result):
        has_nan=math.isnan(result['_metric']["U_Y_f"])
        
        if has_nan:
            print('we got nan!')
        return has_nan


    stopper = CombinedStopper(stopnanloss,TimeStopper(),)


    #mabye u want to combine stopnanloss with a timed stopper if its taking too long, ie
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.stopper.CombinedStopper.html
    
    #combined stopper was clunky, try simpler here:
    # https://github.com/ray-project/ray/issues/31849
    tuner = Tuner(
        tune.with_resources(
            tune.with_parameters(train_SFSM, data=data),
            resources={"cpu": 4, "gpu": 1}
        ),
        #run_config=air.RunConfig(stop=stopper),
        run_config=air.RunConfig(stop={"time_total_s": 100}),#,"_metric/has_nan":True}),  # 100 seconds)
        param_space=rdict,
        tune_config=tune.TuneConfig(num_samples=100,max_concurrent_trials=1,reuse_actors=True),#,search_alg=optuna_search)#,metric='best_val',mode='max')

    )
#ray.tune.stopper.TimeoutStopper()

    results = tuner.fit()


    print('pausing here')




    #results.get_best_result(metric='_metric/val_acc',mode='max').metrics
    
    
    # #code to load old results
    
    # from ray.tune import Tuner

        
    # exp_path='/home/krillman/ray_results/train_SFSM_2023-11-29_12-41-55'
    # trainable=train_SFSM


    # # `trainable` is what was passed in to the original `Tuner`
    # tuner = Tuner.restore(exp_path, trainable=trainable)
    # results = tuner.get_results()
    
    # best_config=results.get_best_result(metric='_metric/best_val',mode='max').config
    
    # best_res=results.get_best_result(metric='_metric/best_val',mode='max').results
    
    
    
    


