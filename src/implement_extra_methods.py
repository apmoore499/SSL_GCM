#eq23...
import math

import numpy as np

    
import scipy
import tqdm

import os

import time


import glob

import torch
torch.set_float32_matmul_precision('high') #try with 4090

#do for task I2T, ie X1 -> T -> X2



#construct for R-th modality the KNN graph!!!!
device_str='cuda'


#get pairwise distances using torch...


def compute_median_pwd_torch(X):
        

    X=X.to(torch.device(device_str))
    pairwise_distances=torch.cdist(X,X)
    I_mat=torch.eye(X.shape[0],device=torch.device(device_str))
    idx_m=torch.ones_like(pairwise_distances)
    idx_m=idx_m-I_mat
    mdist=torch.median(pairwise_distances[idx_m.to(torch.bool)])
    
    return(mdist)

#initalising Laplacian matrix for q'th modality
def compute_L_q_matrix(X,N=None,n_neighbours=10,device_str='cuda',median=None):
    



    Xi=X
    
    

    import scipy
    import sklearn



    if median==None:
        
        
        assert False, 'error need median precomputed'

        # pwd=scipy.spatial.distance.cdist(X.cpu().numpy(), X1_file.cpu().numpy(), metric='euclidean')

        # I_mat=np.eye(pwd.shape[0])
        # idx_m=np.ones_like(pwd)
        # idx_m=idx_m-I_mat

        # idm=idx_m.astype(np.bool_)


        # mdist=np.median(pwd[idm])

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

    with torch.no_grad():
    # Create a sparse tensor
        sparse_matrix = torch.sparse_coo_tensor(indices_tensor, values_tensor, size,device=torch.device(device_str))#.clone().detach()
        smc=sparse_matrix.to_dense()
        x_row=smc.clone()[:,:,None].expand(-1,-1,Xi.shape[1])*Xi.unsqueeze(1).expand(-1,N,-1)
        neighbouring_points=smc[:,:,None].expand(-1,-1,Xi.shape[1])*Xi
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
    pred_oh=torch.nn.functional.one_hot(pred_labels_u,num_classes=2)


    correct = torch.sum(pred_oh[:,1]==V_Y_gt[:,1]) #55pc acc)
    return(correct/ul_final.shape[0])



n_class=2



def train_SFSM(config,dd,median_x_val):
    
    
    gamma=config['gamma']
    lambda_1=config['lambda_1']
    lambda_2=config['lambda_2']
    beta=config['beta']
    nn=config['nn']
    eps=config['eps']
    n_iterations=config['n_iterations']
    
    
    X1_file=dd['X1_file']
    X2_file=dd['X2_file']
    L_Y=dd['L_Y']
    V_Y_gt=dd['V_Y_gt']
    U_Y_gt=dd['U_Y_gt']
    N_lab=dd['N_lab']
    N_ulab=dd['N_ulab']
    

    median_x2=median_x_val

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


    #assert n_class==X1_dim,'error not square for init U matrices!'
    #assert n_class==X2_dim,'error not square for init U matrices!'

    U_11=torch.eye(X1_dim,n_class,device=torch.device(device_str))
    U_12=torch.eye(X2_dim,n_class,device=torch.device(device_str)) #should be able to use wtihout identity?

    #print(U_11)
    #print(U_12)

    Y=torch.cat((L_Y.to(torch.device(device_str)),Y_hat),dim=0)


    R11=(torch.eye(X1_dim)*(1/2*math.sqrt(1+eps))).to(torch.device(device_str))
    R22=(torch.eye(X2_dim)*(1/2*math.sqrt(1+eps))).to(torch.device(device_str))
    #R22=R11


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
        
        
        if any(torch.isnan(U_Y_hat.flatten())):
            break

        Y=torch.cat((L_Y,U_Y_hat),0)
        
        #ulab_y.append(U_Y_hat.detach().cpu().numpy())

        val_acc=get_accuracy_V(U_Y_hat.detach().cpu().numpy(),V_Y_gt=V_Y_gt,N_ulab=N_ulab)
        ulab_acc=get_accuracy_U(U_Y_hat.detach().cpu().numpy(),U_Y_gt=U_Y_gt,N_ulab=N_ulab)

        #if val_acc>max_val_acc:
        #    max_val_acc=val_acc
        #    max_val_ulab_acc=get_accuracy_U(U_Y_hat.detach().cpu().numpy(),U_Y_gt=U_Y_gt,N_ulab=N_ulab)

        #ulab_accs.append(c)
        
        #val_accs.append(val_acc)
        
        ulab_accs.append(ulab_acc.item())
        
        val_accs.append(val_acc.item())
        
        #tune.track.log(val_acc=val_acc)
        #tune.track.log(ulab_acc=ulab_acc)
        
        
    #get max index, return that...
    max_v_idx=np.argsort(val_accs)
    
    max_ulab=ulab_accs[max_v_idx[-1]]
    max_val=val_accs[max_v_idx[-1]]
    
        
    return dict(val_accs=val_accs,ulab_accs=ulab_accs,max_ulab=max_ulab),max_ulab,max_val



import torch


def return_data_for_sfsm(dtype,si,X1_idx,X2_idx):




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


    
    #set_trace()

    for k in fns_d3.keys():
        fns_d3[k]=torch.load(fns_d3[k])

    tl='/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_real_sachs_raf_log/saved_models/d_n_real_sachs_raf_log_s_i_0_unlabel_features.pt'

    #split out....

    U_Y=fns_d3['U_Y_fn']
    U_X=fns_d3['U_X_fn']

    L_X=fns_d3['L_X_fn']
    L_Y=fns_d3['L_Y_fn']

    V_X=fns_d3['V_X_fn']
    V_Y=fns_d3['V_Y_fn']


    #first=[0,1]
    #second=[2,3]
    #third=[4,5]

    #X1_idx=second
    #X2_idx=first


    #initialisging....
    U_X1=U_X[:,X1_idx]
    #print(U_X1.shape)
    U_X2=U_X[:,X2_idx]
    #print(U_X2.shape)


    L_X1=L_X[:,X1_idx]
    #print(L_X1.shape)
    L_X2=L_X[:,X2_idx]
    #print(L_X2.shape)



    V_X1=V_X[:,X1_idx]
    #print(V_X1.shape)
    V_X2=V_X[:,X2_idx]
    #print(V_X2.shape)





    N_lab=L_X1.shape[0]
    N_ulab=U_X1.shape[0]
    N_val=V_X1.shape[0]


    V_Y_gt=V_Y
    U_Y_gt=U_Y



    print('n lab ulab val')

    print(f'{N_lab}\t {N_ulab}\t {N_val}')



    X1_file=torch.cat((L_X1,U_X1,V_X1),dim=0)
    X2_file=torch.cat((L_X2,U_X2,V_X2),dim=0)



    return(dict(L_Y=L_Y,X1_file=X1_file,X2_file=X2_file,V_Y_gt=V_Y_gt,U_Y_gt=U_Y_gt,N_lab=N_lab,N_ulab=N_ulab))

def return_data_for_second_method(dtype,si):


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
        fns_d3[k]=torch.load(fns_d3[k],map_location=torch.device('cuda'))



    #split out....

    U_Y=fns_d3['U_Y_fn']
    U_X=fns_d3['U_X_fn']

    L_X=fns_d3['L_X_fn']
    L_Y=fns_d3['L_Y_fn']

    V_X=fns_d3['V_X_fn']
    V_Y=fns_d3['V_Y_fn']






    N_lab=L_X.shape[0]
    N_ulab=U_X.shape[0]
    N_val=V_X.shape[0]


    V_Y_gt=V_Y
    U_Y_gt=U_Y



    print('n lab ulab val')

    print(f'{N_lab}\t {N_ulab}\t {N_val}')
    
    X_cat=torch.cat((L_X,U_X,V_X),dim=0)




    return(dict(L_Y=L_Y,X_cat=X_cat,V_Y_gt=V_Y_gt,U_Y_gt=U_Y_gt,N_lab=N_lab,N_ulab=N_ulab))



import scipy
import sklearn
import pandas as pd

def get_Hd(k,device=torch.device('cuda')):
    
    d=k
    I_d=torch.ones(d,device=device)[:,None]

    H_d=torch.eye(d,device=device) - (1/d)*I_d.clone() @ I_d.clone().transpose(1,0)
    
    return(H_d)




def get_D_from_W(W_i):
    W_sq=W_i**2
    
    diags=torch.sum(W_sq,dim=1)
    
    diags_sqr=torch.sqrt(diags)
    
    inv_diags=1/diags_sqr
    
    outval=torch.diag(inv_diags)
    
    outval*=0.5
    
    
    print(diags)
    print(diags_sqr)
    print(inv_diags)
    print(outval)




def train_second_method(dd,ntrials=100,MAX_IT=20,return_for_tune=False):
    #dd=return_data_for_second_method(dtype=dtype,si=si)#,X1_idx=X1_idx,X2_idx=X2_idx)


    L_Y=dd['L_Y'].cuda()
    V_Y_gt=dd['V_Y_gt'].cuda()
    U_Y_gt=dd['U_Y_gt'].cuda()
    N_lab=dd['N_lab']
    N_ulab=dd['N_ulab']
    
    X_ALL=dd['X_cat'].cuda()
    
    X_dim=X_ALL.shape[1]
    #X_ALL=torch.cat((X1_file,X2_file),dim=1)

    U_Y_init=U_Y_gt*0.0
    V_Y_init=V_Y_gt*0.0


    Y_START=torch.cat((L_Y,U_Y_init,V_Y_init),dim=0)



        

    d=X_ALL.shape[0]
    N=d




    X_combined=X_ALL
    Xi=X_ALL

    
    import random


    


    search_result={}

    for hpms_search in range(ntrials):
        print(f' doing hpms search number: {hpms_search} of total {ntrials}')
        
        
            #NB THESE NEED TO BE PROPERLY INIT!!!
        #k=15
        n_neighbours=random.choice([100,200])

        #k=n_neighbours
        
        # params=torch.rand(1,3)*10
        
        # pp=params.cpu().numpy().flatten()
        
        
        # alpha=pp[0].item()
        # gamma=pp[1].item()
        # beta=pp[2].item()
        
        
        
        # alpha=random.uniform(5,15)
        # gamma=random.uniform(0.05,2.5)
        # beta=random.uniform(0.05,1.0)
        
        
        #random hyperparmaeters for CG3-CG7
        # alpha=random.uniform(5,15)
        # gamma=random.uniform(0.75,2.15)
        # beta=random.uniform(0.05,1.0)
        
        #random hyperparams for CG2
        alpha=random.uniform(5,15)
        gamma=random.uniform(0.75,10)
        beta = random.uniform(5,15)
        
        W_l=torch.ones((X_dim,2),device=torch.device('cuda'))
        
        
        
        
        
            
            
            
        #N=100
        #X_combined=X_combined[:N,:]#[:,None]


        H_d=get_Hd(d)#.cuda()

        L_l=H_d @ torch.linalg.inv(X_combined@X_combined.transpose(1,0) + gamma * torch.eye(d,device=torch.device('cuda'))) @ H_d


        L_l=L_l.to(torch.device('cuda'))


        #get U matrix, LINE 5

        #N_lab=40
        U_l=torch.eye(N,device=torch.device('cuda'))
        EPS_BIG=1e6
        U_l[:N_lab,:N_lab]*=EPS_BIG

        # COMPUTE H_NL, LINE 6
        H_nl=get_Hd(Xi.shape[0])#.to(torch.device('cuda'))


            
        # COMPUTE P_L LINE 7
        P_l=torch.linalg.inv(alpha*beta*H_nl + U_l + L_l)


        # COMPUTE R_L LINE 8
        R_l = Xi.transpose(1,0) @ H_nl @ (torch.eye(Xi.shape[0],device=torch.device(device_str)) - alpha * beta * P_l) @ H_nl @ Xi#.transpose(1,0)
        L_Y=L_Y.to(torch.device('cuda'))
        Y_START=Y_START.to(torch.device('cuda'))
        
        # COMPUTE T_l line 9
        T_l = Xi.transpose(1,0) @ H_nl @ P_l @ U_l @ Y_START #need to set Y_l !!!


        
        
        
        
        
        
        #reinitialise values here
        
        list_of_Wl=[]
        list_of_Fl=[]
        list_of_bl=[]

        pc_correct=[]
        ulab_pc_corect=[]
        softmax_of_label=[]

        try:

            for n_trials in range(MAX_IT):
                

                # Compute D_l line 16
                d_weights = torch.linalg.norm(W_l,axis=1,ord=2) 

                d_num=torch.ones_like(d_weights)

                diag_terms = d_num / (2*d_weights) #should be [1xnl vec?]

                D_l = torch.eye(d_num.shape[0],device=torch.device(device_str)) * diag_terms


                D_l_tilde=D_l


                # update W_l line 18
                W_l = torch.linalg.inv(R_l + (alpha/beta * D_l) + (gamma/(alpha * beta) * D_l_tilde )) @ T_l


                # update F_r line 19
                F_l = torch.linalg.inv(alpha*beta * H_nl+ U_l + L_l  ) @ (alpha*beta*H_nl @ Xi @ W_l + U_l @ Y_START) #this is the predicted labels!!!


                # update b_l line 20
                b_l = (1/N) * ( F_l  - Xi@ W_l).transpose(1,0) @ torch.ones(N,device=torch.device(device_str)).view(-1,1)


                list_of_Wl.append(W_l)
                list_of_Fl.append(F_l)
                list_of_bl.append(b_l)
                
                
                smax=torch.nn.functional.softmax(F_l,dim=1)

                softmax_of_label.append(smax)
                
                
                val_partition=smax[N_lab+N_ulab:,:]
                val_pc_cor=(torch.argmax(val_partition,1)==torch.argmax(V_Y_gt,1).cuda()).float().mean()
                
                ulab_partition=smax[N_lab:N_ulab+N_lab,:]
                
                ulab_pc_cor=(torch.argmax(ulab_partition,1)==torch.argmax(U_Y_gt,1).cuda()).float().mean()
                
                
                pc_correct.append(val_pc_cor.item())
                ulab_pc_corect.append(ulab_pc_cor.item())
                
                if len(list_of_Wl)>=2:
                    W_l_last=list_of_Wl[-2]
                    W_l=list_of_Wl[-1]
                    W_diff=torch.linalg.norm((W_l-W_l_last).flatten(),2,0)


                    eps_converged=1e-4

                    if W_diff<=eps_converged:
                        break
                    
        except:
            print('singular value encountered, continuing')       
            
        finally:
            
            next
                    
    
        max_idx=np.argsort(pc_correct)[-1]
        
        pc_cor=pc_correct[max_idx]
        ulab_cor=ulab_pc_corect[max_idx]
        
        search_result[hpms_search]={}
        search_result[hpms_search]['alpha']=alpha#.item()
        search_result[hpms_search]['beta']=beta#.item()
        search_result[hpms_search]['gamma']=gamma#item()
        search_result[hpms_search]['val_pc_cor']=pc_cor
        search_result[hpms_search]['ulab_pc_cor']=ulab_cor
        search_result[hpms_search]['n_neighbours']=n_neighbours#.item()
        
        
    
    




    pdlist=[]


    for k in search_result.keys():
        df=pd.DataFrame.from_dict(search_result[k],orient='index')
        df.columns=[f'trial_{k}']
        pdlist.append(df.transpose())
        
        
    cat_results=pd.concat(pdlist).sort_values(by='val_pc_cor',ascending=False)

    max_val_pc=cat_results.val_pc_cor.values[0]

    sorted_first=cat_results[cat_results.val_pc_cor==max_val_pc].sort_index()

    best_ulab=torch.tensor(sorted_first.ulab_pc_cor.values[0],device=torch.device('cuda'))

    if return_for_tune:
        return(cat_results)
    else:
        return(best_ulab)

        
    



import time
import math

dt_x1={'n36_gaussian_mixture_d3_10000':[2,3],
       'n36_gaussian_mixture_d4_10000':[2,3],
       'n36_gaussian_mixture_d5_10000':[2,3],
       'n36_gaussian_mixture_d6_10000':[4,5],
       'n36_gaussian_mixture_d7_10000':[4,5],
       'n36_gaussian_mixture_d3':[2,3],
       'n36_gaussian_mixture_d4':[2,3],
       'n36_gaussian_mixture_d5':[2,3],
       'n36_gaussian_mixture_d6':[4,5],
       'n36_gaussian_mixture_d7':[4,5],
       'n36_gaussian_mixture_d3_5000':[2,3],
       'n36_gaussian_mixture_d4_5000':[2,3],
       'n36_gaussian_mixture_d5_5000':[2,3],
       'n36_gaussian_mixture_d6_5000':[4,5],
       'n36_gaussian_mixture_d7_5000':[4,5],
       'real_bcancer_diagnosis_zscore':[1, 5, 8, 9, 12, 15, 16, 19],
       'real_sachs_raf_log':[1,2]}
       


dt_x2={'n36_gaussian_mixture_d3_10000':[0,1],
       'n36_gaussian_mixture_d4_10000':[0,1],
       'n36_gaussian_mixture_d5_10000':[0,1],
       'n36_gaussian_mixture_d6_10000':[0,1],
       'n36_gaussian_mixture_d7_10000':[0,1],       
       'n36_gaussian_mixture_d3':[0,1],
       'n36_gaussian_mixture_d4':[0,1],
       'n36_gaussian_mixture_d5':[0,1],
       'n36_gaussian_mixture_d6':[0,1],
       'n36_gaussian_mixture_d7':[0,1],
       'n36_gaussian_mixture_d3_5000':[0,1],
       'n36_gaussian_mixture_d4_5000':[0,1],
       'n36_gaussian_mixture_d5_5000':[0,1],
       'n36_gaussian_mixture_d6_5000':[0,1],
       'n36_gaussian_mixture_d7_5000':[0,1],
       'real_bcancer_diagnosis_zscore':[0, 20, 14, 7],
       'real_sachs_raf_log':[0]}
       




si_list=[str(i) for i in range(100)]

dtypes=['n36_gaussian_mixture_d3','n36_gaussian_mixture_d5','n36_gaussian_mixture_d6','n36_gaussian_mixture_d4','n36_gaussian_mixture_d7',
    'n36_gaussian_mixture_d3_5000']


dtypes=['n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d4_5000']


dtypes=['n36_gaussian_mixture_d7_5000',
    'n36_gaussian_mixture_d3_10000','n36_gaussian_mixture_d5_10000']


#dtypes=['n36_gaussian_mixture_d6_10000','n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d7_10000']
dtypes=['n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d7_10000','n36_gaussian_mixture_d2','n36_gaussian_mixture_d2_5000','n36_gaussian_mixture_d2_10000']


#dtypes=['n36_gaussian_mixture_d3','n36_gaussian_mixture_d5','n36_gaussian_mixture_d6','n36_gaussian_mixture_d4','n36_gaussian_mixture_d7']
    # 'n36_gaussian_mixture_d3_5000','n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d4_5000','n36_gaussian_mixture_d7_5000',
    # 'n36_gaussian_mixture_d3_10000','n36_gaussian_mixture_d5_10000','n36_gaussian_mixture_d6_10000','n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d7_10000']


BEST_PARAMS_ALG1=dict(
beta=0.3503,
gamma=0.02727,
lambda_1=1.1571,
lambda_2=0.8727,
nn=200,
eps=0.01,
n_iterations=10)


def get_neighbours_method1(dtype):
    
    if dtype.endswith('_5000'):
        return(100)
    
    if dtype.endswith('_10000'):
        return(200)
    
    else:
        return(30)



dtypes=['n36_gaussian_mixture_d3','n36_gaussian_mixture_d4','n36_gaussian_mixture_d5','n36_gaussian_mixture_d6','n36_gaussian_mixture_d7']

dtypes=['n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d7_5000']

# dtypes=['n36_gaussian_mixture_d5']#,'n36_gaussian_mixture_d6','n36_gaussian_mixture_d7']
# dtypes=['n36_gaussian_mixture_d6','n36_gaussian_mixture_d7']
# dtypes=['n36_gaussian_mixture_d7']

dtypes=['n36_gaussian_mixture_d6']#,'n36_gaussian_mixture_d7']

dtypes=['n36_gaussian_mixture_d6_10000']#,'n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d7_5000']


dtypes=['real_bcancer_diagnosis_zscore']

dtypes=['real_sachs_raf_log']
#dtypes=['n36_gaussian_mixture_d7_5000']

all_results={}

import random

n_trials=5

out_fn='/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results'
import os
import pandas as pd

from IPython.core.debugger import set_trace


compute_second_method=True

compute_first_method=False
import torch._dynamo
torch._dynamo.config.suppress_errors = True
if __name__=='__main__':

    if compute_first_method:
        train_sfsm_compiled=torch.compile(train_SFSM)
        for dtype in dtypes:
            all_results[dtype]={}
            for si in si_list:
                
            
                X1_idx=dt_x1[dtype]
                X2_idx=dt_x2[dtype]
                dd=return_data_for_sfsm(dtype=dtype,si=si,X1_idx=X1_idx,X2_idx=X2_idx)
                
                vals_list=[]
                ulabs_list=[]
                list_of_param=[]
                
                #x_for_median=torch.cat((dd['X1_file'],dd['X2_file']),0)
    
                median_x=compute_median_pwd_torch(dd['X2_file'])
                
                #BEST_PARAMS_ALG1
                for nt in range(n_trials):
                    
                    sampled_algorith_parameters=dict(
                                                    beta=random.uniform(0.1,0.5),
                                                    gamma=random.uniform(0.03,0.05),
                                                    lambda_1=random.uniform(0.8,1.5),
                                                    lambda_2=random.uniform(0.8,1.5),
                                                    nn=200,#get_neighbours_method1(dtype),
                                                    eps=0.01,
                                                    n_iterations=10)

                    _, best_ulab,best_val =train_SFSM(config=sampled_algorith_parameters,dd=dd,median_x_val=median_x)
                    #_, best_ulab,best_val =train_sfsm_compiled(config=sampled_algorith_parameters,dd=dd)
                    
                    list_of_param.append(sampled_algorith_parameters)
                    vals_list.append(best_val)
                    ulabs_list.append(best_ulab)
                #set_trace()
                best_val_idx=np.argsort(vals_list)    
                best_ulab_overall=ulabs_list[best_val_idx[-1]]
                all_results[dtype][si]=best_ulab_overall
                
                print(f'completed for si: {si}')
                    
                    

            newdir=os.path.join(out_fn,'revision_method_1_results')
            os.makedirs(newdir,exist_ok=True)
            res_df=pd.DataFrame.from_dict(all_results[dtype],orient='index')
            save_fn=os.path.join(newdir,f'{dtype}_results.csv')
            res_df.to_csv(save_fn)
            print(f' results saved for dataset: {dtype}')
            
        
            
    if compute_second_method:
        
        #dtypes=['n36_gaussian_mixture_d3_10000']#,'n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d7_10000']
        # dtypes=['n36_gaussian_mixture_d3','n36_gaussian_mixture_d5','n36_gaussian_mixture_d6','n36_gaussian_mixture_d4','n36_gaussian_mixture_d7',
        #     'n36_gaussian_mixture_d3_5000','n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d4_5000','n36_gaussian_mixture_d7_5000',
        #     'n36_gaussian_mixture_d3_10000','n36_gaussian_mixture_d5_10000','n36_gaussian_mixture_d6_10000','n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d7_10000']
        
        #dtypes=['n36_gaussian_mixture_d2']#,'n36_gaussian_mixture_d1']

        si_list=[str(i) for i in range(100)]
        
        return_for_tune=False

        ntrials=10
        MAX_IT=20
        
        train_second_compile=torch.compile(train_second_method)
        
        for dtype in dtypes:
                all_results[dtype]={}
                best_meths=[]
                for si in si_list:
                        #X1_idx=dt_x1[dtype]
                        #X2_idx=dt_x2[dtype]
                        dd=return_data_for_second_method(dtype=dtype,si=si)#,X1_idx=X1_idx,X2_idx=X2_idx)
                        
                        if not return_for_tune:
                            best_ulab_overall=train_second_compile(dd,ntrials=ntrials,MAX_IT=MAX_IT,return_for_tune=return_for_tune)
                            all_results[dtype][si]=best_ulab_overall
                            
                            
                        
                        #best_ulab_overall=train_second_compile(dd,ntrials=ntrials,MAX_IT=MAX_IT)
                        
                        
                        #vals_list=[]
                        #ulabs_list=[]
                        #for nt in range(n_trials):
                        #best_ulab_overall=train_second_method(dtype,ntrials=ntrials,MAX_IT=MAX_IT)#,X1_idx,X2_idx)

                        #best_ulab_overall=train_second_method(dtype,ntrials=ntrials,MAX_IT=MAX_IT)#,X1_idx,X2_idx)
                        #best_ten=train_second_method(dd,ntrials=ntrials,MAX_IT=MAX_IT)#,X1_idx,X2_idx)
                        
                        if return_for_tune:
                            best_df_for_tune=train_second_compile(dd,ntrials=ntrials,MAX_IT=MAX_IT,return_for_tune=return_for_tune)
                            best_meths.append(best_df_for_tune)
                            
                        
                        #from IPython.core.debugger import set_trace
                        
                        
                        
                        
                        print(f'compute for si: {si}')

                if return_for_tune:       
                    set_trace()

                newdir=os.path.join(out_fn,'revision_method_2_results')
                os.makedirs(newdir,exist_ok=True)
                
                
                
                for k in all_results[dtype].keys():
                    all_results[dtype][k]=all_results[dtype][k].cpu().item()
                
                res_df=pd.DataFrame.from_dict(all_results[dtype],orient='index')
                save_fn=os.path.join(newdir,f'{dtype}_results.csv')
                res_df.to_csv(save_fn)
                print(f' results saved for dataset: {dtype}')
                
                
                
                
    #copy across results
    
    
    import pandas as pd

    import glob

    results_in_folder=glob.glob('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/revision_method_1_results/*.csv')#[0:1]


    rf_dict={r.split('/')[-1].replace('_results.csv',''):r for r in results_in_folder}
    model_name='ASSFSCMR'


    for r in rf_dict.keys():
        
        current_result=pd.read_csv(rf_dict[r])
        
        current_result.columns=['s_i','ulab_acc']
        
        current_result.set_index('s_i',inplace=True)
        #/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_n36_gaussian_mixture_d2_5000/saved_models/CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER-s_i=0_test_acc.out
        
        for si in current_result.index:
            ulab_acc=current_result.loc[si].ulab_acc.item()
            
            out_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{r}/saved_models/{model_name}-s_i={si}_unlabel_acc.out'
            with open(out_fn,'w') as f:
                f.write(str(ulab_acc)+'\n')
                
        print(r)
        print(si)




    results_in_folder=glob.glob('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/revision_method_2_results/*.csv')#[0:1]


    rf_dict={r.split('/')[-1].replace('_results.csv',''):r for r in results_in_folder}
    model_name='SFAMCAMT'


    for r in rf_dict.keys():
        
        current_result=pd.read_csv(rf_dict[r])
        
        current_result.columns=['s_i','ulab_acc']
        
        current_result.set_index('s_i',inplace=True)
        #/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_n36_gaussian_mixture_d2_5000/saved_models/CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER-s_i=0_test_acc.out
        
        for si in current_result.index:
            ulab_acc=current_result.loc[si].ulab_acc.item()
            
            out_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{r}/saved_models/{model_name}-s_i={si}_unlabel_acc.out'
            with open(out_fn,'w') as f:
                f.write(str(ulab_acc)+'\n')
                
        print(r)
        print(si)
