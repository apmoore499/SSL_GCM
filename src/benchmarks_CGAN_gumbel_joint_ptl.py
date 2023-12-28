#pip how to search for nightly versions
#--d_n=synthdag_dset_3 --s_i=1 --n_iterations=100 --lr=1e-3 --use_single_si=False --ulab_bsize=128 --patience=2 --min_temp=0.1 --init_temp=1.0 --n_trials=1 --val_loss_criterion=labelled_bce_and_all_feat_mmd --lab_bsize=4 --balance=False

######
#
# NEW PROCEDURE FOR TRAINING CGAN WITH GUMBEL SOFTMAX FOR SIMULTANEOUS LEARNING OF ENTIRE CAUSAL GRAPH
#
######
import os


import torch
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')


from torch.nn.functional import gumbel_softmax

import time
#from generative_models.gen_data_loaders import SSLDataModule_Unlabel_X
from collections import OrderedDict
st=time.time()

import sys
sys.path.append('generative_models')
sys.path.append('src')
sys.path.append('src/generative_models/')
import shutil
import copy
import argparse
from pathlib import Path

from benchmarks_utils import *
from generative_models.Generator_Y_from_X1 import *
from generative_models.Generator_X2_from_Y import *
from generative_models.Generator_X2_from_YX1 import *
from generative_models.Generator_Y import *
from generative_models.Generator_X1 import *
from generative_models.Generator_X_from_X import *

from gen_data_loaders import *
from generative_models.benchmarks_cgan import *
from parse_data import *
from torch import optim
import copy
from generative_models.Gumbel_module_combined import GumbelModuleCombined,return_chkpt_min_mmd_gumbel,GumbelDataModule,return_estop_min_mmd_gumbel
from generative_models.Gumbel_module_merge_cause_spouse import GumbelModuleCombinedCS#,return_chkpt_min_mmd_gumbel,GumbelDataModule,return_estop_min_mmd_gumbel


parser = argparse.ArgumentParser()

parser.add_argument('--d_n', help='dataset name',type=str)
parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ',type=int)
parser.add_argument('--n_iterations', help='how many iterations to train classifier for',type=int)
parser.add_argument('--lr',help='learning rate ie 1e-2, 1e-3,...',type=float)
parser.add_argument('--use_single_si',help='do we want to train on collection of si or only single instance',type=str)
parser.add_argument('--tot_bsize',help='unlabelled batch size, a power of 2',type=int,default=128)
parser.add_argument('--estop_patience',help='end training if loss not improve in this # epoch',type=int,default=10)
parser.add_argument('--min_temp',help='minimum annealing temp during gumbel',type=float)
parser.add_argument('--init_temp',help='initial temperature',type=float)
parser.add_argument('--n_trials',help='number of experiments to try for gumbel model',type=int,default=1)
parser.add_argument('--val_loss_criterion',help='labelled or unlabelled loss for val',type=str,default='labelled_bce_and_all_feat_mmd')
parser.add_argument('--lab_bsize',help='labeled batch size, a power of 2',type=int,default=4)
parser.add_argument('--balance',help='balancing labelled and unlabelled',type=str,default='False')
parser.add_argument('--use_benchmark_generators',help='using benchmark generators or not',type=str,default='False')
parser.add_argument('--use_bernoulli',help='using bernoulli or not',type=str,default='False')
parser.add_argument('--nhidden_layer', help='how many hidden layers in implicit model: 1,3,5', type=int, default=1)
parser.add_argument('--n_neuron_hidden', help='how many neurons in hidden layer if nhidden_layer==1', type=int,default=50)
parser.add_argument('--estop_mmd_type',help='use trans or val on mmd validation',default='trans')
parser.add_argument('--use_optimal_y_gen',help='use optimal y generator or not',default='False')
parser.add_argument('--plot_synthetic_dist',help='plotting of synthetic data (take extra time), not necessary',default='False')
parser.add_argument('--precision',help='traainer precision ie 32,16',default='32')
parser.add_argument('--compile_mmd_mode',help='compile mode for mmd losses',default='reduce-overhead')
parser.add_argument('--ignore_plot_5',help='ignore_first_plot_five, if true then dont do any plot',default='False')
parser.add_argument('--scale_for_indiv_plot',help='scale for pltoting',default=5)
parser.add_argument('--synthesise_w_cs',help='synthesise new data using groudn truth cause/spouse values',default='False')
















args = parser.parse_args()

args.use_single_si=str_to_bool(args.use_single_si)
args.balance=str_to_bool(args.balance)
args.use_benchmark_generators=str_to_bool(args.use_benchmark_generators)
args.use_bernoulli=str_to_bool(args.use_bernoulli)
args.use_optimal_y_gen=str_to_bool(args.use_optimal_y_gen)
args.plot_synthetic_dist=str_to_bool(args.plot_synthetic_dist)
args.ignore_plot_5=str_to_bool(args.ignore_plot_5)

print(f'plot synthetic dist: {args.plot_synthetic_dist}')



print('all args')
print(args)

# get dataspec, read in as dictionary
# this is the master dictionary database for parsing different datasets / misc modifications etc
master_spec=pd.read_excel('combined_spec.xls',sheet_name=None)

#write dataset spec shorthand
dspec=master_spec['dataset_spec']

dspec.set_index("d_n",inplace=True) #set this index for easier

#store index of pandas loc where we find the value
dspec=dspec.loc[args.d_n] #use this as reerence..
dspec.d_n= str(args.d_n) if dspec.d_n_type=='str' else int(args.d_n)

# GPU preamble
has_gpu=torch.cuda.is_available()
#gpu_kwargs={'gpus':torch.cuda.device_count(),'precision':16} if has_gpu else {}
gpu_kwargs={'gpus':torch.cuda.device_count()} if has_gpu else {}
hpms_dict={}

if has_gpu:
    device_string='cuda'
else:
    device_string='cpu'

d_n=args.d_n

n_iterations=args.n_iterations

dn_log=dspec.dn_log


SAVE_FOLDER=dspec.save_folder


algo_variant='gumbel' #ie gumbel joint

#now we want to read in dataset_si
csi=master_spec['dataset_si'][dspec.d_n].values
candidate_si=csi[~np.isnan(csi)]
args.optimal_si_list = [int(s) for s in candidate_si]
if args.use_single_si==True: #so we want to use single si, not entire range
    #args.optimal_si_list=[args.optimal_si_list[args.s_i]]
    args.optimal_si_list = [args.s_i]
#now we are onto training



x1=torch.randn((100,2)).cuda()

slist=[get_median_pwd(x1)*s for s in [0.125,0.25,0.5,1,2]]
slist=torch.tensor(slist).cuda().flatten()

#setup your mmd kernels and compile them for fast


dict_of_precompiled=return_dict_of_precompiled_mmd(args.compile_mmd_mode)



dict_of_precompiled['mix_rbf_kernel'](x1,x1,slist)[1]


# dict_of_precompiled={}

# # dict_of_precompiled['mix_rbf_kernel']=torch.compile(mix_rbf_kernel_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# # dict_of_precompiled['mix_rbf_mmd2']=torch.compile(mix_rbf_mmd2_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# # dict_of_precompiled['mix_rbf_mmd2_joint_1_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# # dict_of_precompiled['mix_rbf_mmd2_joint_regress_2_feature']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')


# #dict_of_precompiled['mix_rbf_kernel']=torch.compile(mix_rbf_kernel_class().to(torch.float16).cuda(),fullgraph=True,mode='max-autotune')
# dict_of_precompiled['mix_rbf_mmd2']=torch.compile(mix_rbf_mmd2_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# dict_of_precompiled['mix_rbf_mmd2_joint_1_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# dict_of_precompiled['mix_rbf_mmd2_joint_regress_2_feature']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')

# #mix_rbf_mmd2_class
# dict_of_precompiled['mix_rbf_mmd2_joint_regress_2_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')










for k, s_i in enumerate(args.optimal_si_list):
    print('doing s_i: {0}'.format(s_i))
    
    
    
    
    
    
    print('========================================')
    print(f'\n\n DOING dn : {args.d_n}\n\n')
    print('========================================')
    
    
    
    
    
    
    
    
    
    
    
    
    
    args.st = time.time()
    st=time.time()
    dsc_loader = eval(dspec.dataloader_function)  # within the spec
    dsc = dsc_loader(args, s_i, dspec)
    dsc = manipulate_dsc(dsc, dspec)  # adding extra label column and convenient features for complex data mod later on

    dsc.label_names = copy.deepcopy(dsc.labels)
    dsc.label_names_alphan={}

    for l in dsc.labels:
        dsc.label_names_alphan[l]=[l]
    dsc.label_names = [[d] for d in dsc.label_names]

    dsc.feature_dim = dspec.feature_dim
    dsc.label_name=dsc.labels[np.where([d=='label' for d in dsc.variable_types])[0][0]]

    if dsc.feature_dim==1:
        old_cols=[c for c in dsc.merge_dat.columns]
        newcols=[]
        for c in old_cols:
            if c[-2:]=='_0':
                newcols.append(c[:-2])
            else:
                newcols.append(c)
        dsc.merge_dat.columns=newcols

    if dsc.feature_dim>1:
        #need to modify dsc label names as these will correspond to cols in the data frame with values
        df_names=[c for c in dsc.merge_dat.columns]
        for v_i,label in enumerate(dsc.labels):
            #get data frame and pull out columns
            if dsc.variable_types[v_i]=='feature':
                #match columns
                matching_columns=[c for c in df_names if label in c]
                dsc.label_names[v_i]=matching_columns
                dsc.label_names_alphan[label]=matching_columns

            if dsc.variable_types[v_i]=='label':
                dsc.label_names_alphan[label]=[dsc.label_names_alphan[label][0]+'_0']


    #modify dsc.dag so that connnection between:
    # - cause and effect is gone


    temperature = args.init_temp # initial temperature
    order_to_train=dsc.dag.topological_sorting()
    
    
    
    #------------------------------------------
    
    
    
    
    
    
    
    
    # shuffle val/test/labelled so that the sample statistics / variance mean all match approximately.
    
    
    #set_trace()
    #dsc.merge_dat.type
    
    
    #set_trace()
    
    # dmt=dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','validation','test'])]
    # dmt['type']=dmt.type.sample(frac=1.0,random_state=1) #shuffle it
    # ulab=dsc.merge_dat[dsc.merge_dat.type=='unlabelled'] #pull out unlabeleld
    # dsc.merge_dat=pd.concat([dmt,ulab],axis=0)
    
    
    
    
    #-----------------------------------------
    
    
    
    
    
    all_dsc_vars=dsc.labels
    #derive the variable type from the labels
    all_dsc_vtypes=['feature' for v in all_dsc_vars]
    #get idx of y
    for k,vlabel in enumerate(all_dsc_vars):
        if 'Y' in vlabel or 'location' in vlabel:
            all_dsc_vtypes[k]='label'

    ddict_vtype={}
    for v,vtype in zip (all_dsc_vars,all_dsc_vtypes):
        ddict_vtype[v]=vtype

    dsc_generators=OrderedDict() # ordered dict of the generators
    dsc.causes_of_y = []

    # name of the labelled variable
    label_name = dsc.labels[np.where([d == 'label' for d in dsc.variable_types])[0][0]]

    label_idx=2

    causal_sources = [e.source for e in dsc.dag.es.select(_target=label_idx)]

    effect_variables = [e.target for e in dsc.dag.es.select(_source=label_idx)]

    # first sweep through = create generators
    var_conditional_feature_names={}
    dict_conditional_on_label={}

    for k_n,v_i in enumerate(order_to_train):
        # get variable type
        # get antecedent variables
        source_edges=dsc.dag.es.select(_target=v_i)
        source_vertices=[s_e.source_vertex for s_e in source_edges]
        sv=[v.index for v in source_vertices]# get index of them
        vinfo_dict={} #reset this dict

        print('inspecting causes / parents of variable number: {0} of {1}'.format(k_n+1,len(order_to_train)))
        cur_variable=dsc.labels[v_i]
        source_variable=[dsc.labels[s] for s in sv]
        print('cur_var_name : {0}'.format(cur_variable))
        print('source_var_name(s): {0}'.format(source_variable))

        cond_lab = source_variable #labels of conditinoal vars
        cond_vtype = [dsc.variable_types[l] for l in sv] #types of conditional vars
        cond_idx = sv #index of conditioanl vars in pandas df

        if len(sv)==0 and dsc.variable_types[v_i]=='feature':
            #if 'X' in dsc.labels[v_i]:
            cur_x_lab=dsc.label_names[v_i]
            #get the matching data...
            all_x=dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])]
            #match column names
            #ldc=[c for c in all_x.columns if cur_x_lab in c]
            #subset
            x_vals=torch.tensor(all_x[cur_x_lab].to_numpy('float32'),device=torch.device('cuda'))
            #get median pwd
            median_pwd=get_median_pwd(x_vals).item()
            #make generator for X
            curmod=0
            gen_x=Generator_X1(args.lr,
                               args.d_n,
                               s_i,
                               dspec.dn_log,
                               input_dim=dsc.feature_dim,
                               median_pwd=median_pwd,
                               sel_device='gpu',
                               precision=16,
                               batchnorm=True,
                               #num_hidden_layer=args.nhidden_layer,
                               #middle_layer_size=args.n_neuron_hidden,
                               label_batch_size=args.lab_bsize,
                               unlabel_batch_size=args.tot_bsize)


            dsc_generators[dsc.labels[v_i]]=gen_x
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = []

        elif len(sv) == 0 and dsc.variable_types[v_i]=='label':
            #make generator for Y
            yv=dsc.merge_dat[dsc.merge_dat.type=='labelled'][dsc.labels[v_i]].mean()
            geny=Generator_Y(args.d_n,
                 s_i,
                 dspec.dn_log,yv)
            dsc_generators[dsc.labels[v_i]]=geny
            dsc_generators[dsc.labels[v_i]].conditional_feature_names=[]
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False

        #########
        #X2->
        #X1->  Y
        #########

        elif len(sv)>0 and dsc.variable_types[v_i]=='label': #need to incorporate some conditional variables


            #get input dim from feature_dim and cond_vtype....
            num_features=np.sum([c=='feature' for c in cond_vtype])
            feature_inputs=num_features*dsc.feature_dim #features


            y_x1gen=Generator_Y_from_X1(args.lr,
                                       args.d_n,
                                       s_i,
                                       dspec.dn_log,
                                       input_dim=feature_inputs,
                                       output_dim=2,
                                       num_hidden_layer=args.nhidden_layer,
                                       middle_layer_size=args.n_neuron_hidden)


            y_x1gen.conditional_feature_names=cond_lab #save out for later
            dsc_generators[dsc.labels[v_i]]=copy.deepcopy(y_x1gen)
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab
            var_conditional_feature_names[dsc.labels[v_i]] = cond_lab
            dict_conditional_on_label[dsc.labels[v_i]]=False



        #X from Y, ie Y->X
        elif len(sv) == 1 and dsc.variable_types[v_i] == 'feature' and cond_vtype == ['label']:
            cur_x_lab=dsc.label_names[v_i]
            # subset
            x_vals=torch.tensor(all_x[cur_x_lab].to_numpy(),device=torch.device('cuda'))
            # get median pwd
            median_pwd = get_median_pwd(x_vals)
            # make gen
            x2_y_gen = Generator_X2_from_Y(args.lr,
                                           args.d_n,
                                           s_i,
                                           dspec.dn_log,
                                           input_dim=dsc.feature_dim + n_classes,
                                           output_dim=dsc.feature_dim,
                                           median_pwd=median_pwd,
                                           n_lab=-1,
                                           n_ulab=-1,
                                           num_hidden_layer=args.nhidden_layer,
                                           middle_layer_size=args.n_neuron_hidden,
                                           label_batch_size=args.lab_bsize,
                                           unlabel_batch_size=args.tot_bsize)  # this is it

            dsc_generators[dsc.labels[v_i]] = x2_y_gen
            dsc_generators[dsc.labels[v_i]].conditional_on_label=True
            dsc_generators[dsc.labels[v_i]].conditional_feature_names=[]
            var_conditional_feature_names[dsc.labels[v_i]] = []
            dict_conditional_on_label[dsc.labels[v_i]]=True

        elif (len(sv) >= 1) and (dsc.variable_types[v_i] == 'feature') and (np.all([c=='feature' for c in cond_vtype])):
            num_features = np.sum([c == 'feature' for c in cond_vtype])
            feature_inputs = num_features * dsc.feature_dim  # features
            cur_x_lab = dsc.labels[v_i]
            cond_x_lab = cond_lab
            # get the matching data...
            all_dat = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
            ##############################
            #    median pwd for target x
            ##############################
            # match column names
            ldc = [c for c in all_dat.columns if cur_x_lab in c]
            median_pwd_target = get_median_pwd(torch.tensor(all_dat[ldc].to_numpy(dtype='float32'),device=torch.device('cuda'))).item()

            ################################
            #  median pwd for conditional x
            ################################
            # match column names
            #ldc = [c for c in lab_dat.columns if any(c in cond_x_lab)]
            cond_x_vals = torch.tensor(all_dat[cond_x_lab].to_numpy(dtype='float32'),device=torch.device('cuda'))
            #median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))
            median_pwd_cond = get_median_pwd(cond_x_vals).item()#,device=torch.device('cuda'))


            genx_x = Generator_X_from_X(args.lr,
                               args.d_n,
                               s_i,
                               dspec.dn_log,
                                        input_dim=dsc.feature_dim + len(cond_x_lab)*dsc.feature_dim,
                                        output_dim=dsc.feature_dim,
                                        median_pwd_tx=median_pwd_target,
                                        median_pwd_cx=median_pwd_cond,
                                        num_hidden_layer=args.nhidden_layer,
                                        middle_layer_size=args.n_neuron_hidden,
                                        label_batch_size=args.lab_bsize,
                                        unlabel_batch_size=args.tot_bsize)  # this is it

            dsc_generators[dsc.labels[v_i]] = copy.deepcopy(genx_x)
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab
            var_conditional_feature_names[dsc.labels[v_i]] = cond_lab
            dict_conditional_on_label[dsc.labels[v_i]]=False

            # Y  ->  X2
            # X1 ->
        elif len(sv) > 1 and dsc.variable_types[v_i] == 'feature' and 'label' in cond_vtype:  # need to incorporate some conditional variables
            conditional_feature_names=[]
            label_name=[]
            for cl,ct in zip(cond_lab,cond_vtype):
                if ct=='feature':
                    conditional_feature_names.append(cl)
                if ct=='label':
                    label_name.append(cl)

            cur_x_lab = dsc.labels[v_i]
            all_dat = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]

            ##############################
            #    median pwd for target x
            ##############################
            # match column names
            ldc = [c for c in all_dat.columns if cur_x_lab in c]
            target_x_vals = all_dat[ldc].values
            median_pwd_target = get_median_pwd(torch.tensor(target_x_vals,device=torch.device('cuda'))).item()

            ################################
            #  median pwd for conditional x
            ################################
            # match column names
            concat_cond_lab = []
            for c in conditional_feature_names:
                concat_cond_lab = concat_cond_lab + dsc.label_names_alphan[c]

            cond_x_vals = all_dat[concat_cond_lab].values
            median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals,device=torch.device('cuda'))).item()

            lab_dat=dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])]

            input_dim=dsc.feature_dim+cond_x_vals.shape[1]+n_classes
            output_dim=dsc.feature_dim
            genx2_yx1=Generator_X2_from_YX1(args.lr,
                               args.d_n,
                               s_i,
                               dspec.dn_log,
                                           input_dim=input_dim,
                                           output_dim=output_dim,
                                           median_pwd_tx=median_pwd_target,
                                           median_pwd_cx=median_pwd_cond,
                                            n_lab=-1,
                                            n_ulab=-1,
                                           num_hidden_layer=args.nhidden_layer,
                                           middle_layer_size=args.n_neuron_hidden,
                                           label_batch_size=args.lab_bsize,
                                           unlabel_batch_size=args.tot_bsize)  

            genx2_yx1.conditional_feature_names=conditional_feature_names #store these for later
            genx2_yx1.conditional_on_label=True

            dsc_generators[dsc.labels[v_i]]=copy.deepcopy(genx2_yx1)
            dsc_generators[dsc.labels[v_i]].conditional_on_label = True
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = conditional_feature_names
            var_conditional_feature_names[dsc.labels[v_i]] = conditional_feature_names
            dict_conditional_on_label[dsc.labels[v_i]]=True

    for k in dsc_generators.keys():
        dsc_generators[k].configure_optimizers()

    #split the list of generators into:
    #1. unlabelled feature not dependent on Y
    #2. Y dependent on some X
    #3. X dependent on some Y

    label_name = dsc.labels[np.where([d == 'label' for d in dsc.variable_types])[0][0]]
    # first get the ordered list of keys...
    ordered_keys=[k for k in dsc_generators.keys()]

    ##set_trace()
    if type(label_name)=='list':
        label_idx = np.where(np.array(ordered_keys) == label_name[0])[0][0]
    else:
        label_idx = np.where(np.array(ordered_keys) == label_name)[0][0]



    order_to_train=dsc.dag.topological_sorting()
    #dsc_generators=OrderedDict() #ordered dict of the generators
    label_name = dsc.label_var
    ordered_keys=[dsc.labels[v] for v in order_to_train]
    if type(label_name)=='list':
        ln=label_name[0]
        label_idx = np.where(np.array(ordered_keys) == ln)[0][0]
    else:
        label_idx = np.where(np.array(ordered_keys) == label_name)[0][0]



    unlabelled_keys = ordered_keys[:label_idx]
    labelled_key = ordered_keys[label_idx]
    conditional_keys = ordered_keys[label_idx + 1:]


    #split into parent, spouse, child
    topsort_order = np.array(dsc.variable_types)

    #print('pausing here')

    lab_idx = np.where(topsort_order == 'label')[0]

    # need to partition into spouse also

    networkx_dag = dsc.dag.to_networkx()  # converting to networkx object for easier
    mb_dict = return_mb_dict(networkx_dag)
    mb_label_var = mb_dict[lab_idx[0]]

    #convert to unique elements

    #then remove variables shared bw parent&spouse to just parent

    for k in mb_label_var.keys():
        mb_label_var[k]=list(set(mb_label_var[k]))

    # ok now if any v common to spouse/children, put in effect only

    for v in mb_label_var['children']:
        if v in mb_label_var['spouses']:
            mb_label_var['spouses'].remove(v)

    # ok now if any v common to spouse/parent, put in parent only
    for v in mb_label_var['parent']:
        if v in mb_label_var['spouses']:
            mb_label_var['spouses'].remove(v)


    print('pausing here')
    # remove self from spouses
    try:
        mb_label_var['spouses'].remove(lab_idx[0])
    except:
        next
    finally:
        next



    # partition into causal/label/effect index
    ce_dict = {'cause': list(set(mb_label_var['parent'])),
                    'spouse': list(set(mb_label_var['spouses'])),
                    'lab': lab_idx[0],
                    'effect': list(set(mb_label_var['children']))}

    print('pausing here')


    unlabelled_keys = ordered_keys[:label_idx]
    labelled_key = ordered_keys[label_idx]
    conditional_keys = ordered_keys[label_idx + 1:]




    #print('pausing here')
    print('#########################')
    print('#                        ')
    print('TRAINING CAUSES + SPOUSES')
    print('#                        ')
    print('#########################')



    cause_spouse_v_idx=ce_dict['cause']+ce_dict['spouse']
    label_v_idx=ce_dict['lab']
    effect_v_idx=ce_dict['effect']


    # try like this - overwrite instead of code above!
    unlabelled_keys = [dsc.labels[k] for k in cause_spouse_v_idx] # cause spouse ... ordered_keys[:label_idx]
    labelled_key =  dsc.labels[ce_dict['lab']]   # label variable ordered_keys[label_idx]
    conditional_keys = [dsc.labels[k] for k in effect_v_idx]  # effect ...ordered_keys[label_idx + 1:]



    unlabelled_keys = [dsc.labels[k] for k in cause_spouse_v_idx] # cause spouse ... ordered_keys[:label_idx]

    cause_keys = [dsc.labels[k] for k in ce_dict['cause']]
    spouse_keys = [dsc.labels[k] for k in ce_dict['spouse']]


    unlabelled_keys = [dsc.labels[k] for k in cause_spouse_v_idx] # cause spouse ... ordered_keys[:label_idx]



    cur_x_lab = reduce_list([dsc.label_names[v] for v in cause_spouse_v_idx])
    
    #cur_x_lab = dsc.label_names[v_i]
    # get the matching data...
    all_x = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
    # match column names
    # ldc=[c for c in all_x.columns if cur_x_lab in c]
    # subset
    
    ##set_trace()
    
    x_vals = torch.tensor(all_x[cur_x_lab].to_numpy('float32'),device=torch.device('cuda'))
    # get median pwd
    median_pwd = get_median_pwd(x_vals).item()
    
    print(f'median pwd {median_pwd}')
    # make generator for X
    curmod = 0
    gen_x = Generator_X1(args.lr,
                            args.d_n,
                            s_i,
                            dspec.dn_log,
                            input_dim=dsc.feature_dim*len(cause_spouse_v_idx),
                            gen_layers=[100],
                            batchnorm=True,
                            median_pwd=median_pwd,
                            label_batch_size=args.lab_bsize,
                            unlabel_batch_size=args.tot_bsize)

    tloader = SSLDataModule_Unlabel_X(dsc.merge_dat, target_x=cur_x_lab, batch_size=args.tot_bsize)
    #model_name = create_model_name(dsc.labels[v_i], algo_variant)
    #vn_concat='_'.join([dsc.labels[v] for v in cause_spouse_v_idx])
    vn_concat='CAUSE_SPOUSE'
    model_name = create_model_name(vn_concat, algo_variant)

    gen_x.set_precompiled(dict_of_precompiled)

    #NB SHOULD USE MIN OVER TRANSDUCTIVE, IE ALL UNLABELLED CASES FOR X1
    
    if args.estop_mmd_type == 'val':
        estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
        min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,dspec.save_folder)  # returns min checkpoint
        
    elif args.estop_mmd_type == 'trans':
        estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
        min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,dspec.save_folder)  # returns min checkpoint
        

    elif args.estop_mmd_type == 'val_trans': #uses trans just for generatorX1 only!!! 
        #estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
        estop_cb = return_early_stop_min_val_trans_mmd(patience=args.estop_patience)
        min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,dspec.save_folder)  # returns min checkpoint
    
    
    #from pytorch_lightning.profilers import Profiler, PassThroughProfiler,AdvancedProfiler
    
    #profiler = AdvancedProfiler(dirpath='/media/krillman/240GB_DATA/codes2/SSL_GCM/src/profiling',filename='profile_gumbel_x1.log')
    
    
    profiler=None


    callbacks = [min_mmd_checkpoint_callback, estop_cb]

    tb_logger = create_logger(model_name, d_n, s_i)
    trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations,precision=args.precision,profiler=profiler)
    delete_old_saved_models(model_name, dspec.save_folder, s_i)

    trainer.fit(gen_x, tloader)  # train here

    mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)
    gen_x = type(gen_x).load_from_checkpoint(checkpoint_path=mod_names[0])  # loads correct model
    
    
    
    #set_trace()
    
    
    





    dsc_generators['ordered_v']={} #to be used later on for retrieving variable names, in correct order
    dsc_generators['ordered_v_alphan']={}
    
    
    dsc_generators['cause_spouse_generator']=gen_x
    dsc_generators['cause_spouse_generator'].conditional_on_label = False
    dsc_generators['cause_spouse_generator'].conditional_feature_names = []
    dsc_generators['ordered_v']['cause']=[dsc.labels[v] for v in cause_spouse_v_idx]
    dsc_generators['ordered_v_alphan']['cause'] = cur_x_lab








        
        
        
        
        
        # dsc_generators[dsc.labels[v_i]] = gen_x
        # dsc_generators[dsc.labels[v_i]].conditional_on_label = False
        # dsc_generators[dsc.labels[v_i]].conditional_feature_names = []
        
    
    
    del trainer
    
    
    del tloader










    print(f'unlabelled_keys: {unlabelled_keys}')
    print(f'labelled_key: {labelled_key}')
    print(f'conditional_keys: {conditional_keys}')

    # get all keys together
    all_keys=unlabelled_keys+[labelled_key]+conditional_keys

    # create new optimiser object to perform simultaneous update on all parameters (during gumbel)
    all_parameters_labelled=[]
    for k in [labelled_key]+conditional_keys:
        all_parameters_labelled=all_parameters_labelled+list(dsc_generators[k].parameters())
    combined_labelled_optimiser=optim.Adam(all_parameters_labelled,lr=1e-3)

    all_parameters_unlabelled=[]
    for k in conditional_keys:
        all_parameters_unlabelled=all_parameters_unlabelled+list(dsc_generators[k].parameters())
    combined_unlabelled_optimiser=optim.Adam(all_parameters_unlabelled,lr=1e-3)

    # all feature names
    all_feat_names=unlabelled_keys+conditional_keys


    # convert all_feat_names to desired
    i = [dsc.label_names_alphan[k] for k in all_feat_names]
    all_feat_names_sub = [item for sublist in i for item in sublist]

    #find out idx of each variable cos we need this later on for dataloader
    feature_idx_dict={}

    for n in all_feat_names:
        feature_idx_dict[n] = all_feat_names.index(n)


    # partitioning datasets for training, validation etc
    all_unlabelled=dsc.merge_dat[dsc.merge_dat.type=='unlabelled'][all_feat_names_sub]
    all_unlabelled_and_labelled = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])][all_feat_names_sub]
    all_labelled_features = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled'])][all_feat_names_sub]
    labelled_key_df=dsc.label_names_alphan[labelled_key]
    all_labelled_label = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled'])][labelled_key]










    dsc_idx_dict = {}
    dsc_idx_dict['cause_spouse'] = np.array(dsc_generators['ordered_v_alphan']['cause'])



    # get median_pwd of cause/spouse
    median_pwd_dict={}
    median_pwd_dict['cause_spouse']=get_median_pwd(torch.tensor(all_unlabelled_and_labelled[dsc_idx_dict['cause_spouse']].values,device=device_string)).item()



    #sigma_list_total

    #validation data
    all_validation_features = dsc.merge_dat[dsc.merge_dat.type == 'validation'][all_feat_names_sub]
    all_validation_label = dsc.merge_dat[dsc.merge_dat.type == 'validation'][labelled_key]


    # convert all_feat_names to desired
    i = [dsc.label_names_alphan[k] for k in conditional_keys]
    ck_sub = [item for sublist in i for item in sublist]



    #get median pairwise distances for mmd loss func
    #median pwd of labelled + unlabelled X in training set
    total_median_pwd=get_median_pwd(torch.tensor(all_unlabelled_and_labelled.to_numpy('float32'),device=torch.device('cuda'))).item()
    #median pwd of labelled X in training set only
    labelled_median_pwd = get_median_pwd(torch.tensor(all_labelled_features.to_numpy('float32'),device=torch.device('cuda'))).item()




    #get median pwd on label/unlabel datar

    ck_lab_ulab= dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])][ck_sub]
    ck_lab= dsc.merge_dat[dsc.merge_dat.type.isin(['labelled'])][ck_sub]

    #all_unlabelled_and_labelled = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])][all_feat_names_sub]

    total_median_pwd=get_median_pwd(torch.tensor(ck_lab_ulab.to_numpy('float32'),device=torch.device('cuda'))).item()
    #median pwd of labelled X in training set only
    labelled_median_pwd = get_median_pwd(torch.tensor(ck_lab.to_numpy('float32'),device=torch.device('cuda'))).item()



    

    #median pwd of each feature variable...
    #median_pwd_dict={}
    for c in unlabelled_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_unlabelled_and_labelled[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.tensor(cur_feature_vals.to_numpy('float32'),device=torch.device('cuda'))).item()
        #store
        median_pwd_dict[c]=cur_mpwd

    for c in conditional_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_unlabelled_and_labelled[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.tensor(cur_feature_vals.to_numpy('float32'),device=torch.device('cuda'))).item()
        #store
        median_pwd_dict[c]=cur_mpwd
        
        
    
    
    median_pwd_dict_lab={}
        
    for c in unlabelled_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_labelled_features[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.tensor(cur_feature_vals.to_numpy('float32'),device=torch.device('cuda'))).item()
        #store
        median_pwd_dict_lab[c]=cur_mpwd



    for c in conditional_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_labelled_features[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.tensor(cur_feature_vals.to_numpy('float32'),device=torch.device('cuda'))).item()
        
        # dists=[]
        
        # for yl in [0,1]:
        #     cfv=all_labelled_features[a_labs][all_labelled_label==yl]
            
        #     it=torch.tensor(cfv.to_numpy('float32'),device=torch.device('cuda'))
        #     pd2=torch.cdist(it,it)
            
        #     pd2=pd2[torch.triu(torch.ones_like(pd2), diagonal=1).to(torch.bool)]
        
        #     dists.append(pd2)
            
        # mdist=torch.hstack(dists).median()
        # cur_mpwd=mdist.item()

        
        #store
        median_pwd_dict_lab[c]=cur_mpwd
        
        
    
    
    median_pwd_dict_val={}
        
    for c in unlabelled_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_validation_features[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.tensor(cur_feature_vals.to_numpy('float32'),device=torch.device('cuda'))).item()
        #store
        median_pwd_dict_val[c]=cur_mpwd



    for c in conditional_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_validation_features[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.tensor(cur_feature_vals.to_numpy('float32'),device=torch.device('cuda'))).item()
        
        # dists=[]
        
        # for yl in [0,1]:
        #     cfv=all_labelled_features[a_labs][all_labelled_label==yl]
            
        #     it=torch.tensor(cfv.to_numpy('float32'),device=torch.device('cuda'))
        #     pd2=torch.cdist(it,it)
            
        #     pd2=pd2[torch.triu(torch.ones_like(pd2), diagonal=1).to(torch.bool)]
        
        #     dists.append(pd2)
            
        # mdist=torch.hstack(dists).median()
        # cur_mpwd=mdist.item()

        
        #store
        median_pwd_dict_val[c]=cur_mpwd
        
        
    
    
    
    
    
    
    
    
    
    
    
    effect_vars=[]
    for c in conditional_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]    
        effect_vars+=a_labs
    #dsc_generators['ordered_v']['effect']['outputs_alphan']
        
    
    top_sort=np.array(dsc.dag.topological_sorting())
    
    dsc_generators['ordered_v']['label'] = {}
    
    
    label_idx=np.argwhere(np.array(dsc.variable_types)=='label').flatten()[0]
    
    
    
    
    
    
    ancestor_of_label_idx=dsc.get_source_vi(label_idx)
    
    
    dsc_generators['ordered_v']['label']['inputs']=[dsc.labels[a] for a in ancestor_of_label_idx]
    
    
    # ancestors=dsc.dag.subcomponent(label_idx,mode=ig.IN)
    
    # if label_idx in ancestors:
    #     ancestors.remove(label_idx)
    
    #ancestor_names=dsc.
    
    #unlabelled_keys
    # cond_labs=[dsc.label_names[a] for a in ancestors]
    
    
    # cat_labs=[]
    
    # for c in cond_labs:
    #     cat_labs+=c
    
    
    
    
    
    # dsc_generators['ordered_v']['effect'] = {}
    # dsc_generators['ordered_v']['effect']['inputs'] = cond_lab
    # dsc_generators['ordered_v']['effect']['input_features_alphan'] = []
    # dsc_generators['ordered_v']['effect']['input_label_alphan'] = [label_name]

    # dsc_generators['ordered_v']['effect']['outputs_alphan']=effect_vars
    all_keys = unlabelled_keys + [labelled_key] + conditional_keys
        
        
        
    dsc_generators['ordered_v']['effect']={}
    dsc_generators['ordered_v']['effect']['label_names']=conditional_keys
    
    
    
    #dsc_generators['ordered_v']['effect']['inputs']={}
    
    

    
    
    vn=[dsc.label_names_alphan[v] for v in conditional_keys]
    
    cat_out=[]
    
    for v in vn:
        cat_out+=v
    
    dsc_generators['ordered_v']['effect']['outputs_alphan'] = cat_out
        
    all_feature_names_alphan= dsc_idx_dict['cause_spouse'].tolist() + dsc_generators['ordered_v']['effect']['outputs_alphan']
        
        
        
        #ce_dict['cause']
        
        
    #dsc_idx_dict['effect'] = np.array(dsc_generators['ordered_v']['effect']['outputs_alphan'])
    dsc_idx_dict['all_features'] = np.array(all_feature_names_alphan)
    dsc_idx_dict['cs_idx'] = [np.where(c == dsc_idx_dict['all_features'])[0][0] for c in dsc_idx_dict['cause_spouse']]
    dsc_idx_dict['ef_idx'] = [np.where(c == dsc_idx_dict['all_features'])[0][0] for c in dsc_generators['ordered_v']['effect']['outputs_alphan']]
    dsc_idx_dict['all_features'] = [np.where(c == dsc_idx_dict['all_features'])[0][0] for c in dsc_idx_dict['all_features']]

    


    #dsc_generators['ordered_v']['label']['inputs'] = cause_keys#dsc_generators['ordered_v']['cause']
    
    
    ccc=[]
    #dsc.get_source_vi(0)
    
    for c in dsc_generators['ordered_v']['label']['inputs']:
        current_an=dsc.label_names_alphan[c]
        ccc+=current_an
    
    dsc_generators['ordered_v']['label']['inputs_alphan'] = ccc
    dsc_generators['ordered_v']['label']['output'] = dsc.labels[label_idx]

    #print('pausing here')

    # pull out generators which are causes of Y, store in ```causes_of_y```
    # for example, if we have:
    # X1 -> Y <- X2
    # pull out generators for X1, X2, stored in "conditional_feature_variables"

    causes_of_y= dsc_generators['ordered_v']['label']['inputs']

    i = [dsc.label_names_alphan[k] for k in causes_of_y]

    # store causes of y column names here:
    causes_of_y_feat_names = [item for sublist in i for item in sublist]

    all_x_cols=[c for c in all_unlabelled.columns]

    #convert column names to index
    causes_of_y_idx_dl=[]

    for k,cf in enumerate(all_x_cols):
        if cf in causes_of_y_feat_names:
            causes_of_y_idx_dl.append(k)


    #get concat y causes for model prediction

    #all_y_antecedent = [dsc.label_names_alphan[f] for f in causes_of_y_feat_names]
    #concat_ycauses = np.array(all_y_antecedent).flatten().tolist()

    #ulab_dloader=DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled.values)),batch_size=args.lab_bsize,shuffle=True)
    feature_idx_subdict={c:idx for idx,c in enumerate(all_unlabelled_and_labelled.columns) if all_unlabelled_and_labelled.columns[idx]==c }

    # now, find variables for which Y is a cause, ie:
    # Y -> X2

    val_losses_joint=[]
    label_ancestor_dict={}
    for k,var_name in enumerate(dsc.labels):
        current_neighbourhood=dsc.dag.neighborhood(k, mode='in')
        current_neighbourhood.remove(k)#remove the actual vertex itself..
        #get the labels corresponding to variables
        ancestor_labels=[dsc.labels[vn] for vn in current_neighbourhood]
        ancestor_contains_y=labelled_key in ancestor_labels
        label_ancestor_dict[var_name]=ancestor_contains_y

    # we need val_losses_joint to keep track if
    # loss hasn't improved in previous_trajectory epochs
    PREVIOUS_TRAJECTORY=args.estop_patience
    # but delay this comparative process to start only after epo here
    DELAYED_MIN_START=100 # set to 50 so that it does some nice work args.estop_patience
    # labelled batch size, set to 4 usually but can be changed by kwargs input to main.py
    labelled_batch_size=args.lab_bsize
    # set min overall loss to 9999 cos we use this as value to minimise
    min_overall_loss=9999
    # optimised models will be held in this dictionary
    min_overall_mod_dict={}

    # all variables in causal graph
    #all_keys = unlabelled_keys + [labelled_key] + conditional_keys

    # array of the min loss at each trial end.
    # only used if we experiment with training generator multiple trials from scratch
    mins_end_of_trial = []
    # list for saving optimal-performing model after each trial.
    # only used if we are doing multiple trials
    optimal_mods = []
    # all minimum values for each trial
    # only used if we are doing multiple trials
    all_mins=[]






    ###################
    ###################
    # TRAINING GUMBEL #
    ###################
    ###################

    # Now we are using gumbel method to train all other generators
    # form dict comprising the idx of conditional features in the dataloader for each variable to be trained
    all_keys=[k for k in dsc_generators.keys()]
    # make sure that label is removed from conditional_feature_names...
    for k in all_keys:
        if hasattr(dsc_generators[k],'conditional_feature_names'):
            cond_feat_var=dsc_generators[k].conditional_feature_names
            if dsc.label_name in cond_feat_var:
                new_fn=[f for f in cond_feat_var if f!=dsc.label_name] #new feature names
                dsc_generators[k].conditional_feature_names=new_fn #overwrite conditional_feature_names w label var removed



    # GPU preamble
    has_gpu=torch.cuda.is_available()
    #gpu_kwargs={'gpus':torch.cuda.device_count(),'precision':16} if has_gpu else {}
    gpu_kwargs={'gpus':torch.cuda.device_count()} if has_gpu else {}
    if has_gpu:
        for k in [labelled_key] + conditional_keys:
            dsc_generators[k].cuda()

    for ttt in range(args.n_trials): #n_trials = 1 usually. set > 1 to check stability of gumbel estimation
        # create new optimiser object to perform simultaneous update
        # when using gumbel noise

        # all_parameters_labelled = [] # parameters of all generators including generator for Y

        # for k in [labelled_key] + conditional_keys:
        #     all_parameters_labelled = all_parameters_labelled + list(dsc_generators[k].parameters())

        # combined_labelled_optimiser = optim.Adam(all_parameters_labelled, lr=args.lr) #reset labelled opt

        # all_parameters_unlabelled = []

        # for k in conditional_keys:
        #     all_parameters_unlabelled = all_parameters_unlabelled + list(dsc_generators[k].parameters())

        # combined_unlabelled_optimiser = optim.Adam(all_parameters_unlabelled, lr=args.lr)  #reset unlabelled opt

        #reset label generators
        dsc_generators[labelled_key].apply(init_weights)

        # reset conditional generators
        for c in conditional_keys:
            dsc_generators[c].apply(init_weights)

        mintemps_dict={}
        optimal_mods_dict=OrderedDict()
        temps_tried=[]
        val_losses_joint=[]
        inv_val_accuracies=[]
        val_bces=[]

        rt_dict={} #for storing loss at each temp
        otemp_dict={} #for storing optimal temp value
        tcounter=1

    
        import torch
        import torchvision.models as models
        from torch.profiler import profile, record_function, ProfilerActivity
        
                # Set up the profiler
        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),record_shapes=True,profile_memory=True) as prof:
        #with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
        
        
        #here is where we import new code...
        import numpy as np

        def generate_exponential_decay_list(a, b, n):
            """
            Generate a list of n values decreasing exponentially from a to b.

            Args:
            a (float): Start value.
            b (float): Final value.
            n (int): Number of values.

            Returns:
            np.array: An array of n values decreasing exponentially from a to b.
            """
            k = -np.log(b/a) / (n - 1)  # Calculate the decay constant
            x_values = np.arange(n)    # Generate n values from 0 to n-1
            return a * np.exp(-k * x_values)  # Apply the exponential decay formula

        # Example usage
        a = 10   # Start value
        b = 0.2  # Final value
        n = 100  # Number of values

        #exponential_decay_list = generate_exponential_decay_list(a, b, n)
        #print(exponential_decay_list)

        
        
        templist = np.linspace(0.99,0.1,args.n_iterations)
        #templist = np.linspace(0.99,0.1,3)
        
        
        #templist = exponential_decay_list
        
        MIN_MONITOR='labelled_bce_and_all_feat_mmd'
        MODEL_NAME='GUMBEL_COMBINED_MODULE'
        
        n_labelled = all_labelled_features.shape[0]
        n_unlabelled = all_unlabelled.shape[0]
        
        
            #     if args.precision==16:
            # #move to half precision
            
            # for k in gumbel_module.dsc_generators.keys():
            #     current_module= gumbel_module.dsc_generators[k]
                
            #     if type(current_module)==Generator_Y_from_X1
            #         current_module.classifier=current_module.classifier.half()
            #         current_module=current_module.half()
                
                
            #     else:
            #         current_module.gen=current_module.gen.half()
            #         current_module=current_module.half()
                
            #     gumbel_module.dsc_generators[k]=current_module
        
        
        dscg_label={}
        
        dscg_effect={}
        dsc_generators_gumbel={}
        
        
        for k in dsc_generators.keys():
            
            if k==dsc.label_var:
                
                dscg_label[k]=dsc_generators[k]
                
                dsc_generators_gumbel[k]=dsc_generators[k]
                
            if k in conditional_keys:
                
                #dscg_label[k]=dsc_generators[k]
                dscg_effect[k]=dsc_generators[k]
                
                dsc_generators_gumbel[k]=dsc_generators[k]
                
        
        
        
        
        
        if args.precision==16:
            #move to half precision
            for k in dsc_generators_gumbel.keys():
                
                dsc_generators_gumbel[k].eval()
                
                dsc_generators_gumbel[k]=dsc_generators_gumbel[k].half()
                
                if hasattr(dsc_generators_gumbel[k],'classifier'):
                    dsc_generators_gumbel[k].classifier=dsc_generators_gumbel[k].classifier.half()
                    
                if hasattr(dsc_generators_gumbel[k],'gen'):
                    dsc_generators_gumbel[k].gen=dsc_generators_gumbel[k].gen.half()
                    
                
        # for p in dsc_generators_gumbel[k].parameters():
        #     dummy=1
        #     break
        
        
        
        init_kwargs=dict(dsc_generators=dsc_generators_gumbel,
                                           labelled_key=labelled_key,
                                           feature_idx_subdict=feature_idx_subdict,
                                            unlabelled_keys=unlabelled_keys,
                                            conditional_keys=conditional_keys,
                                            median_pwd_dict=median_pwd_dict,
                                            median_pwd_dict_lab=median_pwd_dict_lab,
                                            median_pwd_dict_val=median_pwd_dict_val,
                                            labelled_median_pwd=labelled_median_pwd,
                                            total_median_pwd=total_median_pwd,
                                            val_loss_criterion=MIN_MONITOR,
                                            causes_of_y_idx_dl=causes_of_y_idx_dl,
                                            all_feat_names=all_feat_names,
                                            templist=templist,
                                            d_n=d_n,
                                            s_i=s_i,
                                            dn_log=dn_log,
                                            n_labelled=n_labelled,
                                            n_unlabelled=n_unlabelled,
                                            lab_bsize=args.lab_bsize,
                                            tot_bsize=args.tot_bsize,
                                            lr=args.lr,
                                            dsc=dsc)
    
     
        gumbel_module=GumbelModuleCombinedCS(**init_kwargs)
        
        
        gumbel_module.setup_compiled_mmd_losses(dict_of_precompiled=dict_of_precompiled) #important
        
        
        import copy
        
        old_dsc_generators=copy.deepcopy(torch.nn.ModuleDict(dsc_generators_gumbel).state_dict())
        
        #gumbel_module.set_dsc_generators(dsc_generators=dsc_generators)

   
        # gumbel_module=GumbelModuleCombined(dsc_generators=dsc_generators,
        #                                    labelled_key=labelled_key,
        #                                    feature_idx_subdict=feature_idx_subdict,
        #                                     unlabelled_keys=unlabelled_keys,
        #                                     conditional_keys=conditional_keys,
        #                                     median_pwd_dict=median_pwd_dict,
        #                                     labelled_median_pwd=labelled_median_pwd,
        #                                     total_median_pwd=total_median_pwd,
        #                                     val_loss_criterion=MIN_MONITOR,
        #                                     causes_of_y_idx_dl=causes_of_y_idx_dl,
        #                                     all_feat_names=all_feat_names,
        #                                     templist=templist,
        #                                     d_n=d_n,
        #                                     s_i=s_i,
        #                                     dn_log=dn_log,
        #                                     n_labelled=n_labelled,
        #                                     n_unlabelled=n_unlabelled,
        #                                     lr=args.lr,
        #                                     dsc=dsc)
        
        min_chkpt=return_chkpt_min_mmd_gumbel(model_name=MODEL_NAME,dspec_save_folder=dspec.save_folder,monitor=MIN_MONITOR)
        
        
        estop_min=return_estop_min_mmd_gumbel(monitor=MIN_MONITOR,patience=50)
        
    
        data_module=GumbelDataModule(merge_dat=dsc.merge_dat,
                                        lab_name=dsc.label_var,
                                        lab_bsize=args.lab_bsize,
                                        tot_bsize=args.tot_bsize)
        

        data_module.setup(stage='fit',precision=args.precision)
        callbacks=[min_chkpt,estop_min]
        
        #from pytorch_lightning.profilers import Profiler, PassThroughProfiler,AdvancedProfiler
        
        #profiler = AdvancedProfiler(dirpath='/media/krillman/240GB_DATA/codes2/SSL_GCM/src/profiling',filename='profile_gumbel_togeth.log')
        
        profiler=None
        
        
        trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=len(templist),precision=args.precision,profiler=profiler)
        
        
        #set_trace()
        
        delete_old_saved_models(model_name=MODEL_NAME, save_dir=dspec.save_folder, s_i=s_i)
        
        trainer.fit(gumbel_module, data_module)  # train here

        
        ##set_trace()
        
        mod_names = return_saved_model_name(model_name=MODEL_NAME, save_dir=dspec.save_folder, d_n=dspec.dn_log, s_i=s_i)

        
        assert len(mod_names)==1,'error more than one optimal model!!!'
        
        gumbel_module = type(gumbel_module).load_from_checkpoint(checkpoint_path=mod_names[0],**init_kwargs)  # loads correct model

        
        print('pausing here')

    #setting optimal mods like so
    optimal_mods = copy.deepcopy(gumbel_module.dsc_generators)
    #setting our gen for label variable to be whatever was decided according to min inverse acc


    if args.use_optimal_y_gen:
        print('overwriting p(Y|X) for optimal generator...')

        optimal_mods[dsc.label_var] = optimal_label_gen

    #print('list of bce')
    #print(val_bces)


    #print('min of all bce')
    #print(current_min_bce)
    #mintemp = temp
    #current_val_loss = val_losses_joint[-1]
    #minloss = current_val_loss
    #converged = False

    #print('pause placeholder')
    #print('min loss and temp: ')
    #minloss=1
    #mintemp=1
    #print(minloss)
    #print(mintemp)


    
    #dsc_generators={}
    for k in optimal_mods.keys():
        dsc_generators[k]=dsc_generators_gumbel[k]


    keep_keys=['cause_spouse_generator']+list(optimal_mods.keys())


    all_keys=list(dsc_generators.keys())
    
    
    dsc_cp={}

    for k in keep_keys:
        dsc_cp[k]=dsc_generators[k]

    #move everything back onto cpu (not necessary!)
    #if has_gpu:
    #    for k in [labelled_key]+conditional_keys:
    #        dsc_generators[k].to('cpu')


    #dsc_generators['cause_spouse_generator']

    #dsc_generators=dsc_cp
    #put in eval mode
    for k in keep_keys:
        dsc_generators[k].eval()

    modules=[m for m in dsc_generators[k].modules()]
    
    
    for k in dsc_generators.keys():
        
        try:
            dsc_generators[k].delete_compiled_modules()
        except:
            
            print(f'error fail to delte compield for : {k}')
            
        finally:
            
            continue

    


    for k in keep_keys:
        #find old model and delete, using string matching
        bmodel = create_model_name(k, 'gumbel')
        model_to_search_for = dspec.save_folder + '/saved_models/' + bmodel + "-s_i={0}-*".format(s_i)
        candidate_models = glob.glob(model_to_search_for)
        #remove these models
        for m in candidate_models:
            os.remove(m)
        #create model name
        model_save_name = dspec.save_folder + '/saved_models/' + bmodel + "-s_i={0}-.pt".format(s_i)
        #and then save
        #try dleete any compiled modules!!
        
        
        
        torch.save(dsc_generators[k],model_save_name)

    # # creating synthetic data
    # if args.use_benchmark_generators:
    #     # we need to replace the generators in dsc_generators,
    #     for k in unlabelled_keys:
    #         # find model name...
    #         bmodel=create_model_name(k,'basic')
    #         model_to_search_for=dspec.save_folder+'/saved_models/'+bmodel+"*-s_i={0}-epoch*".format(s_i)
    #         candidate_models=glob.glob(model_to_search_for)
    #         #load thee model
    #         dsc_generators[k]=type(dsc_generators[k]).load_from_checkpoint(checkpoint_path=candidate_models[0])

    #n_samples = 30000
    n_samples=int(30000*min(dspec.n_unlabelled/1000,5)) #try to set this to deal wtih very large unalbeleld size...
    
    #synthetic_samples_dict = generate_samples_to_dict(dsc, has_gpu, dsc_generators, device_string, n_samples,gumbel=True,tau=mintemp)
    #don't use gumbel dist with the temp
    #synthetic_samples_dict = generate_samples_to_dict(dsc, has_gpu, dsc_generators, device_string, n_samples)#,gumbel=True,tau=None)
    
    
    
    
    
    
    
    
    if not args.synthesise_w_cs:
    
        synthetic_samples_dict = generate_samples_to_dict_tripartite(dsc=dsc, has_gpu=has_gpu, dsc_generators=dsc_generators, device_string=device_string,n_samples=n_samples,use_gt_cspouse=False)
        joined_synthetic_data = samples_dict_to_df(dsc, synthetic_samples_dict, balance_labels=True,exact=False,extra_sample_frac=1.0,resample=True)
    
    
    
    
    
    
    
    
    if args.synthesise_w_cs:
    
    
    
        synthetic_samples_dict = generate_samples_to_dict_tripartite(dsc=dsc, has_gpu=has_gpu, dsc_generators=dsc_generators, device_string=device_string,n_samples=n_samples,use_gt_cspouse=True) #use cause spouse variables as antecedents rather than synthyesising from the generator
        joined_synthetic_data = samples_dict_to_df(dsc, synthetic_samples_dict, balance_labels=True,exact=False,extra_sample_frac=1.0,resample=False)
    
    
#def generate_samples_to_dict_tripartite(dsc, has_gpu, dsc_generators, device_string,n_samples=10000,gumbel=False,tau=None):
    
    
    #synthetic_samples_dict = generate_samples_to_dict(dsc, has_gpu, dsc_generators, device_string, n_samples,tau=0.5)
    #joined_synthetic_data = samples_dict_to_df(dsc, synthetic_samples_dict, balance_labels=True,exact=False,extra_sample_frac=1.0,resample=False)

    algo_spec = copy.deepcopy(master_spec['algo_spec'])
    algo_spec.set_index('algo_variant', inplace=True)

    synthetic_data_dir = algo_spec.loc[algo_variant].synthetic_data_dir

    
    ##set_trace()
    save_synthetic_data(joined_synthetic_data,d_n,s_i,master_spec,dspec,algo_spec,synthetic_data_dir)

    et = time.time()
    total_time = et - st
    total_time /= 60

    print('total time taken for n_iterations: {0}\t {1} minutes'.format(n_iterations, total_time))
    # dict of hyperparameters - parameters to be written as text within synthetic data plot later on
    hpms_dict = {'lr': args.lr,
                 'n_iterations': args.n_iterations,
                 'lab_bs': args.lab_bsize,
                 'ulab_bs': args.tot_bsize,
                 'nhidden_layer': args.nhidden_layer,
                 'neuron_in_hidden': args.n_neuron_hidden,
                 'use_bernoulli': args.use_bernoulli,
                 'time': total_time}
    
    
    
    if args.plot_synthetic_dist and s_i >=5:
        
        if dsc.feature_dim == 1:
            print('feature dim==1 so we are going to to attempt to plot synthetic v real data')
            plot_synthetic_and_real_data(hpms_dict, dsc, args, s_i, joined_synthetic_data, synthetic_data_dir, dspec,scale_for_indiv_plot=dspec.scale_for_indiv_plot,scale_for_cat_plot=dspec.scale_for_cat_plot)
            #plot_synthetic_and_real_data(hpms_dict, dsc, args, s_i, joined_synthetic_data, synthetic_data_dir, dspec,scale_for_cat_plot=0.3)
        else:
            # scols = [s.replace('_0', '') for s in joined_synthetic_data.columns]
            # joined_synthetic_data.columns = scols
            joined_synthetic_data.rename(columns={'Y_0': 'Y'}, inplace=True)
            # joined_synthetic_data=joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
            if 'y_given_x_bp' in dsc.merge_dat.columns:
                dsc.merge_dat = dsc.merge_dat[[c for c in joined_synthetic_data.columns]]
            if 'y_given_x_bp' in joined_synthetic_data.columns:
                joined_synthetic_data = joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
            synthetic_and_orig_data = pd.concat([dsc.merge_dat, joined_synthetic_data], axis=0,ignore_index=True)

            dsc.merge_dat = synthetic_and_orig_data
            dsc.d_n = d_n
            dsc.var_types = dsc.variable_types
            dsc.var_names = dsc.labels
            dsc.feature_varnames = dsc.feature_names
            dsc.s_i = s_i
            plot_2d_data_w_dag(dsc, s_i,synthetic_data_dir=synthetic_data_dir,scale_for_indiv_plot=dspec.scale_for_indiv_plot,subset_large_data=False)

            print('now plotting individual variables')
            print('skip plot individual for now')
            # for f in dsc.feature_names:
            #     for data_type in ['unlabelled', 'labelled', 'synthetic']:
            #         plot_2d_single_variable_data_type(dsc,
            #                                           s_i,
            #                                           data_type=data_type,
            #                                           variable_name=f,
            #                                           synthetic_data_dir=synthetic_data_dir)

            print('data plotted')


    elif s_i <5 and not args.ignore_plot_5:
        print('plotting data cos si = 0')
        
        dspec.scale_for_indiv_plot
        
        if dsc.feature_dim == 1:
            print('feature dim==1 so we are going to to attempt to plot synthetic v real data')
            plot_synthetic_and_real_data(hpms_dict, dsc, args, s_i, joined_synthetic_data, synthetic_data_dir, dspec,scale_for_indiv_plot=dspec.scale_for_indiv_plot,scale_for_cat_plot=dspec.scale_for_cat_plot)

            
        else:
                
            # scols = [s.replace('_0', '') for s in joined_synthetic_data.columns]
            # joined_synthetic_data.columns = scols
            joined_synthetic_data.rename(columns={'Y_0': 'Y'}, inplace=True)
            # joined_synthetic_data=joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
            if 'y_given_x_bp' in dsc.merge_dat.columns:
                dsc.merge_dat = dsc.merge_dat[[c for c in joined_synthetic_data.columns]]
            if 'y_given_x_bp' in joined_synthetic_data.columns:
                joined_synthetic_data = joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
            synthetic_and_orig_data = pd.concat([dsc.merge_dat, joined_synthetic_data], axis=0,ignore_index=True)
            
            #huffle it....
            synthetic_and_orig_data.reset_index(inplace=True,drop=True)
            
            synthetic_and_orig_data=synthetic_and_orig_data.sample(frac=1.0)
            
            #print('before shuffle')
            #print(synthetic_and_orig_data.head(5))
            
            #s_idx=[i for i in synthetic_and_orig_data.index]
            
            #from IPython.core.debugger import set_trace
            
            ##set_trace()
            
            
            types_orders={'synthetic':4,'test':2,'unlabelled':3,'labelled':0,'validation':1}

            
            
            
            #s = synthetic_and_orig_data['type'].apply(lambda x: types_orders[x])
            #s.sort_values()
            
            
            
            #df.set_index(s.index).sort()
            
            synthetic_and_orig_data=synthetic_and_orig_data.sort_values(by=['type'], key=lambda x: x.map(types_orders))
            
            
            #s_idx=s_idx[torch.randperm(len(s_idx)).cpu().numpy()]
            
            #synthetic_and_orig_data=synthetic_and_orig_data.loc[s_idx]
            #print('after shuffle')
            
            #print(synthetic_and_orig_data.head(5))

            dsc.merge_dat = synthetic_and_orig_data
            dsc.d_n = d_n
            dsc.var_types = dsc.variable_types
            dsc.var_names = dsc.labels
            dsc.feature_varnames = dsc.feature_names
            dsc.s_i = s_i
            plot_2d_data_w_dag(dsc, s_i,synthetic_data_dir=synthetic_data_dir,subset_large_data=False,scale_for_indiv_plot=dspec.scale_for_indiv_plot)#,scale_for_cat_plot=dspec.scale_for_cat_plot)

            print('now plotting individual variables')
            print('skip plot individual for now')
            # for f in dsc.feature_names:
            #     for data_type in ['unlabelled', 'labelled', 'synthetic']:
            #         plot_2d_single_variable_data_type(dsc,
            #                                           s_i,
            #                                           data_type=data_type,
            #                                           variable_name=f,
            #                                           synthetic_data_dir=synthetic_data_dir)

            print('data plotted')
            







    else:
        print(f'data not plotted for si {args.s_i} dn {args.d_n}, plot ignore 5 is {args.ignore_plot_5}')
            
            
        
    del dsc_generators
    
    del optimal_mods
    
    #del GumbelModuleCombined
    
    del old_dsc_generators
    
    del data_module
    
    del gumbel_module
    
    del trainer
    
    del dsc
    
    
    #del unlabelled
    
    
    del joined_synthetic_data
    
    del synthetic_samples_dict