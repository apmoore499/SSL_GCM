
#--d_n=synthdag_dset_3 --s_i=1 --n_iterations=100 --lr=1e-3 --use_single_si=False --ulab_bsize=128 --patience=2 --min_temp=0.1 --init_temp=1.0 --n_trials=1 --val_loss_criterion=labelled_bce_and_all_feat_mmd --lab_bsize=4 --balance=False

######
#
# NEW PROCEDURE FOR TRAINING CGAN WITH GUMBEL SOFTMAX FOR SIMULTANEOUS LEARNING OF ENTIRE CAUSAL GRAPH
#
######
import os

from torch.nn.functional import gumbel_softmax

import time
#from generative_models.gen_data_loaders import SSLDataModule_Unlabel_X
from collections import OrderedDict
st=time.time()

import sys
sys.path.append('generative_models')
sys.path.append('py')
sys.path.append('py/generative_models/')
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



args = parser.parse_args()

args.use_single_si=str_to_bool(args.use_single_si)
args.balance=str_to_bool(args.balance)
args.use_benchmark_generators=str_to_bool(args.use_benchmark_generators)
args.use_bernoulli=str_to_bool(args.use_bernoulli)
args.use_optimal_y_gen=str_to_bool(args.use_optimal_y_gen)
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


algo_variant='gumbel_disjoint'

#now we want to read in dataset_si
csi=master_spec['dataset_si'][dspec.d_n].values
candidate_si=csi[~np.isnan(csi)]
args.optimal_si_list = [int(s) for s in candidate_si]
if args.use_single_si==True: #so we want to use single si, not entire range
    #args.optimal_si_list=[args.optimal_si_list[args.s_i]]
    args.optimal_si_list = [args.s_i]
#now we are onto training
for k, s_i in enumerate(args.optimal_si_list):
    print('doing s_i: {0}'.format(k))
    args.st = time.time()

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

    # delete the possible connection
   #for c in causal_sources:
    #    for e in effect_variables:
    #        if dsc.dag.are_connected(c, e):
    #            dsc.dag.delete_edges([(c, e)])

    # actaully don't do that - remove causal dependence incorrect

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
            x_vals=all_x[cur_x_lab].to_numpy()
            #get median pwd
            median_pwd=get_median_pwd(torch.tensor(x_vals))
            #make generator for X
            curmod=0
            gen_x=Generator_X1(args.lr,
                               args.d_n,
                               s_i,
                               dspec.dn_log,
                               input_dim=dsc.feature_dim,
                               median_pwd=median_pwd,
                               num_hidden_layer=args.nhidden_layer,
                               middle_layer_size=args.n_neuron_hidden,
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
            x_vals=all_x[cur_x_lab].to_numpy()
            # get median pwd
            median_pwd = get_median_pwd(torch.tensor(x_vals))
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
            target_x_vals = all_dat[ldc].values
            median_pwd_target = get_median_pwd(torch.tensor(target_x_vals))

            ################################
            #  median pwd for conditional x
            ################################
            # match column names
            #ldc = [c for c in lab_dat.columns if any(c in cond_x_lab)]
            cond_x_vals = all_dat[cond_x_lab].values
            median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))


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
            median_pwd_target = get_median_pwd(torch.tensor(target_x_vals))

            ################################
            #  median pwd for conditional x
            ################################
            # match column names
            concat_cond_lab = []
            for c in conditional_feature_names:
                concat_cond_lab = concat_cond_lab + dsc.label_names_alphan[c]

            cond_x_vals = all_dat[concat_cond_lab].values
            median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))

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

    #set_trace()
    if type(label_name)=='list':
        label_idx = np.where(np.array(ordered_keys) == label_name[0])[0][0]
    else:
        label_idx = np.where(np.array(ordered_keys) == label_name)[0][0]

    # topological sort ie like,
    # XC -> Y -> XE
    unlabelled_keys=ordered_keys[:label_idx] # XC is unlabelled keys
    labelled_key=ordered_keys[label_idx] # Y is labelled key
    conditional_keys=ordered_keys[label_idx+1:] # XE is conditional keys





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

    #validation data
    all_validation_features = dsc.merge_dat[dsc.merge_dat.type == 'validation'][all_feat_names_sub]
    all_validation_label = dsc.merge_dat[dsc.merge_dat.type == 'validation'][labelled_key]


    # convert all_feat_names to desired
    i = [dsc.label_names_alphan[k] for k in conditional_keys]
    ck_sub = [item for sublist in i for item in sublist]




    #get median pairwise distances for mmd loss func
    #median pwd of labelled + unlabelled X in training set
    total_median_pwd=get_median_pwd(torch.Tensor(all_unlabelled_and_labelled.values))
    #median pwd of labelled X in training set only
    labelled_median_pwd = get_median_pwd(torch.Tensor(all_labelled_features.values))




    #get median pwd on label/unlabel datar

    ck_lab_ulab= dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])][ck_sub]
    ck_lab= dsc.merge_dat[dsc.merge_dat.type.isin(['labelled'])][ck_sub]

    #all_unlabelled_and_labelled = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])][all_feat_names_sub]

    total_median_pwd=get_median_pwd(torch.Tensor(ck_lab_ulab.values))
    #median pwd of labelled X in training set only
    labelled_median_pwd = get_median_pwd(torch.Tensor(ck_lab.values))



    

    #median pwd of each feature variable...
    median_pwd_dict={}
    for c in unlabelled_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_unlabelled_and_labelled[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.Tensor(cur_feature_vals.values))
        #store
        median_pwd_dict[c]=cur_mpwd

    for c in conditional_keys:
        #get dsc alpha labels
        a_labs=dsc.label_names_alphan[c]
        #dsc.label_names_alphan[c]
        #subset dataframe
        cur_feature_vals=all_unlabelled_and_labelled[a_labs]
        #get median pwd
        cur_mpwd=get_median_pwd(torch.Tensor(cur_feature_vals.values))
        #store
        median_pwd_dict[c]=cur_mpwd

    print('pausing here')

    # pull out generators which are causes of Y, store in ```causes_of_y```
    # for example, if we have:
    # X1 -> Y <- X2
    # pull out generators for X1, X2, stored in "conditional_feature_variables"

    causes_of_y=var_conditional_feature_names[labelled_key]

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

    ulab_dloader=DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled.values)),batch_size=args.lab_bsize,shuffle=True)
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
    all_keys = unlabelled_keys + [labelled_key] + conditional_keys

    # array of the min loss at each trial end.
    # only used if we experiment with training generator multiple trials from scratch
    mins_end_of_trial = []
    # list for saving optimal-performing model after each trial.
    # only used if we are doing multiple trials
    optimal_mods = []
    # all minimum values for each trial
    # only used if we are doing multiple trials
    all_mins=[]

    for k_n, v_i in enumerate(order_to_train[:len(unlabelled_keys)]):
    # create generators and train individually, in order
        source_edges = dsc.dag.es.select(_target=v_i)
        source_vertices = [s_e.source_vertex for s_e in source_edges]  # source vertex
        sv = [v.index for v in source_vertices]  # source idx
        vinfo_dict = {}  # reset this dict
        cur_variable = dsc.labels[v_i]
        source_variable = [dsc.labels[s] for s in sv]
        print('training on variable number: {0} of {1}'.format(k_n + 1, len(order_to_train)))
        print('+---------------------------------------+')
        print('cur_var_name : {0}\tsource_var_name(s): {1}'.format(cur_variable, source_variable))

        cond_lab = source_variable  # labels of conditinoal vars
        cond_vtype = [dsc.variable_types[l] for l in sv]  # types of conditional vars
        cond_idx = sv  # index of conditioanl vars in pandas df

        if len(sv) == 0 and dsc.variable_types[v_i] == 'feature':
            # if 'X' in dsc.labels[v_i]:
            cur_x_lab = dsc.label_names[v_i]
            # get the matching data...
            all_x = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
            # match column names
            # ldc=[c for c in all_x.columns if cur_x_lab in c]
            # subset
            x_vals = all_x[cur_x_lab].to_numpy()
            # get median pwd
            median_pwd = get_median_pwd(torch.tensor(x_vals))
            # make generator for X
            curmod = 0
            gen_x = Generator_X1(args.lr,
                                 args.d_n,
                                 s_i,
                                 dspec.dn_log,
                                 input_dim=dsc.feature_dim,
                                 median_pwd=median_pwd,
                                 num_hidden_layer=args.nhidden_layer,
                                 middle_layer_size=args.n_neuron_hidden,
                                 label_batch_size=args.lab_bsize,
                                 unlabel_batch_size=args.tot_bsize)

            tloader = SSLDataModule_Unlabel_X(dsc.merge_dat, target_x=cur_x_lab, batch_size=args.tot_bsize)
            model_name = create_model_name(dsc.labels[v_i], algo_variant)
            min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
                                                                   dspec.save_folder)  # returns max checkpoint

            if args.estop_mmd_type == 'val':
                estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
            elif args.estop_mmd_type == 'trans':
                estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
            callbacks = [min_mmd_checkpoint_callback, estop_cb]

            tb_logger = create_logger(model_name, d_n, s_i)
            trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)
            delete_old_saved_models(model_name, dspec.save_folder, s_i)

            trainer.fit(gen_x, tloader)  # train here

            mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)
            gen_x = type(gen_x).load_from_checkpoint(checkpoint_path=mod_names[0])  # loads correct model
            dsc_generators[dsc.labels[v_i]] = gen_x
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = []

        elif len(sv) == 0 and dsc.variable_types[v_i] == 'label':
            # make generator for Y
            yv = dsc.merge_dat[dsc.merge_dat.type == 'labelled'][dsc.labels[v_i]].mean()
            geny = Generator_Y(args.d_n, s_i, dspec.dn_log, yv)
            dsc_generators[dsc.labels[v_i]] = geny
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = []
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False

        #########
        # X2->
        # X1->  Y
        #########

        elif len(sv) > 0 and dsc.variable_types[v_i] == 'label':  # need to incorporate some conditional variables

            # get input dim from feature_dim and cond_vtype....
            num_features = np.sum([c == 'feature' for c in cond_vtype])
            feature_inputs = num_features * dsc.feature_dim  # features

            y_x1gen = Generator_Y_from_X1(args.lr,
                                          args.d_n,
                                          s_i,
                                          dspec.dn_log,
                                          input_dim=feature_inputs,
                                          output_dim=2,
                                          num_hidden_layer=args.nhidden_layer,
                                          middle_layer_size=args.n_neuron_hidden)

            concat_cond_lab = []
            for c in cond_lab:
                concat_cond_lab = concat_cond_lab + dsc.label_names_alphan[c]

            tloader = SSLDataModule_Y_from_X(dsc.merge_dat,
                                             tvar_name=cur_variable,
                                             cvar_name=concat_cond_lab,
                                             batch_size=args.lab_bsize)

            model_name = create_model_name(dsc.labels[v_i], algo_variant)
            cond_vars = ''.join(cond_lab)
            min_bce_chkpt_callback = return_chkpt_min_bce(model_name, dspec.save_folder)
            estop_cb = return_early_stop_cb_bce(patience=args.estop_patience)
            callbacks = [min_bce_chkpt_callback, estop_cb]
            tb_logger = create_logger(model_name, d_n, s_i)
            trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)
            delete_old_saved_models(model_name, dspec.save_folder, s_i)
            trainer.fit(y_x1gen, tloader)  # train here
            mod_names = return_saved_model_name(model_name, dspec.save_folder, d_n, s_i)
            if len(mod_names) > 1:
                print('error duplicate model names')
                assert 1 == 0
            y_x1gen = type(y_x1gen).load_from_checkpoint(checkpoint_path=mod_names[0])
            dsc_generators[dsc.labels[v_i]] = copy.deepcopy(y_x1gen)
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False

            # modify original merge_dat data frame to get y|x on unlabelled data
            # we can use this later if we want to set use_bernoulli==True
            # for using the marginal distribution
            dsc.merge_dat['y_given_x_bp'] = dsc.merge_dat[cur_variable]  # cur_variable must be Y in this case
            concat_cond_lab = []
            for c in cond_lab:
                concat_cond_lab = concat_cond_lab + dsc.label_names_alphan[c]

            # get unlabelled data
            odf = dsc.merge_dat
            odf_unlabelled = odf[odf.type == 'unlabelled']
            x_unlab = odf_unlabelled[concat_cond_lab]
            # convert to numpy
            x_unlab = x_unlab.values
            # convert to torch tensor
            x_unlab = torch.Tensor(x_unlab)
            # now do prediction
            y_softmax = get_softmax(dsc_generators[dsc.labels[v_i]](x_unlab))
            # 1st col as bernoulli parameter
            y_bern_p = y_softmax[:, 1]
            # sample from bernoulli dist
            y_hat_unlabelled = torch.bernoulli((y_bern_p))
            # convert to data frame
            y_df = pd.DataFrame(y_bern_p.cpu().detach().numpy())
            y_df.columns = ['y_given_x_bp']
            y_df['ulab_idx'] = odf_unlabelled.index.values
            y_df.set_index(['ulab_idx'], inplace=True)
            dsc.merge_dat['y_given_x_bp'].loc[y_df.index.values] = y_df['y_given_x_bp'].loc[y_df.index]
            dsc.causes_of_y = cond_lab


        # X from Y, ie Y->X
        elif len(sv) == 1 and dsc.variable_types[v_i] == 'feature' and cond_vtype == ['label']:
            cur_x_lab = dsc.labels[v_i]
            # get the matching data...
            all_vals = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
            # match column names
            # subset
            x_vals = all_vals[dsc.label_names[v_i]].values
            # get median pwd
            median_pwd = get_median_pwd(torch.tensor(x_vals))
            n_lab = dsc.merge_dat[dsc.merge_dat.type == 'labelled'].shape[0]
            n_ulab = x_vals.shape[0]

            # make gen
            x2_y_gen = Generator_X2_from_Y(args.lr,
                                           args.d_n,
                                           s_i,
                                           dspec.dn_log,
                                           input_dim=dsc.feature_dim + dsc.n_classes,
                                           output_dim=dsc.feature_dim,
                                           median_pwd=median_pwd,
                                           num_hidden_layer=args.nhidden_layer,
                                           middle_layer_size=args.n_neuron_hidden,
                                           n_lab=n_lab,
                                           n_ulab=n_ulab,
                                           label_batch_size=args.lab_bsize,
                                           unlabel_batch_size=args.tot_bsize)  # this is it
            # get data loader
            tloader = SSLDataModule_X_from_Y(orig_data_df=dsc.merge_dat,
                                             tvar_name=dsc.label_names[v_i],
                                             cvar_name=cond_lab,
                                             cvar_type='label',
                                             labelled_batch_size=args.lab_bsize,
                                             unlabelled_batch_size=args.tot_bsize,
                                             **vinfo_dict)
            model_name = create_model_name(dsc.labels[v_i], algo_variant)
            cond_vars = ''.join(cond_lab)
            min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name, dspec.save_folder)
            if args.estop_mmd_type == 'val':
                estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
            elif args.estop_mmd_type == 'trans':
                estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
            callbacks = [min_mmd_checkpoint_callback, estop_cb]
            tb_logger = create_logger(model_name, d_n, s_i)
            trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, args.n_iterations)
            delete_old_saved_models(model_name, dspec.save_folder, s_i)
            trainer.fit(x2_y_gen, tloader)
            mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)

            if len(mod_names) > 1:
                print(mod_names)
                print('error duplicate model names')
                assert 1 == 0
            elif len(mod_names) == 1:
                x2_y_gen = type(x2_y_gen).load_from_checkpoint(checkpoint_path=mod_names[0])
            else:
                assert 1 == 0

            # training complete, save to list
            dsc_generators[dsc.labels[v_i]] = x2_y_gen
            dsc_generators[dsc.labels[v_i]].conditional_on_label = True
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = []


        elif (len(sv) >= 1) and (dsc.variable_types[v_i] == 'feature') and (
        np.all([c == 'feature' for c in cond_vtype])):
            num_features = np.sum([c == 'feature' for c in cond_vtype])
            feature_inputs = num_features * dsc.feature_dim  # features
            cur_x_lab = dsc.label_names[v_i]
            cond_x_lab = cond_lab
            # get the matching data...
            all_dat = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]

            if len(cur_x_lab) == 1 and type(cur_x_lab) == list:
                cur_x_lab = cur_x_lab[0]
            ##############################
            #    median pwd for target x
            ##############################
            # match column names
            ldc = [c for c in all_dat.columns if cur_x_lab in c]
            target_x_vals = all_dat[ldc].values
            median_pwd_target = get_median_pwd(torch.tensor(target_x_vals))

            ################################
            #  median pwd for conditional x
            ################################
            # match column names
            # ldc = [c for c in lab_dat.columns if any(c in cond_x_lab)]
            cond_x_vals = all_dat[cond_x_lab].values
            median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))

            genx_x = Generator_X_from_X(args.lr,
                                        args.d_n,
                                        s_i,
                                        dspec.dn_log,
                                        input_dim=dsc.feature_dim + len(cond_x_lab) * dsc.feature_dim,
                                        output_dim=dsc.feature_dim,
                                        median_pwd_tx=median_pwd_target,
                                        median_pwd_cx=median_pwd_cond,
                                        num_hidden_layer=args.nhidden_layer,
                                        middle_layer_size=args.n_neuron_hidden,
                                        label_batch_size=args.lab_bsize,
                                        unlabel_batch_size=args.tot_bsize)

            tloader = SSLDataModule_X_from_X(orig_data_df=dsc.merge_dat,
                                             tvar_names=cur_x_lab,
                                             cvar_names=cond_x_lab,
                                             cvar_types=cond_vtype,
                                             labelled_batch_size=args.lab_bsize,
                                             unlabelled_batch_size=args.tot_bsize, **vinfo_dict)
            model_name = create_model_name(dsc.labels[v_i], algo_variant)
            cond_vars = ''.join(cond_lab)
            min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
                                                                   dspec.save_folder)  # returns max checkpoint
            if args.estop_mmd_type == 'val':
                estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
            elif args.estop_mmd_type == 'trans':
                estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
            callbacks = [min_mmd_checkpoint_callback, estop_cb]
            tb_logger = create_logger(model_name, d_n, s_i)
            trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)
            # deletions
            delete_old_saved_models(model_name, dspec.save_folder, s_i)
            # training
            trainer.fit(genx_x, tloader)
            mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)
            if len(mod_names) > 1:
                print(mod_names)
                print('error duplicate model names')
                assert 1 == 0
            elif len(mod_names) == 1:
                genx_x = type(genx_x).load_from_checkpoint(checkpoint_path=mod_names[0])
            else:
                assert 1 == 0
            # training complete, save to list
            dsc_generators[dsc.labels[v_i]] = copy.deepcopy(genx_x)
            dsc_generators[dsc.labels[v_i]].conditional_on_label = False
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab

            # Y  ->  X2
            # X1 ->
        elif len(sv) > 1 and dsc.variable_types[v_i] == 'feature' and 'label' in cond_vtype:

            conditional_feature_names = []
            label_name = []
            for cl, ct in zip(cond_lab, cond_vtype):
                if ct == 'feature':
                    conditional_feature_names.append(cl)
                if ct == 'label':
                    label_name.append(cl)

            concat_cond_lab = []
            for c in conditional_feature_names:
                concat_cond_lab = concat_cond_lab + dsc.label_names_alphan[c]

            # get target variable names if multidimensional
            c = dsc.labels[v_i]
            cur_x_lab = dsc.labels[v_i]
            lab_dat = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled'])]
            all_dat = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
            n_lab = lab_dat.shape[0]
            n_ulab = all_dat.shape[0]

            ##############################
            #    median pwd for target x
            ##############################
            # match column names
            ldc = [c for c in all_dat.columns if cur_x_lab in c]
            target_x_vals = all_dat[ldc].values
            median_pwd_target = get_median_pwd(torch.tensor(target_x_vals))

            ################################
            #  median pwd for conditional x
            ################################
            # match column names
            cond_x_vals = all_dat[concat_cond_lab].values
            median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))

            input_dim = dsc.feature_dim + cond_x_vals.shape[1] + dsc.n_classes
            output_dim = dsc.feature_dim

            genx2_yx1 = Generator_X2_from_YX1(args.lr,
                                              args.d_n,
                                              s_i,
                                              dspec.dn_log,
                                              input_dim=input_dim,
                                              output_dim=output_dim,
                                              median_pwd_tx=median_pwd_target,
                                              median_pwd_cx=median_pwd_cond,
                                              num_hidden_layer=args.nhidden_layer,
                                              middle_layer_size=args.n_neuron_hidden,
                                              n_lab=n_lab,
                                              n_ulab=n_ulab,
                                              label_batch_size=args.lab_bsize,
                                              unlabel_batch_size=args.tot_bsize)

            if dsc.feature_dim > 1:
                cur_x_lab = dsc.label_names_alphan[cur_x_lab]
            else:
                cur_x_lab = cur_x_lab

            tloader = SSLDataModule_X2_from_Y_and_X1(
                orig_data_df=dsc.merge_dat,
                tvar_names=ldc,  # change from cur_x_lab
                cvar_names=concat_cond_lab,
                label_var_name=label_name,
                labelled_batch_size=args.lab_bsize,
                unlabelled_batch_size=args.tot_bsize,
                use_bernoulli=args.use_bernoulli,
                causes_of_y=None,
                **vinfo_dict)

            # train`
            model_name = create_model_name(dsc.labels[v_i], algo_variant)
            cond_vars = ''.join(cond_lab)
            min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name, dspec.save_folder)
            if args.estop_mmd_type == 'val':
                estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
            elif args.estop_mmd_type == 'trans':
                estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
            callbacks = [min_mmd_checkpoint_callback, estop_cb]
            tb_logger = create_logger(model_name, args.d_n, s_i)
            trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)
            delete_old_saved_models(model_name, dspec.save_folder, s_i)  # delete old
            trainer.fit(genx2_yx1, tloader)  # train here

            mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)
            if len(mod_names) > 1:
                print('error duplicate model names')
                assert 1 == 0
            elif len(mod_names) == 1:
                genx2_yx1 = type(genx2_yx1).load_from_checkpoint(checkpoint_path=mod_names[0])
            else:
                assert 1 == 0

            dsc_generators[dsc.labels[v_i]] = copy.deepcopy(genx2_yx1)
            dsc_generators[dsc.labels[v_i]].conditional_on_label = True
            dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab

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

        all_parameters_labelled = [] # parameters of all generators including generator for Y

        for k in [labelled_key] + conditional_keys:
            all_parameters_labelled = all_parameters_labelled + list(dsc_generators[k].parameters())

        combined_labelled_optimiser = optim.Adam(all_parameters_labelled, lr=args.lr) #reset labelled opt

        all_parameters_unlabelled = []

        for k in conditional_keys:
            all_parameters_unlabelled = all_parameters_unlabelled + list(dsc_generators[k].parameters())

        combined_unlabelled_optimiser = optim.Adam(all_parameters_unlabelled, lr=args.lr)  #reset unlabelled opt

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

        # tensor for validating y BCE loss
        y_features = dsc.merge_dat[dsc.merge_dat.type == 'labelled'][causes_of_y_feat_names].values

        y_truelabel = dsc.merge_dat[dsc.merge_dat.type == 'labelled'][[dsc.label_var]].values.flatten()
        y_truelabel = torch.tensor(y_truelabel)
        y_truelabel = torch.nn.functional.one_hot(y_truelabel).float()


        if has_gpu:

            val_dloader_unlabelled = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled_and_labelled.values).cuda()),
                                            batch_size=all_unlabelled_and_labelled.shape[0],
                                            shuffle=False)

            val_dloader_labelled = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_validation_features.values).cuda(),
                                                                            torch.Tensor(all_validation_label.values).cuda()),
                                                                            batch_size=all_labelled_features.shape[0],
                                                                            shuffle=False)

            y_features = torch.tensor(y_features).float().cuda() #to predict Y from X and validate BCE loss
            y_truelabel = y_truelabel.cuda()
        else:
            val_dloader_unlabelled = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled_and_labelled.values)),
                                            batch_size=all_unlabelled_and_labelled.shape[0],
                                            shuffle=False)

            val_dloader_labelled = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_validation_features.values),
                                                                            torch.Tensor(all_validation_label.values)),
                                                                            batch_size=all_labelled_features.shape[0],
                                                                            shuffle=False)

            y_features = torch.tensor(y_features).float()  #to predict Y from X and validate BCE loss (NO cuda)
            y_truelabel = y_truelabel
        early_end=False
        epoch = 0
        templist = np.linspace(0.99,0.1,160)
        #templist = np.linspace(0.99, 0.9, 10)
        converged=False




        from torch.utils.tensorboard import SummaryWriter

        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter('lightning_logs/gumbel_training_mmd')

        for t_iter,temp in enumerate(templist):
            if converged==False:

                # ...log the running loss
                writer.add_scalar('temp',
                                  temp,t_iter)


                if has_gpu:

                    ulab_dloader = DataLoader(
                        torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled.values).cuda()),
                        batch_size=args.tot_bsize,
                        shuffle=True)

                    lab_dloader = DataLoader(
                        torch.utils.data.TensorDataset(torch.Tensor(all_labelled_features.values).cuda(),
                                                       torch.Tensor(all_labelled_label.values).cuda()),
                        batch_size=args.lab_bsize,
                        shuffle=True)

                else:

                    ulab_dloader = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled.values)),
                                              batch_size=args.tot_bsize,
                                              shuffle=True)

                    lab_dloader = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_labelled_features.values),
                                                                            torch.Tensor(all_labelled_label.values)),
                                             batch_size=args.lab_bsize,
                                             shuffle=True)

                temps_tried.append(temp)
                current_optimal_mods_dict={}
                n_labelled = all_labelled_features.shape[0]
                n_unlabelled = all_unlabelled.shape[0]


                ratio_n_unlabelled=n_labelled/n_unlabelled #ratio of labelled to unlabelled < 1
                ratio_n_labelled=int(n_unlabelled/n_labelled) #ratio of unlabelled to labelled > 1

                #train mode
                dsc_generators[labelled_key].train()
                for k in conditional_keys:
                    dsc_generators[k].train()

                for batch_idx, d in enumerate(lab_dloader):

                    cur_batch_features, cur_batch_label = d
                    # split into features and label for this instance
                    # instantiate our ancestor dict that will be used for current batch
                    current_ancestor_dict = {}
                    # variables where Y is not ancestor
                    # append them to current_ancestor_dict
                    for k in unlabelled_keys:
                        # retrieve potential multidm
                        associated_names = dsc.label_names_alphan[k]
                        # retrieve index of feature
                        current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                        # put into ancestor dict
                        current_ancestor_dict[k] = cur_batch_features[:, current_feature_idx].reshape(
                            (-1, len(current_feature_idx)))  # maintain orig shape
                    # generate label first, and gumbel it
                    input_for_y = cur_batch_features[:, causes_of_y_idx_dl]
                    y_generated = dsc_generators[labelled_key].forward(input_for_y)
                    #hard=False
                    y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=False,
                                                                          tau=temp)  # I think use softmax label here?
                    # now put into ancestor dictionary
                    current_ancestor_dict[labelled_key] = torch.nn.functional.one_hot(y_gumbel_softmax.argmax(1),num_classes=2)
                    # now loop thru rest
                    # each k in this instance is for each feature in causal graph
                    for k in conditional_keys:
                        # get our generator please
                        k_has_label = dsc_generators[k].conditional_on_label
                        extra_causes = dsc_generators[k].conditional_feature_names
                        # get the idx of causes...
                        if k_has_label:
                            cur_feature_inputs = extra_causes +[labelled_key]
                        else:
                            cur_feature_inputs = extra_causes
                        # perform list comprehension to extract variables
                        cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
                        cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                        # then concatenate the variables
                        generator_input = torch.cat(cur_feature_inputs_lc, 1)
                        # then add some noise
                        if has_gpu:
                            current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim),device='cuda:0')
                        else:
                            current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim))

                        # then concatenate the noise
                        generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
                        # then predict
                        predicted_value = dsc_generators[k].forward(generator_input_w_noise)
                        # and put this one into ancestor dict
                        current_ancestor_dict[k] = predicted_value
                        # and pause here for my lord
                        # calculate joint mmd between observable X and generated X
                        # pull out features from current_ancestor_dict
                    cur_ancestors = [current_ancestor_dict[f] for f in all_feat_names]
                    cur_ancestors = tuple(cur_ancestors)
                    batch_estimate_features = torch.cat(cur_ancestors, 1)
                    batch_estimate_label = current_ancestor_dict[labelled_key]
                    cur_batch_label_one_hot = torch.nn.functional.one_hot(cur_batch_label.long(), 2)
                    # pull out features in current batch as given by data loader
                    # try modify median pwd
                    cur_median_pwd = get_median_pwd(cur_batch_features)
                    sigma_list = [labelled_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]

                    labelled_loss_array=[]


                    #get our BCE loss

                    cb = torch.nn.BCEWithLogitsLoss()
                    cbl_oh = cur_batch_label_one_hot.float()
                    loss_lab_bce = cb(y_gumbel_softmax, cbl_oh)

                    #labelled_loss = mix_rbf_mmd2_joint(batch_estimate_features,
                    #                                   cur_batch_features,
                    #                                   y_gumbel_softmax,
                    #                                   cur_batch_label_one_hot,
                    #                                   sigma_list=sigma_list)

                    #labelled_loss_array.append(labelled_loss)

                    # get loss on individual feature variables...
                    feature_y_losses = []
                    for c in unlabelled_keys:
                        associated_names = dsc.label_names_alphan[c]
                        # retrieve index of feature
                        current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                        # put into ancestor dict
                        current_batch_ground_truth = cur_batch_features[:, current_feature_idx].reshape(
                            (-1, dsc.feature_dim))
                        # get sigma list for current variable
                        sigma_list = [median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
                        # calculate mmd loss
                        #mloss = mix_rbf_mmd2(current_ancestor_dict[c], current_batch_ground_truth, sigma_list)
                        #mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c],
                        #                           current_batch_ground_truth,
                        #                           y_gumbel_softmax,
                        #                           cur_batch_label_one_hot,
                        #                           sigma_list=sigma_list)
                        #feature_y_losses.append(mloss)

                    for c in conditional_keys:
                        associated_names = dsc.label_names_alphan[c]
                        # retrieve index of feature
                        current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                        # put into ancestor dict
                        current_batch_ground_truth = cur_batch_features[:, current_feature_idx].reshape(
                            (-1, dsc.feature_dim))
                        # get sigma list for current variable
                        sigma_list = [median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
                        # calculate mmd loss
                        # mloss = mix_rbf_mmd2(current_ancestor_dict[c], current_batch_ground_truth, sigma_list)

                        mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c],
                                                   current_batch_ground_truth,
                                                   y_gumbel_softmax,
                                                   cur_batch_label_one_hot,
                                                   sigma_list=sigma_list)
                        feature_y_losses.append(mloss)

                    for m in feature_y_losses:
                        labelled_loss_array.append(m)
                        #labelled_loss+= m

                    total_labelled_loss=sum(labelled_loss_array)#.mean()
                    n_loss_terms=len(labelled_loss_array)
                    #scale by batch size and whether labelled or unlabelled
                    labelled_loss=total_labelled_loss*cur_batch_features.shape[0]/n_labelled/n_loss_terms

                    #labelled_loss=total_labelled_loss

                    combined_labelled_optimiser.zero_grad()
                    labelled_loss.backward()
                    combined_labelled_optimiser.step()

                    # ...log the running loss
                    writer.add_scalar('train_labelled_mmd_loss',
                                      labelled_loss, t_iter)

                # here, we calculate various validation losses..unclear which criterion is best

                # mmd loss loss on features x in training set
                for batch_idx, d in enumerate(ulab_dloader):
                    # code go here...
                    cur_batch = d[0]
                    # instantiate our ancestor dict that will be used for current batch
                    current_ancestor_dict = {}
                    # variables where Y is not ancestor
                    # append them to current_ancestor_dict
                    for k in unlabelled_keys:
                        associated_names = dsc.label_names_alphan[k]
                        # retrieve index of feature
                        current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                        # put into ancestor dict
                        current_ancestor_dict[k] = cur_batch[:, current_feature_idx].reshape((-1, dsc.feature_dim))  # maintain orig shape
                    # generate label first, and gumbel it
                    input_for_y = cur_batch[:, causes_of_y_idx_dl]
                    y_generated = dsc_generators[labelled_key].forward(input_for_y)

                    y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated,hard=True,tau=temp)
                    # anneal our way out of the temperature
                    # now put into ancestor dictionary
                    current_ancestor_dict[labelled_key] = y_gumbel_softmax
                    # now loop thru rest
                    # each k in this instance is for each feature in causal graph
                    for k in conditional_keys:
                        # get our generator please
                        k_has_label = dsc_generators[k].conditional_on_label
                        extra_causes = dsc_generators[k].conditional_feature_names
                        # get the idx of causes...
                        if k_has_label:
                            cur_feature_inputs = extra_causes + [labelled_key]
                        else:
                            cur_feature_inputs = extra_causes
                        # perform list comprehension to extract variables
                        cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
                        cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                        # then concatenate the variables
                        generator_input = torch.cat(cur_feature_inputs_lc, 1)
                        # then add some noise
                        if has_gpu:
                            current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim),device='cuda:0')
                        else:
                            current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim))
                        #current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim))
                        # then concatenate the noise
                        generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
                        # then predict
                        predicted_value = dsc_generators[k].forward(generator_input_w_noise)
                        # and put this one into ancestor dict
                        current_ancestor_dict[k] = predicted_value
                        # and pause here for my lord
                        # calculate joint mmd between observable X and generated X
                        # pull out features from current_ancestor_dict
                    cur_ancestors = [current_ancestor_dict[f] for f in all_feat_names]
                    cur_ancestors = tuple(cur_ancestors)
                    batch_estimate = torch.cat(cur_ancestors, 1)
                    # pull out features in current batch as given by data loader
                    sigma_list = [total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]

                    unlabelled_loss_array=[]

                    individual_feat_mmd_losses=[]
                    for c in conditional_keys:
                        associated_names = dsc.label_names_alphan[c]
                        # retrieve index of feature
                        current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                        # put into ancestor dict
                        current_batch_ground_truth = cur_batch[:, current_feature_idx].reshape((-1, dsc.feature_dim))
                        #get sigma list for current variable
                        sigma_list=[median_pwd_dict[c] * i for i in [0.125,0.25,0.5,1,2]]
                        #calculate mmd loss
                        mloss = mix_rbf_mmd2(current_ancestor_dict[c], current_batch_ground_truth, sigma_list)
                        individual_feat_mmd_losses.append(mloss)

                    #now add into total loss
                    for m in individual_feat_mmd_losses:
                        unlabelled_loss_array.append(m)

                    joint_feat_mmd_losses=[]
                    #get loss on joint feature variables...
                    for ev in conditional_keys: #effect of Y
                        for cv in unlabelled_keys: #cause of Y
                            #get effect v
                            associated_names = dsc.label_names_alphan[ev]
                            # retrieve index of feature
                            current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                            # put into ancestor dict
                            current_batch_ground_truth = cur_batch[:, current_feature_idx].reshape((-1, dsc.feature_dim))

                            sigma_list_ev=[median_pwd_dict[ev] * i for i in [0.125,0.25,0.5,1,2]]
                            estimate_ev=current_ancestor_dict[ev]
                            true_ev=current_batch_ground_truth

                            sigma_list_cv=[median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]]
                            estimate_cv=current_ancestor_dict[cv]
                            true_cv=current_ancestor_dict[cv]

                            joint_mmd_loss =mix_rbf_mmd2_joint_regress( estimate_ev,
                                                                        true_ev,
                                                                        estimate_cv,
                                                                        true_cv,
                                                                        sigma_list=sigma_list_ev,
                                                                        sigma_list1=sigma_list_cv)

                            joint_feat_mmd_losses.append(joint_mmd_loss)


                    for m in joint_feat_mmd_losses:
                        unlabelled_loss_array.append(m)

                    #take mean of unlabelled_loss
                    unlabelled_loss_total=sum(unlabelled_loss_array)
                    n_loss_terms=len(unlabelled_loss_array)

                    #scale loss by batch size and number of unlabelled cases
                    unlabelled_loss = unlabelled_loss_total * cur_batch.shape[0] / (n_unlabelled)/n_loss_terms

                    combined_unlabelled_optimiser.zero_grad()
                    unlabelled_loss.backward()
                    combined_unlabelled_optimiser.step()

                    # ...log the running loss
                    writer.add_scalar('train_unlabelled_mmd_loss',
                                      unlabelled_loss, t_iter)

                # eval mode
                dsc_generators[labelled_key].eval()
                for k in conditional_keys:
                    dsc_generators[k].eval()

                with torch.no_grad():
                    # mmd loss on unlabelled features in training set, not using label
                    # features drawn from labelled + unlabelled data
                    for batch_idx, d in enumerate(val_dloader_unlabelled):
                        unlabelled_val_losses=[]
                        cur_batch = d[0]
                        current_ancestor_dict = {}
                        # variables where Y is not ancestor
                        # append them to current_ancestor_dict
                        for k in unlabelled_keys:
                            associated_names = dsc.label_names_alphan[k]
                            # retrieve index of feature
                            current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                            # put into ancestor dict
                            current_ancestor_dict[k] = cur_batch[:, current_feature_idx].reshape(
                                (-1, dsc.feature_dim))  # maintain orig shape
                        # generate label first, and gumbel it
                        input_for_y = cur_batch[:, causes_of_y_idx_dl]
                        y_generated = dsc_generators[labelled_key].forward(input_for_y)
                        y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=True,tau=temp)#,tau=temp)
                        # now put into ancestor dictionary
                        current_ancestor_dict[labelled_key] = y_gumbel_softmax

                        # now loop thru rest
                        # each k in this instance is for each feature in causal graph
                        for k in conditional_keys:
                            # get our generator please
                            k_has_label = dsc_generators[k].conditional_on_label
                            extra_causes = dsc_generators[k].conditional_feature_names
                            # get the idx of causes...
                            if k_has_label:
                                cur_feature_inputs =  extra_causes + [labelled_key]
                            else:
                                cur_feature_inputs = extra_causes
                            # perform list comprehension to extract variables
                            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
                            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                            # then concatenate the variables
                            generator_input = torch.cat(cur_feature_inputs_lc, 1)
                            # then add some noise
                            if has_gpu:
                                current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim),device='cuda:0')
                            else:
                                current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim))
                            # then concatenate the noise
                            generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
                            # then predict
                            predicted_value = dsc_generators[k].forward(generator_input_w_noise)
                            # and put this one into ancestor dict
                            current_ancestor_dict[k] = predicted_value
                            # and pause here for my lord
                            # calculate joint mmd between observable X and generated X
                            # pull out features from current_ancestor_dict
                        cur_ancestors = [current_ancestor_dict[f] for f in all_feat_names]
                        cur_ancestors = tuple(cur_ancestors)
                        #batch_estimate_features = torch.cat(cur_ancestors, 1)
                        #batch_estimate_label = current_ancestor_dict[labelled_key]
                        cur_batch_label_one_hot = torch.nn.functional.one_hot(cur_batch_label.long(), 2)
                        # pull out features in current batch as given by data loader
                        # try modify median pwd
                        cur_median_pwd = get_median_pwd(cur_batch_features)
                        # sigma_list = [total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
                        batch_estimate = torch.cat(cur_ancestors,1)
                        # pull out features in current batch as given by data loader
                        sigma_list = [total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
                        
                        if batch_estimate.shape[0]>11000:
                            rp=torch.randperm(batch_estimate.shape[0])[:11000]
                        
                            loss_val_ulab = mix_rbf_mmd2(batch_estimate[rp], cur_batch[rp], sigma_list)
                            
                        else:
                            loss_val_ulab = mix_rbf_mmd2(batch_estimate, cur_batch, sigma_list)
                            

                        # ...log the running loss
                        writer.add_scalar('validation_unlabelled_mmd_loss',
                                          loss_val_ulab, t_iter)


                        unlabelled_val_losses.append(loss_val_ulab)

                        # get loss on individual feature variables...
                        individual_feat_mmd_losses = []
                        for c in conditional_keys:
                            associated_names = dsc.label_names_alphan[c]
                            # retrieve index of feature
                            current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                            # put into ancestor dict
                            current_batch_ground_truth = cur_batch[:, current_feature_idx].reshape((-1, dsc.feature_dim))
                            # get sigma list for current variable
                            sigma_list = [median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
                            # calculate mmd loss
                            if current_ancestor_dict[c].shape[0]>11000:
                                rp=torch.randperm(current_ancestor_dict[c].shape[0])[:11000]
                                mloss = mix_rbf_mmd2(current_ancestor_dict[c][rp], current_batch_ground_truth[rp], sigma_list)
                                
                            else:
                            
                            
                                mloss = mix_rbf_mmd2(current_ancestor_dict[c], current_batch_ground_truth, sigma_list)
                                
                            individual_feat_mmd_losses.append(mloss)


                        for m in individual_feat_mmd_losses:
                            unlabelled_val_losses.append(m)

                        joint_feat_mmd_losses=[]

                        #get loss on joint feature variables...
                        for ev in conditional_keys: #effect of Y
                            for cv in unlabelled_keys: #cause of Y
                                #get effect v
                                associated_names = dsc.label_names_alphan[ev]
                                # retrieve index of feature
                                current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                                # put into ancestor dict
                                current_batch_ground_truth = cur_batch[:, current_feature_idx].reshape((-1, dsc.feature_dim))

                                sigma_list_ev=[median_pwd_dict[ev] * i for i in [0.125,0.25,0.5,1,2]]
                                estimate_ev=current_ancestor_dict[ev]
                                true_ev=current_batch_ground_truth

                                #get cause c
                                sigma_list_cv=[median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]]
                                estimate_cv=current_ancestor_dict[cv]
                                true_cv=current_ancestor_dict[cv]
                                
                                
                                if estimate_ev.shape[0]>11000:
                                    
                                    rp=torch.randperm(estimate_ev.shape[0])[:11000]
                                    
                                    joint_mmd_loss =mix_rbf_mmd2_joint_regress( estimate_ev[rp],
                                                                                true_ev[rp],
                                                                                estimate_cv[rp],
                                                                                true_cv[rp],
                                                                                sigma_list=sigma_list_ev,
                                                                                sigma_list1=sigma_list_cv)
                                else:
                                    
                                

                                    joint_mmd_loss =mix_rbf_mmd2_joint_regress( estimate_ev,
                                                                                true_ev,
                                                                                estimate_cv,
                                                                                true_cv,
                                                                                sigma_list=sigma_list_ev,
                                                                                sigma_list1=sigma_list_cv)

                                joint_feat_mmd_losses.append(joint_mmd_loss)

                        for m in joint_feat_mmd_losses:
                            unlabelled_val_losses.append(m)

                    # joint mmd loss on labelled data, including label y
                    for batch_idx, d in enumerate(val_dloader_labelled):
                        labelled_val_losses=[]
                        # code go here...
                        cur_batch_features, cur_batch_label = d
                        # split into features and label for this instance
                        # instantiate our ancestor dict that will be used for current batch
                        current_ancestor_dict = {}
                        # variables where Y is not ancestor
                        # append them to current_ancestor_dict
                        for k in unlabelled_keys:
                            # retrieve potential multidm
                            associated_names=dsc.label_names_alphan[k]
                            # retrieve index of feature
                            current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                            # put into ancestor dict
                            current_ancestor_dict[k] = cur_batch_features[:, current_feature_idx].reshape((-1, len(current_feature_idx)))  # maintain orig shape
                        # generate label first, and gumbel it
                        input_for_y = cur_batch_features[:, causes_of_y_idx_dl]
                        y_generated = dsc_generators[labelled_key].forward(input_for_y)
                        # y_generated consists of un-normalised probabilities
                        # pass thru gumbel to generate labels

                        y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated,hard=True,tau=temp)
                        current_ancestor_dict[labelled_key] = y_gumbel_softmax

                        # now loop thru rest
                        # each k in this instance is for each feature in causal graph
                        for k in conditional_keys:
                            # get our generator please
                            k_has_label = dsc_generators[k].conditional_on_label
                            extra_causes = dsc_generators[k].conditional_feature_names
                            # get the idx of causes
                            if k_has_label:
                                cur_feature_inputs = extra_causes +[labelled_key]
                            else:
                                cur_feature_inputs = extra_causes
                            # perform list comprehension to extract variables
                            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
                            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                            # then concatenate the variables
                            generator_input = torch.cat(cur_feature_inputs_lc, 1)
                            # then add some noise
                            if has_gpu:
                                current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim),device='cuda:0')
                            else:
                                current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim))

                            #current_noise = torch.randn((generator_input.shape[0], dsc.feature_dim))
                            # then concatenate the noise
                            generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
                            # then predict
                            predicted_value = dsc_generators[k].forward(generator_input_w_noise)
                            # and put this one into ancestor dict
                            current_ancestor_dict[k] = predicted_value

                        #get ancestors of current variable
                        cur_ancestors = [current_ancestor_dict[f] for f in all_feat_names]
                        cur_ancestors = tuple(cur_ancestors)
                        batch_estimate_features = torch.cat(cur_ancestors, 1)
                        batch_estimate_label=current_ancestor_dict[labelled_key]
                        cur_batch_label_one_hot=torch.nn.functional.one_hot(cur_batch_label.long(),2)
                        # pull out features in current batch as given by data loader

                        sigma_list = [total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
                        
                        if batch_estimate_features.shape[0]>11000:
                            
                            rp=torch.randperm(batch_estimate_features.shape[0])[:11000]
                            
                            loss_val_lab = mix_rbf_mmd2_joint(batch_estimate_features[rp],
                                                            cur_batch_features[rp],
                                                            y_gumbel_softmax[rp],
                                                            cur_batch_label_one_hot[rp], sigma_list=sigma_list)

                        else:

                            loss_val_lab = mix_rbf_mmd2_joint(batch_estimate_features,
                                                            cur_batch_features,
                                                            y_gumbel_softmax,
                                                            cur_batch_label_one_hot, sigma_list=sigma_list)

                        labelled_val_losses.append(loss_val_lab)

                        # get loss on individual feature variables...
                        feature_y_losses = []
                        for c in unlabelled_keys:
                            associated_names = dsc.label_names_alphan[c]
                            # retrieve index of feature
                            current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                            # put into ancestor dict
                            current_batch_ground_truth = cur_batch_features[:, current_feature_idx].reshape(
                                (-1, dsc.feature_dim))
                            # get sigma list for current variable
                            sigma_list = [median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
                            # calculate mmd loss
                            
                            if current_ancestor_dict[c].shape[0]>11000:
                                
                                rp=torch.randperm(current_ancestor_dict[c].shape[0])[:11000]
                                
                                mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c][rp],
                                                        current_batch_ground_truth[rp],
                                                        y_gumbel_softmax[rp],
                                                        cur_batch_label_one_hot[rp],
                                                        sigma_list=sigma_list)
                                
                            else:
                            
                                mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c],
                                                        current_batch_ground_truth,
                                                        y_gumbel_softmax,
                                                        cur_batch_label_one_hot,
                                                        sigma_list=sigma_list)
                            feature_y_losses.append(mloss)

                        for c in conditional_keys:
                            associated_names = dsc.label_names_alphan[c]
                            # retrieve index of feature
                            current_feature_idx = [feature_idx_subdict[k] for k in associated_names]
                            # put into ancestor dict
                            current_batch_ground_truth = cur_batch_features[:, current_feature_idx].reshape(
                                (-1, dsc.feature_dim))
                            # get sigma list for current variable
                            sigma_list = [median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
                            # calculate mmd loss
                            if current_ancestor_dict[c].shape[0]>11000:
                                
                                rp=torch.randperm(current_ancestor_dict[c].shape[0])[:11000]
                                
                                mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c][rp],
                                                        current_batch_ground_truth[rp],
                                                        y_gumbel_softmax[rp],
                                                        cur_batch_label_one_hot[rp],
                                                        sigma_list=sigma_list)
                            else:
                                mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c],
                                                        current_batch_ground_truth,
                                                        y_gumbel_softmax,
                                                        cur_batch_label_one_hot,
                                                        sigma_list=sigma_list)
                            feature_y_losses.append(mloss)

                        for m in feature_y_losses:
                            labelled_val_losses.append(m)

                    cb = torch.nn.BCEWithLogitsLoss()
                    cbl_oh = cur_batch_label_one_hot.float()
                    loss_lab_bce=cb(y_generated,cbl_oh)



                    writer.add_scalar('validation bce loss',
                                      loss_lab_bce, t_iter)




                    acc_lab=get_accuracy(y_generated,cur_batch_label_one_hot.argmax(1).float())
                    current_inverse_acc=1-acc_lab

                    #GET OUR TRAINING DATA, LABELLED  FOR Y






                    pred_y_train = dsc_generators[labelled_key].forward(y_features)



                    training_bce=cb(pred_y_train,y_truelabel)
                    writer.add_scalar('labelled bce loss',
                                      training_bce, t_iter)

                    if args.val_loss_criterion=='unlabelled':
                        loss=loss_val_ulab
                    elif args.val_loss_criterion=='labelled':
                        loss=loss_val_lab
                    elif args.val_loss_criterion=='combined_lab_ulab':
                        loss=current_inverse_acc+loss_val_ulab
                    elif args.val_loss_criterion=='labelled_bce':
                        loss=loss_lab_bce
                    elif args.val_loss_criterion=='labelled_bce_and_labelled_mmd':
                        loss=loss_lab_bce+loss_val_lab
                    elif args.val_loss_criterion=='labelled_bce_and_all_feat_mmd':
                        loss=loss_lab_bce+loss_val_ulab
                    elif args.val_loss_criterion=='labelled_bce_and_all_feat_mmd_with_individual':
                        loss=sum(labelled_val_losses)/len(labelled_val_losses)+sum(unlabelled_val_losses)/len(unlabelled_val_losses)#+inverse_acc
                        #loss=sum(unlabelled_val_losses)/len(unlabelled_val_losses)+inverse_acc

                        # ...log the running loss
                        writer.add_scalar('mmd_val_loss',
                                          loss,t_iter)

                    print('epoch: {1}\ttemp: {2}\tval loss: {0}'.format(loss,epoch,str(temp)[:7]))
                    val_losses_joint.append(loss.detach().item())

                    # if loss does not improve after DELAYED_MIN_START epochs, model is optimal
                    epoch+=1


                    val_bces.append(loss_lab_bce)
                    inv_val_accuracies.append(current_inverse_acc)
                    #need to set optimal_label_gen for first epoch
                    if epoch==1:
                        optimal_label_gen = copy.deepcopy(dsc_generators[dsc.label_var])
                        current_min_bce = loss_lab_bce
                        mintemp = temp

                    if epoch>=2:
                        if loss_lab_bce < min(val_bces[-epoch:-1]):
                            optimal_label_gen = copy.deepcopy(dsc_generators[dsc.label_var])
                            current_min_bce = loss_lab_bce
                            mintemp=temp

                    if DELAYED_MIN_START==epoch:
                        optimal_mods = copy.deepcopy(dsc_generators)
                        mintemp = temp
                        current_val_loss = val_losses_joint[-1]
                        minloss = current_val_loss
                        converged = False




                    if DELAYED_MIN_START<epoch:
                        PREVIOUS_TRAJECTORY=args.estop_patience #set trajectory to number of epochs so we just take min val loss of all models
                        #get val_losses...
                        PREVIOUS_TRAJECTORY=epoch #try setting to epoch instead here

                        prev_k_vloss=val_losses_joint[-PREVIOUS_TRAJECTORY:]
                        #get most prev val loss
                        current_val_loss=val_losses_joint[-1]


                        if current_val_loss==max(prev_k_vloss):
                            converged = True
                            #normally write converged=True here
                        if current_val_loss==min(val_losses_joint):
                            optimal_mods=copy.deepcopy(dsc_generators)
                            mintemp=temp
                            minloss=current_val_loss
                            # ...log the running loss
                            writer.add_scalar('min_val_loss',
                                              minloss, t_iter)

                        #get current prediction for val acc on Y generator...
                    gone_thru_once=True
                rt_dict[temp]=min(val_losses_joint)

    #setting optimal mods like so
    optimal_mods = copy.deepcopy(dsc_generators)
    #setting our gen for label variable to be whatever was decided according to min inverse acc


    if args.use_optimal_y_gen:
        print('overwriting p(Y|X) for optimal generator...')

        optimal_mods[dsc.label_var] = optimal_label_gen

    print('list of bce')
    print(val_bces)


    #print('min of all bce')
    #print(current_min_bce)
    #mintemp = temp
    #current_val_loss = val_losses_joint[-1]
    #minloss = current_val_loss
    #converged = False

    print('pause placeholder')
    print('min loss and temp: ')
    minloss=1
    mintemp=1
    print(minloss)
    print(mintemp)

    for k in optimal_mods.keys():
        dsc_generators[k]=optimal_mods[k]

    #move everything back onto cpu
    if has_gpu:
        for k in [labelled_key]+conditional_keys:
            dsc_generators[k].to('cpu')

    #put in eval mode
    for k in all_keys:
        dsc_generators[k].eval()



    for k in dsc_generators.keys():
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
        torch.save(dsc_generators[k],model_save_name)

    # creating synthetic data
    if args.use_benchmark_generators:
        # we need to replace the generators in dsc_generators,
        for k in unlabelled_keys:
            # find model name...
            bmodel=create_model_name(k,'basic')
            model_to_search_for=dspec.save_folder+'/saved_models/'+bmodel+"*-s_i={0}-epoch*".format(s_i)
            candidate_models=glob.glob(model_to_search_for)
            #load thee model
            dsc_generators[k]=type(dsc_generators[k]).load_from_checkpoint(checkpoint_path=candidate_models[0])

    #n_samples = 30000
    n_samples=int(30000*min(dspec.n_unlabelled/1000,5)) #try to set this to deal wtih very large unalbeleld size...
    
    #synthetic_samples_dict = generate_samples_to_dict(dsc, has_gpu, dsc_generators, device_string, n_samples,gumbel=True,tau=mintemp)
    #don't use gumbel dist with the temp
    synthetic_samples_dict = generate_samples_to_dict(dsc, has_gpu, dsc_generators, device_string, n_samples)
    joined_synthetic_data = samples_dict_to_df(dsc, synthetic_samples_dict, balance_labels=True,exact=False)

    algo_spec = copy.deepcopy(master_spec['algo_spec'])
    algo_spec.set_index('algo_variant', inplace=True)

    synthetic_data_dir = algo_spec.loc[algo_variant].synthetic_data_dir

    save_synthetic_data(joined_synthetic_data,
                        d_n,
                        s_i,
                        master_spec,
                        dspec,
                        algo_spec,
                        synthetic_data_dir)

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
    if dsc.feature_dim == 1:
        print('feature dim==1 so we are going to to attempt to plot synthetic v real data')
        plot_synthetic_and_real_data(hpms_dict, dsc, args, s_i, joined_synthetic_data, synthetic_data_dir, dspec)
    else:
        # scols = [s.replace('_0', '') for s in joined_synthetic_data.columns]
        # joined_synthetic_data.columns = scols
        joined_synthetic_data.rename(columns={'Y_0': 'Y'}, inplace=True)
        # joined_synthetic_data=joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
        if 'y_given_x_bp' in dsc.merge_dat.columns:
            dsc.merge_dat = dsc.merge_dat[[c for c in joined_synthetic_data.columns]]
        if 'y_given_x_bp' in joined_synthetic_data.columns:
            joined_synthetic_data = joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
        synthetic_and_orig_data = pd.concat([dsc.merge_dat, joined_synthetic_data], axis=0)

        dsc.merge_dat = synthetic_and_orig_data
        dsc.d_n = d_n
        dsc.var_types = dsc.variable_types
        dsc.var_names = dsc.labels
        dsc.feature_varnames = dsc.feature_names
        dsc.s_i = s_i
        plot_2d_data_w_dag(dsc, s_i,synthetic_data_dir=synthetic_data_dir)

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
