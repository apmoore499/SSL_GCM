from torch.utils.data import sampler
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
rng = np.random.default_rng(12345)
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import pickle
import sys
import os
from mmd_mmg import *
import shutil
import pandas as pd
import numpy as np
import copy
import torch.nn as nn
from numpy.random import normal
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import pairwise_distances,pairwise_distances_chunked
import glob
import warnings
import igraph
import igraph as ig
warnings.filterwarnings('ignore')


dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd() #current working directory
target_dir='/'.join(cwd.split('/')[:-1])
sys.path.append(target_dir)


nworker_val=0
nworker_train=0

#create empty default args class, can use as regular args if we don't call argparse
class dargs:
    def __init__(self):
        return


# for debugging if need, write set_trace() and it will call from terminal
from IPython.core.debugger import set_trace

#-----------------------------------
#     metrics and callbacks and training
#-----------------------------------

# initialize metric
#accuracy = torchmetrics.Accuracy()

# softmax pred
get_softmax = torch.nn.Softmax(dim=1)  # need this to convert classifier predictions

#new accuracy method because there is bug in pytorch lightnign
def get_accuracy(pred,true):
    accuracy=(pred.argmax(1)==true).float().mean()
    return(accuracy)

#new accuracy method because there is bug in pytorch lightnign
def get_accuracy_cgan(pred,true):
    accuracy=(pred.argmax(1)==true[:,1]).float().mean()
    return(accuracy)

#returns model checkpoint to save max performing model
def return_chkpt_max_acc(model_name,data_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="{0}/saved_models".format(data_dir),
        filename=model_name+ "-{s_i:.0f}-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )
    return(checkpoint_callback)

#for model with min mmd loss validation mmd
def return_chkpt_min_val_mmd(model_name, data_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mmd",
        dirpath="{0}/saved_models".format(data_dir),
        filename=model_name+ "-{s_i:.0f}-{epoch:02d}-{val_mmd:.2f}",
        save_top_k=1,
        mode="min",
    )
    return(checkpoint_callback)


#for model with min mmd loss validation mmd
def return_chkpt_min_trans_mmd(model_name,data_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="trans_mmd",
        dirpath="{0}/saved_models".format(data_dir),
        filename=model_name+ "-{s_i:.0f}-{epoch:02d}-{trans_mmd:.2f}",
        save_top_k=1,
        mode="min",
    )
    return(checkpoint_callback)


#returns model checkpoint max val accuracy
def return_chkpt_max_val_acc(model_name,data_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="{0}/saved_models".format(data_dir),
        filename=model_name+ "-{d_n:.0f}-{s_i:.0f}-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )
    return(checkpoint_callback)

#returns model checkpoint to save max performing model w bce validation loss
def return_chkpt_min_bce_validation(model_name,data_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="v_celoss",
        dirpath="{0}/saved_models".format(data_dir),
        filename=model_name+ "-{s_i:.0f}-{epoch:02d}-{classifier_bce_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    return(checkpoint_callback)

#returns model checkpoint to save max performing model
def return_chkpt_min_bce(model_name,data_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="classifier_bce_loss",
        dirpath="{0}/saved_models".format(data_dir),
        filename=model_name+ "-{s_i:.0f}-{epoch:02d}-{classifier_bce_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    return(checkpoint_callback)

#early stopping callback validation mmd
def return_early_stop_min_val_mmd(patience=50):
    early_stop_callback = EarlyStopping(monitor="val_mmd",
                                        min_delta=0.00,
                                        patience=patience,
                                        verbose=False,
                                        mode="min",
                                        check_finite=True)
    return(early_stop_callback)


#early stopping callback transductive mmd
def return_early_stop_min_trans_mmd(patience=50):
    early_stop_callback = EarlyStopping(monitor="trans_mmd",
                                        min_delta=0.00,
                                        patience=patience,
                                        verbose=False,
                                        mode="min",
                                        check_finite=True)
    return(early_stop_callback)


#early stopping callback classifier binary cross entropy loss
def return_early_stop_cb_bce(patience=100):
    early_stop_callback = EarlyStopping(monitor="classifier_bce_loss", min_delta=0.00, patience=patience, verbose=False, mode="min")
    return(early_stop_callback)

#early stopping callback on training loss
def return_early_stop_train_loss(patience=100):
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=patience, verbose=False, mode="min")
    return(early_stop_callback)

# logger for tboard
def create_logger(model_name, d_n, s_i,default_dir="lightning_logs/"):
    tb_logger = pl_loggers.TensorBoardLogger(default_dir,name=combined_name(model_name, d_n, s_i),version=0)
    return (tb_logger)

# trainer for pytorch lightning
def create_trainer(logger,callbacks,gpu_kwargs,max_epochs):
    trainer= pl.Trainer(log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                        logger=logger,
                        #progress_bar_refresh_rate=0,
                        reload_dataloaders_every_n_epochs=1)
                        #**gpu_kwargs)
    return(trainer)


#-----------------------------------
#     return feed forward neural net
#-----------------------------------

#returns standard net architecture
def get_one_net(input_dim,output_dim,middle_layer_n):
    net = nn.Sequential(
            nn.Linear(input_dim, middle_layer_n),
            nn.ReLU(),
            nn.Linear(middle_layer_n, output_dim)
        )
    return(net)


def get_three_net(input_dim,output_dim,middle_layer_n):
    net = nn.Sequential(
            nn.Linear(input_dim, middle_layer_n),
            nn.ReLU(),
            nn.Linear(middle_layer_n, 3),
            nn.ReLU(),
            nn.Linear(3, output_dim)
        )
    return(net)

# returns standard net architecture
def get_standard_net(input_dim,output_dim):

    net = nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Linear(100, 5),
        nn.ReLU(),
        nn.Linear(5, output_dim)
    )
    return(net)



#-----------------------------------
#     data management functions
#-----------------------------------

# get the arg list
def new_empty_results(n_iterations, n_samples):
    retval = np.zeros((n_iterations, n_samples))  #
    return (retval)

# return the data iterator func
def return_data_iterator(model_number):
    model_number = int(model_number)
    retfuncs = [get_iters_x1,
                get_iters_x2,
                get_iters,
                get_iters,
                get_iters]
    retval = retfuncs[model_number - 1]
    return (retval)

# transduction validation dataloader
def get_trans_val(orig_data, l_batch_size=16, ul_batch_size=128, pseudo_label=None):
    # ----------#
    # Training Labelled
    # ----------#
    X_train_lab = orig_data['label_features']

    y_train_lab = torch.argmax(orig_data['label_y'], 1)

    # ----------#
    # Training Unlabelled
    # ----------#
    X_train_ulab = orig_data['unlabel_features']
    y_train_ulab = torch.argmax(orig_data['unlabel_y'], 1)

    # Validation Sets

    # -------------#
    # Validation
    # -------------#

    X_val = orig_data['val_features']
    y_val = torch.argmax(orig_data['val_y'], 1)

    # -------------#
    # Setting up resampling
    # -------------#

    n_unlabelled = X_train_ulab.shape[0]

    # sample with replacement indices of X_train_lab
    n_labelled = X_train_lab.shape[0]

    # dummy tensor of l;abelled representyingh probabiliity weights, just use tensor.ones
    dummy_label_weights = torch.ones(n_labelled)

    resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_unlabelled, replacement=True)

    X_train_lab_rs = X_train_lab[resampled_i]
    y_train_lab_rs = y_train_lab[resampled_i]

    if pseudo_label is None:
        y_unlabeled = y_train_ulab
    else:
        assert isinstance(pseudo_label, np.ndarray)
        y_unlabeled = pseudo_label

    data_loaders = {

        'ulab_mix': DataLoader(
            torch.utils.data.TensorDataset(X_train_lab_rs,
                                           y_train_lab_rs,
                                           X_train_ulab),
            batch_size=ul_batch_size),

        'labeled': DataLoader(
            torch.utils.data.TensorDataset(X_train_lab, y_train_lab),
            batch_size=l_batch_size),

        'unlabeled': DataLoader(
            torch.utils.data.TensorDataset(X_train_ulab, y_train_ulab),
            batch_size=ul_batch_size),

        'make_pl': DataLoader(
            torch.utils.data.TensorDataset(X_train_ulab,
                                           y_train_ulab),
            shuffle=True,
            batch_size=ul_batch_size),

        'val': DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val),
            batch_size=X_val.shape[0],
            shuffle=False),

        'transduction': DataLoader(
            torch.utils.data.TensorDataset(X_train_ulab,
                                           y_train_ulab),
            batch_size=X_train_ulab.shape[0],
            shuffle=False)
    }

    val = (X_val, y_val)
    trans = (X_train_ulab, y_train_ulab)

    return val, trans

# get validation data loader
def get_val_dataloader(orig_data):
    tfeat = get_trans_val(orig_data)[0][0].unsqueeze(0)
    tlab = get_trans_val(orig_data)[0][1].unsqueeze(0)
    vfeat = get_trans_val(orig_data)[1][0].unsqueeze(0)
    vlab = get_trans_val(orig_data)[1][1].unsqueeze(0)
    all_d = torch.utils.data.TensorDataset(tfeat, tlab, vfeat, vlab)
    val_dataloader = DataLoader(all_d,
                                num_workers=nworker_val)  # defining validation dataloader to be used by all models
    return (val_dataloader)


#training data
def load_data(d_n,s_i,dataset_folder,device='cpu'):
    data_types = ['label_features',
                  'unlabel_features',
                  'val_features',
                  'label_y',
                  'unlabel_y',
                  'val_y',
                  'test_features',
                  'test_y']
    out_d={}
    for dt in data_types:
        in_fn = './{3}/d_n_{0}_s_i_{1}_{2}.pt'.format(d_n, s_i, dt,dataset_folder)
        out_d[dt]=torch.load(in_fn,map_location=torch.device(device))
    return(out_d)

#real world data, no test partition
def load_data(d_n,s_i,dataset_folder,device='cpu'):
    data_types = ['label_features',
                  'unlabel_features',
                  'val_features',
                  'test_features',
                  'label_y',
                  'unlabel_y',
                  'val_y',
                  'test_y']
    out_d={}
    for dt in data_types:
        in_fn = './{3}/d_n_{0}_s_i_{1}_{2}.pt'.format(d_n, s_i, dt,dataset_folder)
        out_d[dt]=torch.load(in_fn,map_location=torch.device(device))
    return(out_d)

# convert data tensors into a dataframe
def get_merge_dat_from_orig(orig_data,feature_cols):
    for k in orig_data.keys():
        orig_data[k]=orig_data[k].detach().cpu().numpy()
        if '_y' in k[-2:]:
            orig_data[k]=orig_data[k].argmax(1)
    #ok now we convert to dataframe...
    for k in orig_data.keys():
        orig_data[k]=pd.DataFrame(orig_data[k])
        if 'features' in k:
            orig_data[k].columns=feature_cols
        elif '_y' in k[-2:]:
            orig_data[k].columns=['Y_0']
    dm1=pd.concat([orig_data['label_features'],orig_data['label_y']],axis=1)
    dm1['type']='labelled'
    dm2=pd.concat([orig_data['unlabel_features'],orig_data['unlabel_y']],axis=1)
    dm2['type']='unlabelled'
    dm3=pd.concat([orig_data['val_features'],orig_data['val_y']],axis=1)
    dm3['type']='validation'
    merge_dat=pd.concat([dm1,dm2,dm3],axis=0,ignore_index=True)
    return(merge_dat)


# metadata for gaussain mix class
class gaussian_mixture_metadata:

    def __init__(self):
        self.adj_mats={}
        self.adj_mats[1]=([[0,1],[0,0]])
        self.adj_mats[2]=([[0,1],[0,0]])
        self.adj_mats[3]=([[0,1,0],[0,0,1],[0,0,0]])
        self.adj_mats[4]=([[0,0,1],[0,0,1],[0,0,0]])
        self.adj_mats[5]=([[0,1,1],[0,0,1],[0,0,0]])

        self.vtypes={}
        self.vtypes[1]=['feature','label']
        self.vtypes[2]=['label','feature']
        self.vtypes[3]=['feature','label','feature']
        self.vtypes[4]=['feature','label','feature']
        self.vtypes[5]=['feature','label','feature']

        self.vlabels={}
        self.vlabels[1]=['X1','Y']
        self.vlabels[2]=['Y','X2']
        self.vlabels[3]=['X1','Y','X2']
        self.vlabels[4]=['X1','Y','X2']
        self.vlabels[5]=['X1','Y','X2']


        self.feature_cols={}
        self.feature_cols[1]=['X1_0','X1_1']
        self.feature_cols[2]=['X2_0','X2_1']
        self.feature_cols[3]=['X1_0','X1_1','X2_0','X2_1']
        self.feature_cols[4]=['X1_0','X1_1','X2_0','X2_1']
        self.feature_cols[5]=['X1_0','X1_1','X2_0','X2_1']

# gaussian mixture model data
def load_gaussian_mixture_legacy(args, s_i, dspec):
    orig_data = load_data(d_n=args.d_n, s_i=s_i, dataset_folder=dspec.save_folder)
    gmd = gaussian_mixture_metadata()
    dnumber = int(args.d_n.split('e_d')[1])
    adj_mat = gmd.adj_mats[dnumber]
    var_types = gmd.vtypes[dnumber]
    fcols = gmd.feature_cols[dnumber]
    merge_dat = get_merge_dat_from_orig(orig_data, fcols)
    mdcols = [c for c in merge_dat.columns]
    if 'Y_0' in mdcols: #have to use Y_0 for multidim feature X
        merge_dat.rename(columns={"Y_0": "Y"}, inplace=True)
    labels = gmd.vlabels[dnumber]
    dsc = dsc_mog(adjacency_matrix=adj_mat,
                  var_types=var_types,
                  merge_dat=merge_dat,
                  labels=labels)
    # label_name=dsc.labels[np.where([d=='label' for d in dsc.variable_types])[0][0]]
    dsc.label_var = dsc.labels[np.where([d == 'label' for d in dsc.variable_types])[0][0]]
    dsc.class_varname=dsc.label_var

    return (dsc)

# real data (not synthetic)
def load_real_data_legacy(args, s_i, dspec):
    pkl_n=glob.glob('{0}/*.pickle'.format(dspec.save_folder))[0]
    with open(pkl_n, 'rb') as f:
        data_pkl = pickle.load(f)
    c_name=data_pkl.class_varname
    dcols=[c for c in data_pkl.merge_dat.drop(columns=['type']).columns]
    is_lab=[c_name==d for d in dcols]
    lab_transform = lambda lab: 'label' if lab else 'feature'
    feat_or_lab=[lab_transform(c) for c in is_lab]
    data_pkl.variable_types=feat_or_lab
    data_pkl.feature_dim=dspec.feature_dim
    data_pkl.dstype='real'
    for vname,vtype in zip(dcols,feat_or_lab):
        if vtype=='label':
            label_var=vname
    data_pkl.label_var=label_var
    dsc=data_pkl
    return(dsc)

# misc manipulation to data before training
def manipulate_dsc(dsc,dspec):
    dsc.label_names = copy.deepcopy(dsc.labels)
    dsc.label_names_alphan = {}
    dsc.n_classes=2
    for l in dsc.labels:
        dsc.label_names_alphan[l]=[l]
    dsc.label_names = [[d] for d in dsc.label_names]



    dsc.feature_dim=dspec.feature_dim
    #change the feature dim for the synthetic data we made
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
    dsc.causes_of_y = []
    # get feature names...

    dsc.feature_names=[]

    for l in dsc.labels:
        if l==dsc.label_var:
            dummy=1
        else:
            dsc.feature_names.append(l)

    return(dsc)

# load dsc.pickle object
def load_dsc(args,s_i,dspec):
    dsc_fn='{0}/d_n_{1}_s_i_{2}_dataset_class.pickle'.format(dspec.save_folder,args.d_n,s_i)
    dsc_f=open(dsc_fn,'rb')
    dsc=pickle.load(dsc_f) #dataset class
    dsc.label_var=dsc.class_varname
    return(dsc)

# synthetic samples generation
def generate_samples_to_dict(dsc, has_gpu, dsc_generators, device_string,n_samples=10000,gumbel=False,tau=None):
    with torch.no_grad():
        synthetic_data = {}
        order_to_generate = dsc.dag.topological_sorting()
        labs = dsc.labels
        for v_i in order_to_generate:
            source_edges = dsc.dag.es.select(_target=v_i)
            source_vertices = [s_e.source_vertex.index for s_e in source_edges]
            # conditional_info=[synthetic_data[v_s] for v_s in source_vertices]
            cur_lab = labs[v_i]
            cur_gen = dsc_generators[cur_lab]

            if has_gpu:
                cur_gen.cuda()
            print('synthesising for: {0}'.format(cur_lab))

            #if dsc.variable_types[v_i] == 'label':
                #data_label_variable = cur_lab
                #if dsc.feature_dim > 1:
                #    cur_lab = cur_lab + '_0'
            # synthetic_data['alm']=torch.tensor(dsc.merge_dat['alm'].values).reshape((n_samples,dsc.feature_dim)).float()
            if len(source_vertices) == 0:  # no conditional terms..
                # if Y...label..
                # if cur_lab=='Y':
                if dsc.variable_types[v_i] == 'label':
                    # use the torch.gen....

                    synthetic_samples = cur_gen.forward(n_samples)

                    if gumbel==True:
                        y_samples = torch.nn.functional.gumbel_softmax(synthetic_samples,hard=True,tau=tau)
                    else:
                        y_probs = get_softmax(synthetic_samples.float())  # [:,1]
                        # and take these as prob of y=0,1
                        y_samples = torch.bernoulli(y_probs)
                    # set_trace()
                    # make sure is one_hot vector...
                    synthetic_samples = synthetic_samples.argmax(1)
                    synthetic_samples = torch.nn.functional.one_hot(synthetic_samples)
                    # synthetic_samples = torch.nn.functional.one_hot(y_samples)
                    synthetic_samples = synthetic_samples.to(device_string)

                else:
                    # sample noise....
                    # set_trace()
                    z = torch.randn((n_samples, dsc.feature_dim), device=device_string)
                    synthetic_samples = cur_gen.forward(z)
                synthetic_data[cur_lab] = synthetic_samples
            else:
                conditional_labs = [dsc.labels[s_v] for s_v in source_vertices]
                # put the Y on RIGHT hand side of dataframe

                if dsc.label_var in conditional_labs:
                    conditional_labs.remove(dsc.label_var)
                    conditional_labs=conditional_labs+[dsc.label_var]



                # synthetic_data_for_current={}
                conditional_sdata = [synthetic_data[s_c] for s_c in conditional_labs]
                conditional_sdata = tuple(conditional_sdata)



                # set_trace()
                # now concatenate this one
                cat_cond_sdata = torch.cat(conditional_sdata, 1)
                # if label Y, then no noise..
                if dsc.variable_types[v_i] == 'label':
                    synthetic_samples = cur_gen.forward(cat_cond_sdata)
                    # print(synthetic_samples)
                    sprobs = get_softmax(synthetic_samples)
                    prob_y1 = sprobs[:, 1]
                    # print(synthetic_samples)
                    y1_samp = torch.bernoulli(prob_y1)
                    synthetic_samples = torch.nn.functional.one_hot(y1_samp.long(), dsc.n_classes)
                else:
                    # it's a feature variable
                    z = torch.randn((n_samples, dsc.feature_dim), device=device_string)
                    # cat them...
                    print(cat_cond_sdata)
                    gen_input = torch.cat((z, cat_cond_sdata), 1)

                    synthetic_samples = cur_gen.forward(gen_input)

                synthetic_data[cur_lab] = synthetic_samples

        return (synthetic_data)


# synthetic samples generation
def generate_samples_to_dict_tripartite(dsc, has_gpu, dsc_generators, device_string,n_samples=10000,gumbel=False,tau=None):
    with torch.no_grad():
        synthetic_data = {}

        #cause generator first
        print('synthesising for: CAUSE / SPOUSE')

        cause_spouse_labels=dsc_generators['ordered_v_alphan']['cause']

        z = torch.randn((n_samples, len(cause_spouse_labels)), device='cpu')
        dsc_generators['cause_spouse_generator'].to('cpu')
        synthetic_samples = dsc_generators['cause_spouse_generator'].forward(z)

        synthetic_df=pd.DataFrame(synthetic_samples.numpy(),columns=cause_spouse_labels)

        #label genrator next
        print('synthesising for: LABEL')
        #dsc_generators['ordered_v']['label'] = {}
        #dsc_generators['ordered_v']['label']['inputs'] = cond_lab
        #dsc_generators['ordered_v']['label']['inputs_alphan'] = concat_cond_lab
        #dsc_generators['ordered_v']['label']['output'] = dsc.labels[v_i]

        l_var=dsc_generators['ordered_v']['label']['output']

        dsc_generators[l_var].to('cpu')


        current_inputs=dsc_generators['ordered_v']['label']['inputs_alphan']
        current_input_values=torch.tensor(synthetic_df[current_inputs].values,device='cpu')

        #z = torch.randn((n_samples, len(cause_spouse_labels)), device=device_string)
        synthetic_samples = torch.bernoulli(torch.nn.functional.softmax(dsc_generators[l_var].forward(current_input_values),1)[:,1])

        #now sample bernoulli


        synthetic_data[l_var]=torch.nn.functional.one_hot(synthetic_samples.long()).float()


        #cs_df=pd.DataFrame(synthetic_samples,columns=[l_var])

       # synthetic_df=pd.concat([synthetic_df,cs_df],axis=1)


        # convert label to one-hot


        #now do EFFECT

        #dsc_generators['ordered_v']['effect'] = {}
        #dsc_generators['ordered_v']['effect']['inputs'] = cond_lab
        #dsc_generators['ordered_v']['effect']['inputs_alphan'] = concat_cond_lab
        #dsc_generators['ordered_v']['effect']['outputs'] = conditional_feature_names
        #dsc_generators['ordered_v']['effect']['outputs_alphan'] = cur_x_lab
        #dsc_generators['ordered_v']['effect'] = {}
        #dsc_generators['ordered_v']['effect']['inputs'] = cond_lab
        #dsc_generators['ordered_v']['effect']['input_features_alphan'] = concat_cond_lab
        #dsc_generators['ordered_v']['effect']['input_label_alphan'] = label_name
        #dsc_generators['ordered_v']['effect']['outputs'] = conditional_feature_names
        #dsc_generators['ordered_v']['effect']['outputs_alphan'] = cur_x_lab

        current_feature_inputs = dsc_generators['ordered_v']['effect']['input_features_alphan']
        current_feature_input_values = torch.tensor(synthetic_df[current_feature_inputs].values, device='cpu')

        current_feature_outputs=dsc_generators['ordered_v']['effect']['outputs_alphan']

        z = torch.randn((n_samples, len(current_feature_outputs)), device='cpu')

        cat_input=torch.cat((z,current_feature_input_values),1)

        #now cat w label

        cat_input_w_label=torch.cat((cat_input,synthetic_data[l_var]),1).float()



        dsc_generators['effect_generator'].to('cpu')
        # z = torch.randn((n_samples, len(cause_spouse_labels)), device=device_string)
        synthetic_samples = dsc_generators['effect_generator'].forward(cat_input_w_label)



        cs_df = pd.DataFrame(synthetic_samples.numpy(), columns=current_feature_outputs)

        synthetic_df = pd.concat([synthetic_df, cs_df], axis=1)


        #now split out into the dict of values
        #to maintain compatilibity w existing code

        #get labels...
        all_features=dsc.feature_names
        for f in all_features:
            #get alphan name (for multidim)
            alphan_name=dsc.label_names_alphan[f]
            #assign to dict
            synthetic_data[f]=synthetic_df[alphan_name]
            synthetic_data[f]=torch.tensor(synthetic_data[f].values)

        for f in [dsc.label_var]:
            synthetic_data[f]=synthetic_data[l_var]
            #synthetic_data[f]=torch.tensor(synthetic_data[f].values)

            if l_var!=f:
                del synthetic_data[l_var]

        return (synthetic_data)



# synthetic samples to dictionary
def samples_dict_to_df(dsc, synthetic_data, balance_labels=True,exact=False):
    synthetic_df = {}

    order_to_generate = dsc.dag.topological_sorting()

    if hasattr(dsc, 'dstype'):
        for v_i in order_to_generate:
            k = dsc.labels[v_i]
            vtype = dsc.variable_types[v_i]

            if vtype == 'label':
                cur_dat = synthetic_data[k]
                cur_dat = torch.argmax(cur_dat, 1)
                synthetic_df[k] = pd.DataFrame(cur_dat.cpu().detach().numpy())
                synthetic_df[k].columns = [k]
            else:
                synthetic_df[k] = pd.DataFrame(synthetic_data[k].cpu().detach().numpy())
                synthetic_df[k].columns = [k]
    else:
        print('converting data to df')
        for k in synthetic_data.keys():
            #get type of the variable...
            dsc.label_var
            if dsc.label_var==k and k=='Y':
            #if k == 'Y':  # need to convert from one-hot to long
                cur_dat = synthetic_data[k]
                cur_dat = torch.argmax(cur_dat, 1)
                synthetic_df[k] = pd.DataFrame(cur_dat.cpu().detach().numpy())
                if dsc.feature_dim > 1:
                    synthetic_df[k].columns = ['Y_0']
                    dsc.label_var='Y_0'
                else:
                    synthetic_df[k].columns = ['Y']
            else:
                synthetic_df[k] = pd.DataFrame(synthetic_data[k].cpu().detach().numpy())

                if dsc.feature_dim > 1:
                    synthetic_df[k].columns = ['{0}_{1}'.format(k, ncol) for ncol in
                                               range(synthetic_df[k].shape[1])]
                else:
                    synthetic_df[k].columns = ['{0}'.format(k)]


    # bind all to dataframe...
    join_dfs = [synthetic_df[k] for k in synthetic_data.keys()]
    joined_synthetic_data = pd.concat(join_dfs, axis=1)
    joined_synthetic_data['type'] = 'synthetic'

    if balance_labels and exact:
        # subset to equal numbers of 0,1 in class...
        synth_c0 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 0]
        synth_c1 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 1]
        print('enforcing balance of labels P(Y=0)=P(Y=1)=0.5')
        num_c0 = synth_c0.shape[0]
        num_c1 = synth_c1.shape[0]
        print('num c0 {0}'.format(num_c0))
        print('num c1 {0}'.format(num_c1))
        # take minimum one...
        if num_c0 < num_c1:
            # num_c1 is subset...
            synth_c1 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 1].sample(num_c0)
        elif num_c1 < num_c0:
            synth_c0 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 0].sample(num_c1)

        #now we have how many label==0,1?
        print('number of samples w label==1: {0}'.format(synth_c1.shape[0]))
        print('number of samples w label==0: {0}'.format(synth_c0.shape[0]))
        joined_synthetic_data = pd.concat((synth_c0, synth_c1), axis=0, ignore_index=True)
        print('total size of df: {0} rows'.format(joined_synthetic_data.shape[0]))


    if balance_labels and not exact:

        print('balancing labels in same proportion as exist in training dataset. ie: not necessarily exact')

        orig_lidx=np.where(np.array(dsc.variable_types)=='label')[0][0]

        orig_lname=dsc.label_names[orig_lidx][0]

        orig_labels=dsc.merge_dat[orig_lname]
        orig_p0=sum(orig_labels==0)/dsc.merge_dat.shape[0]
        orig_p1=sum(orig_labels==1)/dsc.merge_dat.shape[0]

        n_p0=int(orig_p0*dsc.merge_dat[dsc.merge_dat.type=='unlabelled'].shape[0])
        n_p1 = int(orig_p1 * dsc.merge_dat[dsc.merge_dat.type=='unlabelled'].shape[0])
        # subset to proportion of 0,1 found in original sample, (ie. not necessarily exact)
        synth_c0 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 0]
        synth_c1 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 1]

        num_c0 = synth_c0.shape[0]
        num_c1 = synth_c1.shape[0]
        print('num c0 {0}'.format(num_c0))
        print('num c1 {0}'.format(num_c1))
        # sample according to prop in orig data dsc.merge_dat


        synth_c1 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 1].sample(n_p1)
        synth_c0 = joined_synthetic_data[joined_synthetic_data[dsc.label_var] == 0].sample(n_p0)

        #now we have how many label==0,1?
        print('number of samples w label==1: {0}'.format(synth_c1.shape[0]))
        print('number of samples w label==0: {0}'.format(synth_c0.shape[0]))
        joined_synthetic_data = pd.concat((synth_c0, synth_c1), axis=0, ignore_index=True)
        print('total size of df: {0} rows'.format(joined_synthetic_data.shape[0]))


    return (joined_synthetic_data)

# DAG dataset class (legacy)
class DAG_dset:

    def __init__(self,
                 adjacency_matrix,
                 feature_dim=2,
                 n_samples=2500,
                 n_labelled=2000,
                 n_validation=500,
                 var_types=None,
                 var_values=None):

        self.adj_mat = np.array(adjacency_matrix)  # store adjacency matrix
        self.dag = igraph.Graph.Adjacency(self.adj_mat)  # get dag
        assert self.dag.is_dag()  # make sure is dag
        self.var_types = var_types
        self.var_values = var_values
        self.feature_dim = 2

        self.mu_mats = get_mus_matrices(self.adj_mat, self.feature_dim)
        self.bs_mats = get_bs_matrices(self.adj_mat, self.feature_dim)
        self.coeffs_mats = get_coeffs_matrices(self.adj_mat, self.feature_dim)
        self.cov_mats = get_wishart_matrices(self.adj_mat, self.feature_dim)

        self.n_samples = n_samples
        self.n_labelled = n_labelled
        self.n_validation = n_validation

    def label_variables(self):
        vnames = [""] * self.dag.vcount()
        feat_num = 1
        for v_i in self.dag.topological_sorting():
            if dg1.variable_types[v_i] == 'label':
                vnames[v_i] = 'Y'
            else:
                vnames[v_i] = 'X{0}'.format(feat_num)
                feat_num += 1
        self.var_names = vnames
        return (self)

    def generate_synthetic_data_wishart(self):
        generated_vertices = self.dag.topological_sorting()
        values = {}
        # set to zero for all in vertices
        for v in generated_vertices:
            values[v] = 0
        for v in generated_vertices:
            current_v_label = self.labels[v]  # get the variable label of the current variable
            assert current_v_label == 'Y' or 'X' in current_v_label

            if current_v_label == 'Y':

                # check if it is a root node
                extra_var = self.adj_mat[:, v]  # these are the linear components added to the variable of interest
                if np.all([v == 0 for v in extra_var]):
                    # set_trace()
                    # it is a root node
                    self.bernoulli_parameter = 0.5
                    values[v] = rng.binomial(1, self.bernoulli_parameter, self.n_samples)



                else:
                    # we have to add up the contrib from each X and find weighted mean and then take sigmoid
                    # centered at this value
                    # to ensure that we have balanced ie same number of label==1 and same label==0
                    theta_v_y = 0

                    feature_contrib_x = np.where(
                        extra_var == 1)  # get index of features which direct causes of y, ie X->y
                    feature_means = 0  # initialise
                    for v_x in feature_contrib_x:
                        # set_trace()
                        current_feature_means = self.mu_mats[v_x, v_x] + self.bs_mats[
                            v_x, v_x]  # get means of multidimensional X
                        current_feature_weights = self.coeffs_mats[v_x, v]  # weight FROM x TO y
                        current_feature_weighted_means = current_feature_weights * current_feature_means
                        current_feature_weighted_mean = current_feature_weighted_means.sum()  # sum them togeth
                        feature_means += current_feature_weighted_mean
                    # now we have the feature_means to use in sigmoid function
                    for source_vertex, incident in enumerate(extra_var):
                        # set_trace()
                        # all must be feature variables, so it is straightforward to add them
                        if incident == 1:
                            sum_feats_v = self.coeffs_mats[source_vertex, v] * values[source_vertex]
                            # set_trace()
                            theta_v_y += sum_feats_v.sum(1).reshape(-1, 1)  # sum over columns

                    # theta_v should look like [s1,s2,s3,...], ie, row vector
                    # now, take sigmoid of the theta_v vector, scaled by 2 (massive overlap), and centered at feature_means
                    sigmoid_of_theta_v = sigmoid(theta_v_y, mean=feature_means, scale=YSCALE)
                    # now, take bernoulli of sigmoid of theta_v
                    y_labels = [rng.binomial(1, b) for b in sigmoid_of_theta_v]
                    # now, set this to values[v]
                    values[v] = y_labels
            else:
                # set_trace()
                theta_v = rng.multivariate_normal(self.mu_mats[v, v], self.cov_mats[v, v], self.n_samples) + \
                          self.bs_mats[v, v]
                extra_var = self.adj_mat[:, v]  # these are the linear components added to the variable of interest
                for source_vertex, incident in enumerate(extra_var):
                    source_v_label = self.labels[source_vertex]
                    if source_v_label == 'Y' and incident == 1:
                        current_y_val = values[source_vertex]  # Y=0 or 1...
                        working_y = np.expand_dims(current_y_val, 0).transpose()
                        # stack together as many Y as feature dim..
                        y_stack = np.hstack([current_y_val])  # get rid of *fdim
                        # stack coeffs for self.n_samples
                        coef_stack = np.vstack([self.coeffs_mats[source_vertex, v]] * self.n_samples)
                        # multiply by weights each row of y
                        # set_trace()
                        y_coeffs = coef_stack * y_stack.reshape(-1, 1)
                        # adding weighted y contribution
                        # set_trace()
                        y_contrib = y_coeffs * incident
                        theta_v += y_contrib
                    else:  # it's a feature variable
                        theta_v += incident * self.coeffs_mats[source_vertex, v] * values[source_vertex]

                values[v] = theta_v
        # modify y variable, pull off array structure

        y_idx = np.where([lab == 'Y' for lab in self.labels])[0][0]
        if type(values[y_idx][0]) == np.ndarray:
            values[y_idx] = np.array([v[0] for v in values[y_idx]])

        self.variable_values = values

        return (self)

    # convert each variable to a dict...

    def convert_all_values_to_dict(self):

        values_dict = {}

        for var_i in self.dag.topological_sorting():
            var_label = self.labels[var_i]
            var_vals = self.variable_values[var_i]
            values_dict[var_label] = var_vals
        self.values_dict = values_dict

        return (self)

    def concat_df(self):
        # get all dfs
        self.dfs = [pd.DataFrame(self.values_dict[label]) for label in self.labels]

        # name them
        self.new_dfs = []

        # self.dfs=[]
        for var, df in zip(self.labels, self.dfs):
            # print(df.columns)
            df.columns = ['{0}_{1}'.format(var, dim) for dim in iter(df.columns)]
            # print(df.columns)
            self.new_dfs.append(df)

        self.dfs = self.new_dfs

        # concat them...

        self.merge_dat = pd.concat(self.dfs, axis=1)

        return (self)

    def val_split(self, n_labelled=50, n_validation=500):
        # self.pc_labelled=pc_labelled
        # split into val/labelled/unlabelled

        # first split into labelled AND unlabelled to ensure
        # proportionate representation of each class label in labelled / unlabelled set

        # self.merge_dat

        self.n_labelled = n_labelled
        self.n_validation = n_validation

        # self.n_unlabelled=self.total_n_samples - self.n_labelled - self.n_validation
        # self.labelled_i=rng.sample()

        list_to_sample = [i for i in range(self.n_samples)]

        labelled = rng.choice(list_to_sample, self.n_labelled, replace=False).ravel()  # labelled cases #rewrite..
        left = set(list_to_sample).difference(set(labelled))
        validation = rng.choice(list(left), self.n_validation, replace=False).ravel()  # validation set
        unlabelled = np.array(list(left.difference(set(validation)))).ravel()  # unlabelled cases
        all_togeth = np.concatenate([labelled, unlabelled, validation], axis=0)
        assert (len(set(all_togeth).difference(set(list_to_sample))) == 0)

        # store indices in class
        self.labelled_i = labelled
        self.validation_i = validation
        self.unlabelled_i = unlabelled

        # set data types in pandas df
        self.merge_dat['type'] = 'unlabelled'
        self.merge_dat.iloc[labelled, self.merge_dat.columns.get_loc('type')] = 'labelled'
        self.merge_dat.iloc[validation, self.merge_dat.columns.get_loc('type')] = 'validation'

        return (self)

        # getting feratures and labels etc

    def partition_data_df(self):
        # labelled feat
        self.label_features_df = self.merge_dat.drop(columns='Y_0')[self.merge_dat.type == 'labelled'].drop(
            columns='type')
        # unlabelled feat
        self.unlabel_features_df = self.merge_dat.drop(columns='Y_0')[self.merge_dat.type == 'unlabelled'].drop(
            columns='type')
        # validation feat
        self.val_features_df = self.merge_dat.drop(columns='Y_0')[self.merge_dat.type == 'validation'].drop(
            columns='type')
        # labelled label
        self.label_y_df = self.merge_dat.filter(items=['Y_0', 'type'])[self.merge_dat.type == 'labelled'].drop(
            columns='type')
        # unlabelled label
        self.unlabel_y_df = self.merge_dat.filter(items=['Y_0', 'type'])[self.merge_dat.type == 'unlabelled'].drop(
            columns='type')
        # validation label
        self.val_y_df = self.merge_dat.filter(items=['Y_0', 'type'])[self.merge_dat.type == 'validation'].drop(
            columns='type')
        return (self)

    def subset_tensors(self):

        all_vars = [c for c in self.merge_dat.columns]

        self.feature_varnames = [v for v in all_vars if 'X' in v]

        self.class_varname = [v for v in all_vars if 'Y' in v]

        # set_trace()

        # labelled

        self.train_label_dataset = self.merge_dat[self.merge_dat['type'] == 'labelled']
        # print(self.train_label_dataset)

        self.train_unlabel_dataset = self.merge_dat[self.merge_dat['type'] == 'unlabelled']

        self.val_dataset = self.merge_dat[self.merge_dat['type'] == 'validation']

        # -----------------------------------
        # split into features and class label
        # -----------------------------------

        # -----------
        # labelled
        # -----------
        self.label_features = torch.Tensor(self.train_label_dataset[self.feature_varnames].values).float()
        self.label_y = torch.Tensor(self.train_label_dataset[self.class_varname].values)

        # -----------
        # unlabelled
        # -----------

        self.unlabel_features = torch.Tensor(self.train_unlabel_dataset[self.feature_varnames].values).float()
        self.unlabel_y = torch.Tensor(self.train_unlabel_dataset[self.class_varname].values)

        # -----------
        # validation
        # -----------

        self.val_features = torch.Tensor(self.val_dataset[self.feature_varnames].values).float()
        self.val_y = torch.Tensor(self.val_dataset[self.class_varname].values)

        # --------------------
        # convert y to one-hot and squeeze
        # --------------------

        self.label_y = torch.nn.functional.one_hot(self.label_y.type(torch.LongTensor), 2).squeeze(1)
        self.unlabel_y = torch.nn.functional.one_hot(self.unlabel_y.type(torch.LongTensor), 2).squeeze(1)
        self.val_y = torch.nn.functional.one_hot(self.val_y.type(torch.LongTensor), 2).squeeze(1)

        return (self)

# plain ds class
class ds:
    def __init__(self,adj_mat,labels):
        self.adj_mat=adj_mat
        self.labels=labels
        self.dag=igraph.Graph.Adjacency(self.adj_mat)

# mixture of gaussian
class dsc_mog:
    def __init__(self,
        adjacency_matrix,
        var_types,
        merge_dat,
        labels):
        self.adj_mat=np.array(adjacency_matrix) #store adjacency matrix
        self.dag=igraph.Graph.Adjacency(self.adj_mat) #get dag
        self.labels=labels
        self.variable_types=var_types
        self.feature_dim=2
        self.merge_dat=merge_dat

# save synthetic data to .csv file
def save_synthetic_data(joined_synthetic_data, d_n, s_i, master_spec, dspec, algo_spec, synthetic_data_dir):
    # save out data
    # synthetic data here, it must happen
    s_dir = "{0}/{1}".format(dspec.save_folder, synthetic_data_dir)
    chk_folder = os.path.isdir(s_dir)
    if not chk_folder:
        os.makedirs(s_dir)
    print('saving synth dat')
    joined_synthetic_data.to_csv(
        "{0}/{3}/synthetic_data_d_n_{1}_s_i_{2}.csv".format(dspec.save_folder, d_n, s_i, synthetic_data_dir))
    print('synth dat saved for d_n: {0} s_i: {1}'.format(d_n, s_i))




def return_mb_dict(dag):

    mb_dict = {}  # getting markov blanket
    for n in dag.nodes:
        mb_dict[n] = {}
        # get parents
        mb_dict[n]['parent'] = list(set([n for n in dag.predecessors(n)]))
        # get children
        mb_dict[n]['children'] = list(set([n for n in dag.successors(n)]))
        # spouses
        mb_dict[n]['spouses'] = list(set(reduce_list([[s for s in dag.predecessors(c)] for c in mb_dict[n]['children']])))

        #remove self from spouse

        try:
            mb_dict[n]['spouses'].remove(n)
        except:
            print('curernt node: {0} not in spouses'.format(n))
        finally:
            print('continuing..')

    return(mb_dict)

def reduce_list(in_list):
    return (sum(in_list, []))



# plot synthetic and real data together on plotly.express for visual confirmation of modelled dist
def plot_synthetic_and_real_data(hpms_dict, dsc, args, s_i, joined_synthetic_data, synthetic_data_dir,dspec):
    print('constructing plot for real and syntheitc data...')
    d = hpms_dict
    out_str = "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in d.items()) + "}"
    img = Image.new('RGB', (600, 600), color=(73, 109, 137))
    fnt = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    d.text((10, 10), out_str, font=fnt, fill=(255, 255, 0))
    # saving out hyperparameters....
    # temporary directory for our plots
    tmp_plot_dir = 'TEMPORARY_PLOT_DIR_d_n_{0}_s_i_{1}'.format(args.d_n, s_i)
    # check existence purge if so
    if os.path.exists(tmp_plot_dir):
        shutil.rmtree(tmp_plot_dir)
    # make new dir
    os.mkdir(tmp_plot_dir)
    img.save('./{0}/hyperparameters.png'.format(tmp_plot_dir))
    # now write out hsmps_dict to a png file...
    order_to_generate = dsc.dag.topological_sorting()
    # need to get class variable name
    lab_i = np.where([d == 'label' for d in dsc.variable_types])[0][0]
    label_var_name = dsc.labels[lab_i]
    # before we do that, need to merge synthetic  data w train data

    scols = [s.replace('_0', '') for s in joined_synthetic_data.columns]
    joined_synthetic_data.columns = scols
    synthetic_and_orig_data = pd.concat([dsc.merge_dat, joined_synthetic_data], axis=0)

    synthetic_and_orig_data=synthetic_and_orig_data.sample(frac=1) #shuffle for random
    # and then we should be able to use facet type to plot 4 varying conditions:
    # unlabel,label,validation,synthetic

    for v_i in order_to_generate:

        # get ancestors for this variable
        source_edges = dsc.dag.es.select(_target=v_i)
        source_vertices = [s_e.source_vertex for s_e in source_edges]
        sv = [v.index for v in source_vertices]  # get index of them

        current_variable_label = dsc.labels[v_i]

        # label var...
        if dsc.variable_types[v_i] == 'label':
            xv = current_variable_label
            yv = ""
            fig = px.histogram(synthetic_and_orig_data,
                               x=current_variable_label,
                               color=current_variable_label,
                               facet_col="type")
            fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)
            # and get source also

            for s in sv:
                current_source_variable = dsc.labels[s]
                xv = current_source_variable
                yv = current_variable_label

                # transpose x,y even tho x->y so we get better resolution in X
                # because the plots are strteched such that Y is larger than X
                fig = px.scatter(synthetic_and_orig_data,
                                 x=yv,
                                 y=xv,
                                 color=label_var_name, facet_col="type",
                                 color_continuous_scale="bluered", opacity=0.3)
                fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)



        elif dsc.variable_types[v_i] != 'label' and len(sv) == 0:  # len source vertices,sv, ==0
            # no ancestor/causal variables
            xv = current_variable_label
            yv = ""
            fig = px.histogram(synthetic_and_orig_data,
                               x=current_variable_label,
                               color=label_var_name,
                               facet_col="type",
                               histnorm='probability density',
                               color_discrete_map={0: "blue", 1: "red"}
                               )

            fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)
        else:
            # we have conditional variables, but get the marginal dist first
            xv = current_variable_label
            yv = ""
            fig = px.histogram(synthetic_and_orig_data,
                               x=current_variable_label,
                               color=label_var_name,
                               facet_col="type",
                               histnorm='probability density',
                               color_discrete_map={0: "blue", 1: "red"})
            fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)

            for s in sv:
                current_source_variable = dsc.labels[s]
                # scatter plot of cause/antecedent
                xv = current_source_variable
                yv = current_variable_label

                fig = px.scatter(synthetic_and_orig_data,
                                 x=current_source_variable,
                                 y=current_variable_label,
                                 color=label_var_name, facet_col="type",
                                 color_continuous_scale="bluered", opacity=0.3)
                fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)

    # split into unlabelled/validation/synthetic and plot
    # get all temporary
    output_img = glob.glob('./{0}/*.png'.format(tmp_plot_dir))
    output_img = [o for o in output_img if 'hyperparameters' not in o]  # removing pil_text_font...
    output_img = [Image.open(o) for o in output_img]

    images = output_img
    widths, heights = zip(*(i.size for i in images))

    dict_img = Image.open('./{0}/hyperparameters.png'.format(tmp_plot_dir))

    total_width = sum(widths)
    max_height = max(heights)

    total_images = len(output_img)  # use this to apportion width / height, based on how many plot we want...

    # take square root of this...and round up..
    sqrt_img = int(np.sqrt(total_images)) + 1  # round up by adding '1'

    new_image_height = max(heights) * sqrt_img
    new_image_width = max(widths) * sqrt_img

    new_im = Image.new('RGB', (new_image_width, new_image_height + dict_img.size[1]))
    x_offset = 0
    y_offset = 0

    im_idx = 0

    for r in range(sqrt_img):
        for c in range(sqrt_img):
            if im_idx < total_images:
                im = images[im_idx]
                new_im.paste(im, (x_offset, y_offset))
                x_offset += im.size[0]
                im_idx += 1
        y_offset += im.size[1]
        x_offset = 0

    new_im.paste(dict_img, (y_offset, int(new_image_height + dict_img.size[1] / 2)))

    # num_already = len(os.listdir(dspec.save_folder + '/TEMPORARY_PLOT_DIR/combined_plots/'))

    new_im.save('./{0}/combined.png'.format(tmp_plot_dir))

    new_im.save("{0}/{3}/synthetic_data_d_n_{1}_s_i_{2}_plot.png".format(dspec.save_folder, args.d_n, s_i,
                                                                         synthetic_data_dir))

    # now delete the directory (if poss)
    try:
        print('now removing plot dir if exists')
        shutil.rmtree(tmp_plot_dir)
    except:
        print('error no plot directory to remove')
    finally:
        print('exiting...')

    print('plot for synthetic and real data finished')



# plotly scatter plot
# 2 dimensional fetures
# plot DAG also
import shutil
# turn off interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px

def plot_2d_data_w_dag(dsc, s_i,synthetic_data_dir='synthetic_data'):
    print('constructing 2d plots')
    tmp_plot_dir = 'TEMPORARY_PLOT_DIR_d_n_{0}_s_i_{1}'.format(dsc.d_n, s_i)
    # check existence purge if so
    if os.path.exists(tmp_plot_dir):
        shutil.rmtree(tmp_plot_dir)
    # make new dir
    os.mkdir(tmp_plot_dir)
    # now write out hsmps_dict to a png file...
    order_to_generate = dsc.dag.topological_sorting()
    # need to get class variable name
    lab_i = np.where([d == 'label' for d in dsc.var_types])[0][0]
    label_var_name = dsc.var_names[lab_i]
    # before we do that, need to merge synthetic  data w train data
    # get idx of feature variable
    dsc.feature_alphan = {}
    dsc.fdict = {v: t for v, t in zip(dsc.var_names, dsc.var_types)}
    dsc.feature_alphan = copy.deepcopy(dsc.fdict)

    # https://stackoverflow.com/questions/13838405/custom-sorting-in-pandas-dataframe
    # preserve orig order of names in column type.
    # after randomisation, resort on this one to make sure that
    # the data still preserves order when  using 'facet_col' option in plotly
    orig_type_order=dsc.merge_dat.type.unique()
    dsc.merge_dat['type'] = pd.Categorical(dsc.merge_dat['type'], orig_type_order)

    dsc.merge_dat=dsc.merge_dat.sample(frac=1) #randomise the data frame for plotting

    # now re-sort, as specified earlier
    dsc.merge_dat=dsc.merge_dat.sort_values('type')

    for x in dsc.feature_alphan.keys():
        relevant_features = []
        fsplit = [f.split('_')[0] for f in dsc.feature_varnames]
        for fsplit, f in zip(fsplit, dsc.feature_varnames):
            if fsplit == x:
                relevant_features.append(f)
        dsc.feature_alphan[x] = relevant_features


    for v_i in order_to_generate:
        # get ancestors for this variable
        source_edges = dsc.dag.es.select(_target=v_i)
        source_vertices = [s_e.source_vertex for s_e in source_edges]
        sv = [v.index for v in source_vertices]  # get index of them
        current_variable_label = dsc.var_names[v_i]
        # label var...
        if dsc.var_types[v_i] == 'label':
            xv = current_variable_label
            yv = ""
            fig = px.histogram(dsc.merge_dat,
                               x=current_variable_label,
                               color=current_variable_label,
                               facet_col="type")
            fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)
            # and get source also
            for s in sv:
                current_source_variable = dsc.var_names[s]
                xv = dsc.label_names_alphan[current_source_variable][0]
                yv = dsc.label_names_alphan[current_source_variable][1]
                fig = px.scatter(dsc.merge_dat,
                                 x=xv,
                                 y=yv,
                                 color=label_var_name, facet_col="type",
                                 color_continuous_scale="bluered_r", opacity=0.1)
                fig.update_yaxes(
                    scaleanchor="x",
                    scaleratio=1,
                )
                fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)

        else:
            # we have conditional variables, but get the marginal dist first
            xv = dsc.label_names_alphan[current_variable_label][0]
            yv = dsc.label_names_alphan[current_variable_label][1]
            # transpose x,y even tho x->y so we get better resolution in X
            # because the plots are strteched such that Y is larger than X
            fig = px.scatter(dsc.merge_dat,
                             x=xv,
                             y=yv,
                             color=label_var_name, facet_col="type",
                             color_continuous_scale="bluered_r", opacity=0.1)
            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )
            fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)

    # save image of DAG
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 18.5)
    # plot(g, layout=layout, target=ax)

    # plt.clf()

    # fig = plt.figure()

    # plt.close(fig)

    print('graph structrure for dataset: {0}'.format(dsc.d_n))

    ig.plot(dsc.dag,
            vertex_label=dsc.var_names,
            vertex_label_size=50,
            vertex_size=50,
            edge_width=5,
            edge_arrow_width=5,
            vertex_color='green',
            vertex_shape='circle',
            layout=dsc.dag.layout_grid(), target=ax)

    fig.savefig("./{0}/dn_{1}_si_{2}_DAG.png".format(tmp_plot_dir, dsc.d_n, dsc.s_i))

    # split into unlabelled/validation/synthetic and plot
    # get all temporary
    import glob
    from PIL import Image
    output_img = glob.glob('./{0}/*.png'.format(tmp_plot_dir))
    output_img = [o for o in output_img if 'hyperparameters' not in o]  # removing pil_text_font...
    output_img = [Image.open(o) for o in output_img]

    images = output_img
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    total_images = len(output_img)  # use this to apportion width / height, based on how many plot we want...

    # take square root of this...and round up..
    sqrt_img = int(np.sqrt(total_images)) + 1  # round up by adding '1'

    new_image_height = max(heights) * sqrt_img
    new_image_width = max(widths) * sqrt_img

    new_im = Image.new('RGB', (new_image_width, new_image_height))
    x_offset = 0
    y_offset = 0

    im_idx = 0

    for r in range(sqrt_img):
        for c in range(sqrt_img):
            if im_idx < total_images:
                im = images[im_idx]
                new_im.paste(im, (x_offset, y_offset))
                x_offset += im.size[0]
                im_idx += 1
        y_offset += im.size[1]
        x_offset = 0

    # new_im.paste(dict_img, (y_offset, int(new_image_height/ 2)))

    # num_already = len(os.listdir(dspec.save_folder + '/TEMPORARY_PLOT_DIR/combined_plots/'))

    new_im.save('./{0}/combined.png'.format(tmp_plot_dir))
    # read in spec to set seed...
    master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)
    # write dataset spec shorthand
    dspec = master_spec['dataset_spec']
    dspec.set_index("d_n", inplace=True)  # set this index for easier
    dspec = dspec.loc[dsc.d_n]  # use this as reference..

    new_im.save("{0}/{3}/synthetic_data_d_n_{1}_s_i_{2}_plot.png".format(dspec.save_folder, dsc.d_n, s_i,synthetic_data_dir))

    # now delete the directory (if poss)
    try:
        print('now removing plot dir if exists')
        shutil.rmtree(tmp_plot_dir)
    except:
        print('error no plot directory to remove')
    finally:
        print('exiting...')

    print('plot for synthetic and real data finished')


def plot_2d_single_variable_data_type(dsc,s_i,data_type,variable_name,synthetic_data_dir='synthetic_data'):
    print('constructing 2d plots')
    tmp_plot_dir = 'TEMPORARY_PLOT_DIR_d_n_{0}_s_i_{1}'.format(dsc.d_n, s_i)
    # check existence purge if so
    if os.path.exists(tmp_plot_dir):
        shutil.rmtree(tmp_plot_dir)
    # make new dir
    os.mkdir(tmp_plot_dir)
    # now write out hsmps_dict to a png file...
    #order_to_generate = dsc.dag.topological_sorting()
    # need to get class variable name
    lab_i = np.where([d == 'label' for d in dsc.var_types])[0][0]
    label_var_name = dsc.var_names[lab_i]
    # before we do that, need to merge synthetic  data w train data
    # get idx of feature variable
    dsc.feature_alphan = {}
    dsc.fdict = {v: t for v, t in zip(dsc.var_names, dsc.var_types)}
    dsc.feature_alphan = copy.deepcopy(dsc.fdict)

    # https://stackoverflow.com/questions/13838405/custom-sorting-in-pandas-dataframe
    # preserve orig order of names in column type.
    # after randomisation, resort on this one to make sure that
    # the data still preserves order when  using 'facet_col' option in plotly
    orig_type_order=dsc.merge_dat.type.unique()
    dsc.merge_dat['type'] = pd.Categorical(dsc.merge_dat['type'], orig_type_order)

    dsc.merge_dat=dsc.merge_dat.sample(frac=1) #randomise the data frame for plotting

    # now re-sort, as specified earlier
    dsc.merge_dat=dsc.merge_dat.sort_values('type')

    # convert class label to string
    dsc.merge_dat[label_var_name] = dsc.merge_dat[label_var_name].astype(str)  # convert to string

    color_discrete_map = {"0": "red",
                             "1": "blue"}

    for x in dsc.feature_alphan.keys():
        relevant_features = []
        fsplit = [f.split('_')[0] for f in dsc.feature_varnames]
        for fsplit, f in zip(fsplit, dsc.feature_varnames):
            if fsplit == x:
                relevant_features.append(f)
        dsc.feature_alphan[x] = relevant_features

    #subset to data type

    #subset to variable name

    current_variable_label = variable_name
    # we have conditional variables, but get the marginal dist first
    xv = dsc.label_names_alphan[current_variable_label][0]
    yv = dsc.label_names_alphan[current_variable_label][1]


    fig = px.scatter(dsc.merge_dat[dsc.merge_dat['type']==data_type],
                     x=xv,
                     y=yv,
                     color=label_var_name,
                     color_discrete_map=color_discrete_map,
                     color_continuous_scale="bluered_r", opacity=0.5)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # get this into the features dict
    features_dict={
        'X1_0':'X_{1,0}',
        'X1_1': 'X_{1,1}',
        'X2_0': 'X_{2,0}',
        'X2_1': 'X_{2,1}',
        'X3_0': 'X_{3,0}',
        'X3_1': 'X_{3,1}'

    }

    fig.update_layout(
        xaxis_title=r'${0}$'.format(features_dict[xv]),
        yaxis_title=r'${0}$'.format(features_dict[yv]),
        title_text=data_type.title(),
        title_x=0.5
    )

    master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)
    # write dataset spec shorthand
    dspec = master_spec['dataset_spec']
    dspec.set_index("d_n", inplace=True)  # set this index for easier
    dspec = dspec.loc[dsc.d_n]  # use this as reference..



    im_save_fn="{0}/{3}/synthetic_data_d_n_{1}_s_i_{2}_single_{4}_type_{5}.png".format(dspec.save_folder, dsc.d_n, s_i,synthetic_data_dir,variable_name,data_type)

    fig.update_traces(marker={'size': 10})

    fig.write_image(im_save_fn, scale=5)

    print(f'image written for: {variable_name},\t {data_type}')



#-----------------------------------
#     file management
#-----------------------------------

# return save location for data
def get_dataset_folder(d_n):
    if d_n in [1,2,3,4,5]:
        return('dataset_dn{0}_wishart'.format(d_n))
    elif 'gaussian_mixture' in d_n:
        return('dataset_{0}'.format(d_n))
    else:
        return('dataset_{0}'.format(d_n))

# return formatted concatented name of saved model
def return_saved_model_name(model_name,save_dir,d_n,s_i):
    #old_fn_match='{0}-s_i={1}-epoch='.format(model_name,s_i)
    old_models=glob.glob(save_dir+'/'+'saved_models'+'/*')
    old_fn_match='{0}-s_i={2}-epoch='.format(model_name,d_n,s_i)
    #old_models_extra=glob.glob(save_dir+'/'+'saved_models'+'/*')
    #old_models=old_models+old_models_extra
    saved_model_names=[m for m in old_models if old_fn_match in m]
    return(saved_model_names)

# combine model name with dn / si for saving
def combined_name(model_name,d_n,s_i):
    return(f'{model_name}_dn_{d_n}_si_{s_i}')

# create model name of causal gan - we have a few variants
def create_model_name(labels,algo_variant):
    retval='GEN'
    if algo_variant=='basic':
        retval+='_BASIC_'
    elif algo_variant=='basic_disjoint':
        retval+='_BASIC_DJ_'
    elif algo_variant=='marginal':
        retval+='_YBP_'
    elif algo_variant=='gumbel':
        retval+='_GUMBEL_'
    elif algo_variant=='gumbel_disjoint':
        retval+='_GUMBEL_DJ_'
    elif algo_variant=='gumbel_disjoint_xces':
        retval+='_GUMBEL_DJ_XCES_'
    elif algo_variant=='gumbel_tc_sim':
        retval+='_GUMBEL_TC_SIM_'
    elif algo_variant=='gumbel_tc_sim_bce':
        retval+='_GUMBEL_TC_SIM_BCE_'
    elif algo_variant=='gumbel_mintemp':
        retval+='_GUMBEL_MINTEMP_'
    elif algo_variant == 'gumbel_ulab':
        retval += '_GUMBEL_ULAB_'
    else:
        assert(1==0)
    retval+=labels
    return(retval)

# delete old saved models
def delete_old_saved_models(model_name, save_dir, s_i):
    old_fn_match = f'{model_name}-s_i={s_i}-epoch='
    old_models = glob.glob(save_dir + '/' + 'saved_models' + '/*')
    delete_models = [m for m in old_models if old_fn_match in m]
    #print(delete_models)
    for d in delete_models:
        os.remove(d)



import numpy as np
from sklearn.neighbors import NearestNeighbors

#-----------------------------------
#     misc functions
#-----------------------------------

# get median pairwise distance of in_tensor
# def get_median_pwd(in_tensor):
    
#     npi=[d for d in pairwise_distances_chunked(in_tensor.cpu().detach().numpy().astype(np.float32))]
#     retval = np.median(npi)
#     return (retval)

# get median pairwise distance of in_tensor
def get_median_pwd(in_tensor):
    # Number of neighbors to consider for approximate median calculation
    n_neighbors = 10

    # Create a nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    
    data=in_tensor.cpu().detach().numpy()
    nn.fit(data)

    # Query neighbors for each point
    distances, _ = nn.kneighbors(data)

    # Compute median pairwise distances for each point
    median_distances = np.median(distances, axis=1)

    # Compute the overall median
    overall_median_distance = np.median(median_distances)
    
    return (overall_median_distance)
    
    #using approxi method with neareast neighoburts to speed upe
    
    #npi=[d for d in pairwise_distances_chunked(in_tensor.cpu().detach().numpy().astype(np.float32))]
    #retval = np.median(npi)
    #return (retval)


    # # Generate random data (replace this with your actual data)
    # n_samples = 1000
    # n_features = 10
    # data = np.random.rand(n_samples, n_features)



    #print("Overall Median Pairwise Distance:", overall_median_distance)


# plot the dag
def plotting_dag(ds,dn):
    print('graph structrure for dataset: {0}'.format(dn))
    retg=ig.plot(ds.dag, 
    bbox=(0, 0, 300, 300),
    vertex_label=ds.labels,
    vertex_label_size=13,
    vertex_size=30,
    vertex_color='white',
    layout=ds.dag.layout_grid())
    return(retg)

# convert string to boolean value true false
def str_to_bool(in_str):
    if in_str=='True':
        return(True)
    if in_str=='False':
        return(False)
    else:
        assert(1==0)
        return(None)

# initialise weights for new network
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

        #   net.apply(init_weights)



