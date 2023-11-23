from torch.utils.data import sampler
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from IPython.core.debugger import set_trace
import glob
import pandas as pd
import copy
from pytorch_lightning import loggers as pl_loggers
import argparse
import time
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
from coloraide import Color # coloraide used to create continuous colour scale for plot of decision boundary,https://pypi.org/project/coloraide/
from PIL import Image
import json

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import sys


import sys

import sys
import os

# Get the directory containing the current script/module using __file__
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script's directory to sys.path
sys.path.append(script_dir)



sys.path.append(f'{script_dir}/generative_models/')


#print(sys.path)
from benchmarks_cgan import load_dsc, manipulate_dsc, load_real_data_legacy

# determinstic seeding for pytorch lightning model training
from pytorch_lightning import Trainer, seed_everything
RANDOM_SEED=999
seed_everything(RANDOM_SEED, workers=True)
DETERMINISTIC_FLAG=True # will be imported by benchmark modules


#-----------------------------------
#     TURN OFF SOME DEFAULT WARNINGS
#-----------------------------------


# https://github.com/PyTorchLightning/pytorch-lightning/issues/10182
# turning off the warnings:
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)




#-----------------------------------
#     INITIALISATIONS
#-----------------------------------

LOG_ON_STEP=False                   #if we are logging on step or not
CHECK_ON_TRAIN_END=not LOG_ON_STEP  #if we check on train end

rng = np.random.default_rng(12345)  #initialise numpy random number generator object


#-----------------------------------
#     MISCELLANEOUS HELPER FUNCTIONS
#-----------------------------------



#string to boolean
def str_to_bool(in_str):
    if in_str=='True':
        return(True)
    if in_str=='False':
        return(False)
    else:
        assert(1==0)
        return(None)

#combine model name with dn / si for saving
def combined_name(model_name,d_n,s_i):
    return(f'{model_name}_dn_{d_n}_si_{s_i}')

#delete old saved models before starting new training run
def clear_saved_models(model_name, save_dir, s_i):
    old_fn_match = f'{model_name}*-s_i={s_i}-*.ckpt'
    old_models = glob.glob(f'{save_dir}/saved_models/{old_fn_match}')
    delete_models = [m for m in old_models]
    for d in delete_models:
        os.remove(d)
    return

def create_model_save_name(optimal_model,optimal_trainer,dspec):
    filepath = "{0}/saved_models/{1}-s_i={2}-epoch={3}-val_acc={4}.ckpt".format(dspec.save_folder,
                                                                                optimal_model.model_name,
                                                                                optimal_trainer.model.hparams[
                                                                                    's_i'],
                                                                                10,
                                                                                max(optimal_trainer.model.val_accs))
    return(filepath)


#loads data from pytorch tensor files saved as '.pt'
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

#gpu kwargs if train on gpu
def get_gpu_kwargs(args):
    gpu_kwargs = {}
    has_gpu = False
    if args.use_gpu:
        has_gpu = torch.cuda.is_available()
        if has_gpu == False:
            print('gpu flag but no gpu, using cpu instead')
        else:
            gpu_kwargs = {'gpus': torch.cuda.device_count()} if has_gpu else {}
            if torch.cuda.device_count() > 1:
                gpu_kwargs['accelerator'] = "dp"
    return(gpu_kwargs)

#-----------------------------------
#     NEURAL NET ARCHITECTURES
#-----------------------------------

#returns standard net architecture
#change to wider shape based on discussion w HJ/MMG 23_02_2022
def get_standard_net(input_dim,output_dim):
    net = nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Linear(100, 5),
        nn.ReLU(),
        nn.Linear(5, output_dim)
    )
    return(net)

#returns wide net architecture. note: this is now default as standard net
def get_wide_net(input_dim,output_dim):
    net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
            nn.ReLU(),
            nn.Linear(5, output_dim)
        )
    return(net)

#get tanh net for nonlin decision boundary
def get_tanh_net_synthetic(input_dim,output_dim):
    net = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, output_dim),
        )
    return(net)


#-----------------------------------
#     NEURAL NET INITIALISATION
#-----------------------------------


# xavier uniform initialisation
def init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

#use he_kaiming initialisation, ie:
#https://arxiv.org/abs/1502.01852v1
def init_weights_he_kaiming(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)

#zero bias used for nonlinear decision boundary generator
def init_weights_zero_bias(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        #torch.nn.init.uniform_(m.weight, a=-1.0, b=1.0)
        m.bias.data.fill_(0)
        #   net.apply(init_weights)

#normal and zero bias
def init_weights_normal_zero_bias(m):
    if isinstance(m, nn.Linear):
        m.data.normal_(0, 0.001)
        if m.ndimension() == 1:
            m.data.fill_(0.)

#-----------------------------------
#     CHECKPOINTS
#-----------------------------------

#returns model checkpoint to save max performing model
def return_chkpt_max_val_acc(model_name, data_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="{0}/saved_models".format(data_dir),
        filename=model_name+ "-{s_i:.0f}-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
    )
    return(checkpoint_callback)

#returns model checkpoint to save checkpoint
#for model with min mmd loss
def return_chkpt_min_mmd(model_name):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mmd",
        dirpath="saved_models/",
        filename=model_name+ "-{d_n:.0f}-{s_i:.0f}-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="min",
    )
    return(checkpoint_callback)


#-----------------------------------
#     EARLY STOPPING
#-----------------------------------

# early stopping train loss
def return_estop_train_loss(patience):
    earlystop_tl = EarlyStopping(monitor="train_loss",
                                                    min_delta=0.00,
                                                    patience=patience,
                                                    verbose=False,
                                                    check_on_train_epoch_end=CHECK_ON_TRAIN_END,
                                                    mode="min",
                                        check_finite=True)
    return(earlystop_tl)

# early stopping val accuracy
def return_estop_val_acc(patience):
    earlystop_val_acc = EarlyStopping(monitor="val_acc",
                                                    min_delta=0.00,
                                                    patience=patience,
                                                    verbose=False,
                                                    stopping_threshold=1.0,
                                                    check_on_train_epoch_end=CHECK_ON_TRAIN_END,
                                                    mode="max",
                                        check_finite=True)
    return(earlystop_val_acc)

# early stopping val binary cross entropy
def return_estop_val_bce(patience):
    earlystop_val_bce = EarlyStopping(monitor="val_bce",
                                                    min_delta=0.00,
                                                    patience=patience,
                                                    verbose=False,
                                                    check_on_train_epoch_end=CHECK_ON_TRAIN_END,
                                                    mode="min",
                                      check_finite=True)
    return(earlystop_val_bce)



#-----------------------------------
#     TENSORBOARD LOGGING
#-----------------------------------

def get_default_logger(model_name, d_n,s_i, t):
    tb_logger = pl_loggers.TensorBoardLogger("lightning_logs/",
                                             name=combined_name(model_name, d_n, s_i),
                                             version=t)
    return (tb_logger)


#-----------------------------------
#     TRAINING
#-----------------------------------

def get_default_trainer(args, tb_logger, callbacks,deterministic_flag=False,min_epochs=0,**gpu_kwargs):
    trainer = pl.Trainer(log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         max_epochs=args.n_iterations,
                         callbacks=callbacks,
                         deterministic=deterministic_flag,
                         #reload_dataloaders_every_epoch=True,
                         reload_dataloaders_every_n_epochs=1,
                         logger=tb_logger,
                         min_epochs=min_epochs,
                         accelerator='gpu',
                         devices=1,
                         enable_progress_bar=True,#progress_bar_refresh_rate=1,
                         #weights_summary=None,
                         **gpu_kwargs)
    return (trainer)

                                #deterministic=True,
                                #logger=tb_logger,

                                #profiler=profiler_simple_first,
                                #**gpu_kwargs)

#-----------------------------------
#     LOADING OPTIMAL MODEL
#-----------------------------------

def load_optimal_model(dspec, current_model):
    # load optimal model
    model_to_search_for = dspec.save_folder + '/saved_models/' + current_model.model_name + "*-s_i={0}-epoch*".format(
        current_model.hparams['s_i'])
    candidate_models = glob.glob(model_to_search_for)
    #assert(len(candidate_models)==1)
    current_model = type(current_model).load_from_checkpoint(checkpoint_path=candidate_models[0])
    return (current_model)

#-----------------------------------
#     COMPARE TWO MODELS FOR OPTIMAL
#-----------------------------------

def return_optimal_model(current_model, trainer, optimal_model,optimal_trainer,val_features,val_lab,metric='val_acc'):
    if optimal_model == None:
        print('optimal model created')
        return(current_model,trainer)
    else:
        optimal_model.eval()
        current_model.eval()

        optimal_pred = optimal_model.forward(val_features)
        optimal_acc = get_accuracy(optimal_pred, val_lab)

        current_pred = current_model.forward(val_features)
        current_acc = get_accuracy(current_pred, val_lab)

        optimal_pred = optimal_model.forward(val_features)
        optimal_bce = get_bce_w_logit(optimal_pred, torch.nn.functional.one_hot(val_lab).float())

        current_pred = current_model.forward(val_features)
        current_bce = get_bce_w_logit(current_pred, torch.nn.functional.one_hot(val_lab).float())

        if ((current_acc > optimal_acc) and metric=='val_acc') or ((current_bce > optimal_bce) and metric=='val_bce'):
            optimal_model = current_model
            optimal_trainer = trainer
            print('optimal model overwritten')
            print('old optimal: {0}'.format(optimal_acc))
            print('new optimal: {0}'.format(current_acc))

        else:
            print('optimal model NOT overwritten')
            print('old optimal: {0}'.format(optimal_acc))
            print('new optimal: {0}'.format(current_acc))

        return (optimal_model, optimal_trainer)

#-----------------------------------
#     MODEL EVALUATION
#-----------------------------------

def evaluate_on_test_and_unlabel(dspec, args, si_iter,current_model,optimal_model,orig_data,optimal_trainer):

    dsc_loader = eval(dspec.dataloader_function)  # within the spec
    dsc = dsc_loader(args, si_iter, dspec)
    dsc = manipulate_dsc(dsc, dspec)
    dsc.s_i = si_iter

    optimal_model.eval()
    
    test_features=torch.tensor(orig_data['test_features'],device='cuda')
    test_y=torch.tensor(orig_data['test_y'],device='cuda')
    ulab_features=torch.tensor(orig_data['unlabel_features'],device='cuda')
    ulab_y=torch.tensor(orig_data['unlabel_y'],device='cuda')

    with torch.no_grad():

        test_acc = optimal_model.predict_test(test_features, torch.argmax(test_y, 1))
        unlabel_acc = optimal_model.predict_test(ulab_features,torch.argmax(ulab_y, 1))

    test_acc = np.array([test_acc.cpu().detach().item()])
    filepath = "{0}/saved_models/{1}-s_i={2}_test_acc.out".format(dspec.save_folder,
                                                                  current_model.model_name,
                                                                  optimal_trainer.model.hparams['s_i'])
    np.savetxt(filepath, test_acc)

    unlabel_acc = np.array([unlabel_acc.cpu().detach().item()])
    filepath = "{0}/saved_models/{1}-s_i={2}_unlabel_acc.out".format(dspec.save_folder,
                                                                     current_model.model_name,
                                                                     optimal_trainer.model.hparams['s_i'])
    np.savetxt(filepath, unlabel_acc)

    print(f'test_acc: {test_acc}')
    print(f'unlabel_acc: {unlabel_acc}')




#-----------------------------------
#     MISC_SQL
#-----------------------------------



#get working device for storing SQL record
def get_sql_working_device():
    cpath=os.getcwd()
    if '/Users/macuser/' in cpath:
        working_device='macbook_m1'
    else:
        working_device='spartan_gpu'
    return(working_device)


#return test/ulab acc for SQL table
def return_test_ulab_for_sql(dspec, args, si_iter,current_model,optimal_model,orig_data,optimal_trainer):

    dsc_loader = eval(dspec.dataloader_function)  # within the spec
    dsc = dsc_loader(args, si_iter, dspec)
    dsc = manipulate_dsc(dsc, dspec)
    dsc.s_i = si_iter

    optimal_model.eval()

    test_acc = optimal_model.predict_test(orig_data['test_features'], torch.argmax(orig_data['test_y'], 1))
    unlabel_acc = optimal_model.predict_test(orig_data['unlabel_features'],
                                             torch.argmax(orig_data['unlabel_y'], 1))

    test_acc = np.array([test_acc.cpu().detach().item()])
    #filepath = "{0}/saved_models/{1}-s_i={2}_test_acc.out".format(dspec.save_folder,
    #                                                              current_model.model_name,
    #                                                              optimal_trainer.model.hparams['s_i'])
    #np.savetxt(filepath, test_acc)

    unlabel_acc = np.array([unlabel_acc.cpu().detach().item()])

    resdict={'test_acc':test_acc,
            'unlabel_acc':unlabel_acc}

    return(resdict)



#-----------------------------------
#     METRICS
#-----------------------------------


# new accuracy method because there is bug in pytorch lightnign
def get_accuracy(pred, true):
    accuracy = (pred.argmax(1) == true).float().mean()
    return (accuracy)

#get binary cross entropy w logit
def get_bce_w_logit(pred, true):
    lfunc=torch.nn.BCEWithLogitsLoss()
    bce_loss=lfunc(pred,true)
    return(bce_loss)

# note this is pytorch lightning default and seems to crash sometimes
# so it's easier to use custom-defined function 'get_accuracy'
#accuracy = torchmetrics.Accuracy()

# softmax pred
get_softmax = torch.nn.Softmax(dim=1)  # need this to convert classifier predictions

#-----------------------------------
#     PLOTTING DECISION BOUNDARIES
#-----------------------------------

# USING THE MACHINE LEARNING TOOLS LIBRARY ie, with function ```plot_decision_regions```

#get decision boundary dicts to plot the decision boundary

def get_dec_dicts(model_name, current_model, dspec, dsc):
    # wrangling data

    print('pausing here')

    lab_dat=dsc.merge_dat[dsc.merge_dat.type == 'labelled']
    ulab_dat=dsc.merge_dat[dsc.merge_dat.type == 'unlabelled']
    all_dat=dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])]

    ylab = dsc.label_var
    all_db_dicts={}

    list_all_db_dicts=[]


    for xv in dsc.feature_names:
        #get x subset
        xlabs=dsc.label_names_alphan[xv]

        if len(xlabs)!=2:
            print('error need features of dimension 2')
            assert(1==0)

        Xl=lab_dat[xlabs].values
        yl=lab_dat[[ylab]].values.flatten()

        Xul = ulab_dat[xlabs].values
        yul = ulab_dat[[ylab]].values.flatten()

        Xall = all_dat[xlabs].values
        yall= all_dat[[ylab]].values.flatten()

        # assuming plot with following axes:
        #
        #  x2
        #  ^
        #  |
        #  |
        #  |
        #  .--->x1
        #

        # get limits for x1 axis

        x1_lim = [np.min(Xall[:, 0]), np.max(Xall[:, 0])]
        x1_range = x1_lim[1] - x1_lim[0]
        x1_p5 = 0.1 * x1_range
        x1_lim[0] -= x1_p5
        x1_lim[1] += x1_p5

        # get limits for x2 axis

        x2_lim = [np.min(Xall[:, 1]), np.max(Xall[:, 1])]
        x2_range = x2_lim[1] - x2_lim[0]
        x2_p5 = 0.05 * x2_range
        x2_lim[0] -= x2_p5
        x2_lim[1] += x2_p5

        dec_dict_lab = {'X': Xl,
                        'y': yl,
                        'title': '{0} decision boundary: labelled'.format(model_name),
                        'out_fn': dspec.save_folder + '/saved_models/' + model_name + '_dn_{0}_si_{1}_lab_db_{2}.png'.format(
                            dspec.d_n, dsc.s_i,xv),
                        'lab_x1': xlabs[0],
                        'lab_x2': xlabs[1],
                        'lim_x1': x1_lim,
                        'lim_x2': x2_lim,
                        'clf': current_model}

        dec_dict_ulab = {'X': Xul,
                         'y': yul,
                         'title': '{0} decision boundary: unlabelled'.format(model_name),
                         'out_fn': dspec.save_folder + '/saved_models/' + model_name + '_dn_{0}_si_{1}_ulab_db_{2}.png'.format(
                             dspec.d_n, dsc.s_i,xv),
                         'lab_x1': xlabs[0],
                         'lab_x2': xlabs[1],
                         'lim_x1': x1_lim,
                         'lim_x2': x2_lim,
                         'clf': current_model}

        dec_dict_all = {'X': Xall,
                        'y': yall,
                        'title': '{0} decision boundary: labelled AND unlabelled'.format(model_name),
                        'out_fn': dspec.save_folder + '/saved_models/' + model_name + '_dn_{0}_si_{1}_lab_and_ulab_db_{2}.png'.format(
                            dspec.d_n, dsc.s_i,xv),
                        'lab_x1': xlabs[0],
                        'lab_x2': xlabs[1],
                        'lim_x1': x1_lim,
                        'lim_x2': x2_lim,
                        'clf': current_model}

        dec_dicts = [dec_dict_lab,
                     dec_dict_ulab,
                     dec_dict_all]
        all_db_dicts[xv]=dec_dicts

        list_all_db_dicts.append(dec_dict_lab)
        list_all_db_dicts.append(dec_dict_ulab)
        list_all_db_dicts.append(dec_dict_all)


    return (list_all_db_dicts)

# plot the classifier regions and save to some file
# using specifications in dicts from get_dec_dicts function
def plot_classifier_regions(**kwargs):

    plt.rcParams["figure.facecolor"] = "w"  # setting defualts for backgorund white
    plt.rcParams["figure.figsize"] = (12, 12)  # setting default fig size
    # Plotting decision regions
    plot_decision_regions(kwargs['X'],
                          kwargs['y'],
                          clf=kwargs['clf'],
                          legend=1,
                          colors='#ff0002,#0000ff',  # colours red and blue
                          zoom_factor=0.1)
    # set axis labels
    plt.xlabel(kwargs['lab_x1'])
    plt.ylabel(kwargs['lab_x2'])

    # set title
    plt.title(kwargs['title'])

    plt.xlim(kwargs['lim_x1'])
    plt.ylim(kwargs['lim_x2'])

    # save to relevant directory
    plt.savefig(kwargs['out_fn'])

    print('saved fig for: {0}'.format(kwargs['title']))
    plt.clf()


# plotting decision boundary using ML method, this method deprecated / worse than plotly

def plot_decision_boundaries_ml(dspec,si_iter,args,optimal_model):
    dsc_loader = eval(dspec.dataloader_function)  # within the spec
    dsc = dsc_loader(args, si_iter, dspec)
    dsc = manipulate_dsc(dsc, dspec)
    dsc.s_i = si_iter
    dec_dicts = get_dec_dicts(optimal_model.model_name, optimal_model, dspec,dsc)
    for d in dec_dicts:
        plot_classifier_regions(**d)
    print('decision boundaries plotted')


# USING PLOTLY LIBRARIES FOR DECISION BOUNDARIES 28_03_2022

# create background for plotly decision boundary
def create_dbound_bg(dsc, optimal_model,hard=False, X1='X1_0', X2='X1_1'):
    '''
    returns a decision boundary plot that can be used as background
    for the scatter plot of label / unlabel data
    '''
    xt = 1  # need extra for end of array
    xlims = [dsc.merge_dat[X1].min()-xt, dsc.merge_dat[X1].max()+xt]
    xl = np.array(xlims)
    xl = xl.round(decimals=2)
    xl[0] *= 100
    xl[1] *= 100
    xl = xl.astype(int) / 100

    yt = 1  # need extra for end of array
    ylims = [dsc.merge_dat[X2].min() - yt, dsc.merge_dat[X2].max() + yt]
    yl = np.array(ylims)
    yl = yl.round(decimals=2)
    yl[0] *= 100
    yl[1] *= 100
    yl = yl.astype(int) / 100

    # define meshgrid for which we will make prediction
    # each point in grid correspond to individual pixel
    xpoints = np.linspace(xl[0], xl[1], num=(xl[1] * 100 - xl[0] * 100).astype(int))
    ypoints = np.linspace(yl[0], yl[1], num=(yl[1] * 100 - yl[0] * 100).astype(int))

    point_combinations = np.transpose([np.tile(xpoints, len(ypoints)), np.repeat(ypoints, len(xpoints))])
    point_combinations = torch.tensor(np.array(point_combinations)).float()

    xadd = [xl[0]] * point_combinations.shape[0]
    yadd = [yl[0]] * point_combinations.shape[0]

    min_add = np.array([xadd, yadd]).transpose()

    # getting pixel combinations for spatial location of points
    xlen = len(xpoints)
    #x_pixels = [x for x in range(xlen)]

    ylen = len(ypoints)
    #y_pixels = [y for y in range(ylen)]

    # pixel_locations=[(x,y) for x,y in all_points]

    # optimal_model=fsup_classifier
    optimal_model = optimal_model.cpu()  # move onto cpu
    # predictions

    with torch.no_grad():
        optimal_model.eval()
        pc = torch.tensor(point_combinations).float().cpu()
        class_predictions = optimal_model.forward(pc)
        class_softmax = get_softmax(class_predictions)
        class_p = class_softmax[:, 1]
        class_p[np.isnan(class_p)]=0.5

        if hard==True: #hard decision boundary...
            class_p[class_p<0.5]=0
            class_p[class_p>0.5]=1

        cp_img = (class_p.reshape((len(xpoints), len(ypoints))) * 1000).cpu().detach().numpy().astype(int) - 1
        cp_img[cp_img == -1] = 0  # need to set this one cos max colours = 1000 and we have to subtract 1 to ensure we don't array index 1000 as 999 is max index of array of size 1000
        cp_i = cp_img.flatten()

    # converting to colour
    cs = Color('red').steps(['white', 'blue'], steps=1000)
    # cs=Color('red').steps([Piecewise('white', 0.5), 'blue'], steps=1002)
    rgb_plotc = [c.to_string() for c in cs]
    # split up all the rgb() srings into tuples
    split_c = [r.split('rgb(')[1].split(')')[0].split(' ') for r in rgb_plotc]
    sc = [rgb_to_int(s) for s in split_c]  # sc rgb vals as int

    # graph colours
    gcols = [sc[x] for x in cp_i]
    gcols = np.array(gcols)
    gcols = gcols.reshape((ylen, xlen, 4))

    # convert gc from a collection of integers to image using PIL library
    gc = gcols.astype(np.uint8)
    out_im = Image.fromarray(gc)
    # transpose cos still some error in implementation, FIX !!!

    dbound_img = out_im.transpose(Image.FLIP_TOP_BOTTOM)
    # ok now 'dbound_img' is a heatmap of the decision boundary.
    # use 'dbound_img' as background for plotly plot
    return (dbound_img)

# plot the decision boundary with unlabelled/labelled points
def plot_dboundary(dsc, dbound_img, model_name,  d_n, s_i, X1='X1_0', X2='X1_1', y='Y', size_ratio=10):
    # Build figure
    fig = go.Figure()

    # unlabelled,y=0
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=dsc.merge_dat[(dsc.merge_dat.type == 'unlabelled') & (dsc.merge_dat[y] == 0)][[X1]].values.flatten(),
            y=dsc.merge_dat[(dsc.merge_dat.type == 'unlabelled') & (dsc.merge_dat[y] == 0)][[X2]].values.flatten(),
            marker=dict(
                color='rgba(214, 39, 40,0.1)',
                size=size_ratio / 4,
                line=dict(
                    color='rgba(214, 39, 40,0.8)',
                    width=1
                ),
                symbol='octagon-dot'
            ),
            name='Unlabelled partition: y=0'
        )
    )
    # unlabelled,y=1
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=dsc.merge_dat[(dsc.merge_dat.type == 'unlabelled') & (dsc.merge_dat[y] == 1)][[X1]].values.flatten(),
            y=dsc.merge_dat[(dsc.merge_dat.type == 'unlabelled') & (dsc.merge_dat[y] == 1)][[X2]].values.flatten(),
            marker=dict(
                color='rgba(40, 53, 199,0.1)',
                size=size_ratio / 4,
                line=dict(
                    color='rgba(40, 53, 199,0.8)',
                    width=1
                ),
                symbol='octagon-dot'
            ),
            name='Unlabelled partition: y=1'
        )
    )

    # labelled,y=0
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=dsc.merge_dat[(dsc.merge_dat.type == 'labelled') & (dsc.merge_dat[y] == 0)][[X1]].values.flatten(),
            y=dsc.merge_dat[(dsc.merge_dat.type == 'labelled') & (dsc.merge_dat[y] == 0)][[X2]].values.flatten(),
            marker=dict(
                color='rgba(214, 39, 40,0.95)',
                size=size_ratio,
                line=dict(
                    color='rgba(9, 9, 9,0.95)',
                    width=1
                ),
                symbol='triangle-up'
            ),
            name='Labelled partition: y=0'
        )
    )

    # labelled,y=1
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=dsc.merge_dat[(dsc.merge_dat.type == 'labelled') & (dsc.merge_dat[y] == 1)][[X1]].values.flatten(),
            y=dsc.merge_dat[(dsc.merge_dat.type == 'labelled') & (dsc.merge_dat[y] == 1)][[X2]].values.flatten(),
            marker=dict(
                color='rgba(40, 53, 199,0.95)',
                size=size_ratio,
                line=dict(
                    color='rgba(9, 9, 71,0.95)',
                    width=1

                ),
                symbol='triangle-up'
            ),
            name='Labelled partition: y=1'
        )
    )

    # validation,y=0
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=dsc.merge_dat[(dsc.merge_dat.type == 'validation') & (dsc.merge_dat[y] == 0)][[X1]].values.flatten(),
            y=dsc.merge_dat[(dsc.merge_dat.type == 'validation') & (dsc.merge_dat[y] == 0)][[X2]].values.flatten(),
            marker=dict(
                color='rgba(255, 220, 255,1)',
                size=size_ratio / 1,
                line=dict(
                    color='rgba(255, 220, 255,1)',
                    width=1
                ),
                symbol='triangle-down'
            ),
            name='Validation partition: y=0'
        )
    )
    # validation,y=1
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=dsc.merge_dat[(dsc.merge_dat.type == 'validation') & (dsc.merge_dat[y] == 1)][[X1]].values.flatten(),
            y=dsc.merge_dat[(dsc.merge_dat.type == 'validation') & (dsc.merge_dat[y] == 1)][[X2]].values.flatten(),
            marker=dict(
                color='rgba(0, 255, 255,1)',
                size=size_ratio / 1,
                line=dict(
                    color='rgba(0, 255, 255,1)',
                    width=1
                ),
                symbol='triangle-down'
            ),
            name='Validation partition: y=1'
        )
    )

    xt = 1
    xlims = [dsc.merge_dat[X1].min() - xt, dsc.merge_dat[X1].max() + xt]
    yt = 1
    ylims = [dsc.merge_dat[X2].min() - xt, dsc.merge_dat[X2].max() + xt]

    fig.update_xaxes(range=xlims)
    fig.update_yaxes(range=ylims)

    # Add image
    fig.add_layout_image(
        dict(
            source=dbound_img,
            xref="x",
            yref="y",
            x=xlims[0],
            y=ylims[0],
            sizex=xlims[1] - xlims[0],
            sizey=ylims[1] - ylims[0],
            opacity=0.5,
            layer="below",
            xanchor="left",
            yanchor="bottom",
            sizing="stretch")
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.2)', zeroline=False)

    fig.update_xaxes(
        title_text=X1,
        title_standoff=25)

    fig.update_yaxes(
        title_text=X2,
        title_standoff=25)

    title_str = '{0}:    d_n={1},    s_i={2}'.format(model_name, d_n, s_i)

    fig.update_layout(
        title={
            'text': title_str,
            'y': 0.9,
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top'})

    return (fig)

#combines the above two methods to produce output
def plot_decision_boundaries_plotly(dspec,si_iter,args,optimal_model,hard,output_html=True):

    dsc_loader = eval(dspec.dataloader_function)  # within the spec
    dsc = dsc_loader(args, si_iter, dspec)
    dsc = manipulate_dsc(dsc, dspec)
    dsc.s_i = si_iter

    if len(dsc.feature_names)>1:
        print('decision boundaries not plotted since > 1 feature variable')
        return()
    else:
        for n in dsc.feature_names:
            current_xy=dsc.label_names_alphan[n]
            # creating / saving plotly plot of decision boundaries
            dbound_img = create_dbound_bg(dsc, optimal_model,hard,X1=current_xy[0],X2=current_xy[1])  # create background for our decision boundary
            fout = plot_dboundary(dsc, dbound_img, optimal_model.model_name, d_n=dspec.d_n, s_i=dsc.s_i,
                                  size_ratio=20)  # get the plotly fig plot object
            # and saving for X1, write the img
            assert (len(dsc.feature_names) == 1)

            #create string to indicate hard / false decision boundary

            if hard==True:
                hard_str='hard'
            elif hard==False:
                hard_str='soft'
            else:
                print('error hard must be set EITHER true or false')
                assert(1==0)

            out_fn_png = dspec.save_folder + '/saved_models/' + optimal_model.model_name + '_dn_{0}_si_{1}_{3}_db_{2}.png'.format(
                dspec.d_n,
                dsc.s_i,
                n,
                hard_str)

            out_fn_html=dspec.save_folder + '/saved_models/' + optimal_model.model_name + '_dn_{0}_si_{1}_{3}_db_{2}.html'.format(
                dspec.d_n,
                dsc.s_i,
                n,
                hard_str
                )


            fout.write_image(out_fn_png, width=1100, height=800, scale=2)
            print(f'png successfully written to: {out_fn_png}')
            if output_html:
                fout.write_html(out_fn_html)
                print(f'html successfully written to: {out_fn_html}')

# convert rgb colour value to int
def rgb_to_int(in_list):
    retval=[int(float(s)) for s in in_list]
    retval+=[255]
    return(tuple(retval))

# dspec sourcing
def get_dspec(d_n):
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)

    dspec = master_spec['dataset_spec']
    dspec.set_index("d_n", inplace=True)  # set this index for easier
    # store index of pandas loc where we find the value
    dspec = dspec.loc[d_n]  # use this as reerence..
    dspec.d_n = str(d_n) if dspec.d_n_type == 'str' else int(d_n)
    return(dspec)



from typing import IO, Any, Dict, Iterable, Optional, Union, cast


# combine synthetic data w original labelled data
class CGANSupervisedDataModule(pl.LightningDataModule):
    def __init__(self, orig_data, synth_dd,inclusions, batch_size: int = 64,n_to_sample_for_orig: str='unlabelled'):
        super().__init__()
        self.orig_data = orig_data
        self.batch_size = batch_size
        self.synth_dd=synth_dd
        self.inclusions=inclusions
        self.n_to_sample_for_orig=n_to_sample_for_orig#'labelled' #'unlabelled'

    def setup(self, stage: Optional[str] = None):
        orig_data = self.orig_data
        synth_dd=self.synth_dd

        # ----------#
        # Training Labelled
        # ----------#
        X_train_lab = orig_data['label_features']
        y_train_lab = orig_data['label_y'].long()

        # ----------#
        # Training Unlabelled
        # ----------#
        X_train_ulab = orig_data['unlabel_features']
        y_train_ulab = orig_data['unlabel_y'].long()

        # -------------#
        # Validation
        # -------------#

        X_val = orig_data['val_features']
        y_val = orig_data['val_y'].long()

        # ----------#
        # Synthetic Data
        # ----------#
        X_train_synthetic = synth_dd['synthetic_features']
        y_train_synthetic = synth_dd['synthetic_y'].long().reshape(-1,1)



        #print(self.inclusions)
        if self.inclusions == 'orig_and_synthetic':
            X_train_total = torch.cat((X_train_lab, X_train_synthetic), 0)
            y_train_total = torch.cat((y_train_lab.flatten(), y_train_synthetic.flatten()), 0).reshape((-1,1))


        elif self.inclusions=='orig_only':

            # -------------#
            # Setting up resampling
            # -------------#

            n_unlabelled = X_train_ulab.shape[0]
            n_labelled = X_train_lab.shape[0]
            dummy_label_weights = torch.ones(n_labelled)
            
            if self.n_to_sample_for_orig=='labelled':
                num_samples=n_labelled
            
            elif self.n_to_sample_for_orig=='unlabelled':
                num_samples=n_unlabelled
                
            elif self.n_to_sample_for_orig=='baseline':
                num_samples=2000 #keep in line with original datsets
                
            
            resampled_i = torch.multinomial(dummy_label_weights, num_samples=num_samples, replacement=True)
            X_train_lab_rs = X_train_lab[resampled_i]
            y_train_lab_rs = y_train_lab[resampled_i]


            X_train_total = X_train_lab_rs
            y_train_total = y_train_lab_rs.flatten().reshape((-1,1))


        elif self.inclusions=='synthetic_only':
            X_train_total = X_train_synthetic
            y_train_total = y_train_synthetic.reshape((-1,1))
        else:
            assert(1==0)


        self.data_train = torch.utils.data.TensorDataset(X_train_total,
                                                         y_train_total)

        # Validation Sets
        vfeat = X_val.unsqueeze(0)
        vlab = y_val.unsqueeze(0)
        self.data_validation = torch.utils.data.TensorDataset(vfeat, vlab)
        self.nval = vlab.shape[0]

        return (self)

    # def train_dataloader(self):
    #     return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.data_validation, batch_size=self.nval)
    
    def train_dataloader(self):
        has_gpu=torch.cuda.is_available()
        if has_gpu:
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True,num_workers=4)
        else:
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        has_gpu=torch.cuda.is_available()
        if has_gpu:
            return DataLoader(self.data_validation, batch_size=self.nval, pin_memory=True,num_workers=4)
        else:
            return DataLoader(self.data_validation, batch_size=self.nval)



