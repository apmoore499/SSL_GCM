
#--d_n=synthdag_dset_3 --s_i=1 --n_iterations=100 --lr=1e-3 --use_single_si=False --ulab_bsize=128 --patience=2 --min_temp=0.1 --init_temp=1.0 --n_trials=1 --val_loss_criterion=labelled_bce_and_all_feat_mmd --lab_bsize=4 --balance=False

######
#
# NEW PROCEDURE FOR TRAINING CGAN WITH GUMBEL SOFTMAX FOR SIMULTANEOUS LEARNING OF ENTIRE CAUSAL GRAPH
#
######
import os

import torch.nn
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

if __name__ == '__main__':

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
    parser.add_argument('--val_loss_criterion',help='labelled or unlabelled loss for val',type=str)
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
    DELAYED_MIN_START=100


    algo_variant='gumbel'

    #now we want to read in dataset_si
    csi=master_spec['dataset_si'][dspec.d_n].values
    candidate_si=csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si==True: #so we want to use single si, not entire range
        #args.optimal_si_list=[args.optimal_si_list[args.s_i]]
        args.optimal_si_list = [args.s_i]
    #now we are onto training
    print('hello there')
    for k, s_i in enumerate(args.optimal_si_list):
        print('doing s_i: {0}'.format(s_i))
        args.st = time.time()

        dsc_loader=eval(dspec.dataloader_function) #within the spec
        dsc=dsc_loader(args,s_i,dspec)
        dsc=manipulate_dsc(dsc,dspec) #adding extra label column and convenient features for complex data mod later on

        order_to_train=dsc.dag.topological_sorting()
        dsc_generators=OrderedDict() #ordered dict of the generators
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

        print('pausing here')

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





        dsc_generators['ordered_v']={} #to be used later on for retrieving variable names, in correct order
        dsc_generators['ordered_v_alphan']={}

        #we train as follows:

        #1. cause + spouse together
        #2. label
        #3. effect variables

        #we need to get the relvant idx for input of variables downstream

        #train the cause_spouse_v_idx ones


        # if 'X' in dsc.labels[v_i]:
        cur_x_lab = reduce_list([dsc.label_names[v] for v in cause_spouse_v_idx])
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
                             input_dim=dsc.feature_dim*len(cause_spouse_v_idx),
                             median_pwd=median_pwd,
                             num_hidden_layer=args.nhidden_layer,
                             middle_layer_size=args.n_neuron_hidden,
                             label_batch_size=args.lab_bsize,
                             unlabel_batch_size=args.tot_bsize)

        tloader = SSLDataModule_Unlabel_X(dsc.merge_dat, target_x=cur_x_lab, batch_size=args.tot_bsize)

        #concat the variable names

        #vn_concat='_'.join([dsc.labels[v] for v in cause_spouse_v_idx])
        vn_concat='CAUSE_SPOUSE'
        model_name = create_model_name(vn_concat, algo_variant)

        if args.estop_mmd_type == 'val':
            estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
            min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
                                                                   dspec.save_folder)  # returns max checkpoint

        elif args.estop_mmd_type == 'trans':
            estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
            min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,
                                                                     dspec.save_folder)  # returns max checkpoint

        callbacks = [min_mmd_checkpoint_callback, estop_cb]

        tb_logger = create_logger(model_name, d_n, s_i)
        trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)

        delete_old_saved_models(model_name, dspec.save_folder, s_i)

        trainer.fit(gen_x, tloader)  # train here

        mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)
        gen_x = gen_x.load_from_checkpoint(checkpoint_path=mod_names[0])  # loads correct model
        #dsc_generators[dsc.labels[v_i]] = gen_x
        #dsc_generators[dsc.labels[v_i]].conditional_on_label = False
        #dsc_generators[dsc.labels[v_i]].conditional_feature_names = []

        dsc_generators['cause_spouse_generator']=gen_x
        dsc_generators['cause_spouse_generator'].conditional_on_label = False
        dsc_generators['cause_spouse_generator'].conditional_feature_names = []
        dsc_generators['ordered_v']['cause']=[dsc.labels[v] for v in cause_spouse_v_idx]
        dsc_generators['ordered_v_alphan']['cause'] = cur_x_lab


        ###############
        #
        # LABEL GENERATOR
        #
        ###############
        print('#########################')
        print('#                        ')
        print('CREATING LABEL GENERATOR ')
        print('#                        ')
        print('#########################')
        # train label generator
        v_i = label_v_idx
        k_n = 1
        source_edges = dsc.dag.es.select(_target=v_i)
        source_vertices = [s_e.source_vertex for s_e in source_edges]  # source vertex
        sv = list(set([v.index for v in source_vertices]))  # source idx


        vinfo_dict = {}  # reset this dict
        cur_variable = dsc.labels[v_i]
        source_variable = [dsc.labels[s] for s in sv]

        print('training on variable number: {0} of {1}'.format(k_n + 1, len(order_to_train)))
        print('+---------------------------------------+')
        print('cur_var_name : {0}\tsource_var_name(s): {1}'.format(cur_variable, source_variable))

        cond_lab = [dsc.labels[l] for l in sv]  # labels of conditinoal vars
        cond_vtype = [dsc.variable_types[l] for l in sv]  # types of conditional vars
        cond_idx = sv  # index of conditioanl vars in pandas df

        LABEL_NAME=cur_variable

        if len(sv) == 0 and dsc.variable_types[v_i] == 'label':
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

        if len(sv) > 0 and dsc.variable_types[v_i] == 'label':  # need to incorporate some conditional variables

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

            #tloader = SSLDataModule_Y_from_X(dsc.merge_dat,
            #                                 tvar_name=cur_variable,
            #                                 cvar_name=concat_cond_lab,
            #                                 batch_size=args.lab_bsize)

            model_name = create_model_name(dsc.labels[v_i], algo_variant)
            cond_vars = ''.join(cond_lab)
            #min_bce_chkpt_callback = return_chkpt_min_bce(model_name, dspec.save_folder)
            #estop_cb = return_early_stop_cb_bce(patience=args.estop_patience)
            #callbacks = [min_bce_chkpt_callback, estop_cb]
            #tb_logger = create_logger(model_name, d_n, s_i)
            #trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)
            #delete_old_saved_models(model_name, dspec.save_folder, s_i)
           # trainer.fit(y_x1gen, tloader)  # train here
            #mod_names = return_saved_model_name(model_name, dspec.save_folder, d_n, s_i)
            #if len(mod_names) > 1:
            #    print('error duplicate model names')
            #    assert 1 == 0
            #y_x1gen = y_x1gen.load_from_checkpoint(checkpoint_path=mod_names[0])
            dsc_generators[dsc.labels[v_i]] = copy.deepcopy(y_x1gen)
            #dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab
            #dsc_generators[dsc.labels[v_i]].conditional_on_label = False

            dsc_generators['ordered_v']['label'] = {}
            dsc_generators['ordered_v']['label']['inputs'] = cond_lab
            dsc_generators['ordered_v']['label']['inputs_alphan'] = concat_cond_lab
            dsc_generators['ordered_v']['label']['output'] = dsc.labels[v_i]
            # dsc_generators['ordered_v']['label']['inputs_alphan'] = concat_cond_lab


        temperature = args.init_temp  # initial temperature
        order_to_train = dsc.dag.topological_sorting()
        all_dsc_vars = dsc.labels
        # derive the variable type from the labels


        all_dsc_vtypes = dsc.variable_types#['feature' for v in all_dsc_vars]
        # get idx of y
        #for k, vlabel in enumerate(all_dsc_vars):
        #    if 'Y' in vlabel or 'location' in vlabel:
       #         all_dsc_vtypes[k] = 'label'

        ddict_vtype = {}
        for v, vtype in zip(all_dsc_vars, all_dsc_vtypes):
            ddict_vtype[v] = vtype

        #dsc_generators = OrderedDict()  # ordered dict of the generators
        dsc.causes_of_y = []

        # name of the labelled variable
        label_name = dsc.labels[np.where([d == 'label' for d in dsc.variable_types])[0][0]]

        # first sweep through = create generators
        var_conditional_feature_names = {}
        dict_conditional_on_label = {}

        #now concat all of the effect $X$ generators...

        #effect_v_idx

        # if 'X' in dsc.labels[v_i]:
        cur_x_lab = reduce_list([dsc.label_names[v] for v in effect_v_idx])


        #v_i = label_v_idx[0]
        k_n = 2


        source_edges = [dsc.dag.es.select(_target=v) for v in effect_v_idx]
        all_source_vertices = []
        for se in source_edges:
            for edge in se:
                source_vertex_id = edge.source
                target_vertex_id = edge.target
                source_vertex = dsc.dag.vs[source_vertex_id]
                all_source_vertices.append(source_vertex.index)
                # print(source_vertex)


        #source_vertices = [s_e.source_vertex for s_e in source_edges]  # source vertex
        #sv = [v.index for v in source_vertices]  # source idx
        #vinfo_dict = {}  # reset this dict
        #cur_variable = dsc.labels[v_i]
        #source_variable = [dsc.labels[s] for s in sv]
        print('#########################')
        print('#                        ')
        print('CREATING EFFECT GENERATOR ')
        print('#                        ')
        print('#########################')

        sv=list(set(all_source_vertices))
        #print('+---------------------------------------+')
        #print('cur_var_name : {0}\tsource_var_name(s): {1}'.format(cur_variable, source_variable))
        cond_lab = [dsc.labels[l] for l in sv]  # labels of conditinoal vars
        cond_vtype = [dsc.variable_types[l] for l in sv]  # types of conditional vars
        cond_idx = sv  # index of conditioanl vars in pandas df

        #cond_lab = source_variable  # labels of conditinoal vars
        #cond_vtype = [dsc.variable_types[l] for l in all_source_vertices]  # types of conditional vars
        #cond_idx = all_source_vertices  # index of conditioanl vars in pandas df
        #sv = list(set(all_source_vertices))


        if len(sv) == 1 and cond_vtype == ['label']:
            #cur_x_lab = dsc.labels[v_i]
            # get the matching data...
            all_vals = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
            # match column names
            # subset
            x_vals = all_vals[cur_x_lab].values
            # get median pwd
            median_pwd = get_median_pwd(torch.tensor(x_vals))
            n_lab = dsc.merge_dat[dsc.merge_dat.type == 'labelled'].shape[0]
            n_ulab = x_vals.shape[0]

            # make gen
            x2_y_gen = Generator_X2_from_Y(args.lr,
                                           args.d_n,
                                           s_i,
                                           dspec.dn_log,
                                           input_dim=dsc.feature_dim*len(effect_v_idx) + dsc.n_classes,
                                           output_dim=dsc.feature_dim*len(effect_v_idx),
                                           median_pwd=median_pwd,
                                           num_hidden_layer=args.nhidden_layer,
                                           middle_layer_size=args.n_neuron_hidden,
                                           n_lab=n_lab,
                                           n_ulab=n_ulab,
                                           label_batch_size=args.lab_bsize,
                                           unlabel_batch_size=args.tot_bsize)  # this is it
            # get data loader
            #tloader = SSLDataModule_X_from_Y(orig_data_df=dsc.merge_dat,
            #                                 tvar_name=cur_x_lab,
            #                                 cvar_name=cond_lab,
            #                                 cvar_type='label',
            #                                 labelled_batch_size=args.lab_bsize,
            #                                 unlabelled_batch_size=args.tot_bsize,
            #                                 **vinfo_dict)
            #model_name = create_model_name(dsc.labels[v_i], algo_variant)
            #cond_vars = ''.join(cond_lab)

            #if args.estop_mmd_type == 'val':
            #    estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
            #    min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
            #                                                          dspec.save_folder)  # returns max checkpoint

            #elif args.estop_mmd_type == 'trans':
            #    estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
            #    min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,
            #                                                             dspec.save_folder)  # returns max checkpoint

            #callbacks = [min_mmd_checkpoint_callback, estop_cb]
            #tb_logger = create_logger(model_name, d_n, s_i)
            #trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, args.n_iterations)
            #delete_old_saved_models(model_name, dspec.save_folder, s_i)
            #trainer.fit(x2_y_gen, tloader)
            #mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)

            #if len(mod_names) > 1:
            #    print(mod_names)
            #    print('error duplicate model names')
            #    assert 1 == 0
            #elif len(mod_names) == 1:
            #    x2_y_gen = x2_y_gen.load_from_checkpoint(checkpoint_path=mod_names[0])
            #else:
            #    assert 1 == 0

            # training complete, save to list
            dsc_generators['effect_generator'] = copy.deepcopy(x2_y_gen)
            dsc_generators['effect_generator'].conditional_on_label = True
            dsc_generators['effect_generator'].conditional_feature_names = []

            #dsc_generators['effect_generator'] = copy.deepcopy(genx2_yx1)
           # dsc_generators['effect_generator'].conditional_on_label = True
            #dsc_generators['effect_generator'].conditional_feature_names = cond_lab

            dsc_generators['ordered_v']['effect'] = {}
            dsc_generators['ordered_v']['effect']['inputs'] = cond_lab
            dsc_generators['ordered_v']['effect']['input_features_alphan'] = []
            dsc_generators['ordered_v']['effect']['input_label_alphan'] = label_name
            dsc_generators['ordered_v']['effect']['outputs'] = reduce_list([dsc.labels[v] for v in effect_v_idx])
            dsc_generators['ordered_v']['effect']['outputs_alphan'] = reduce_list([dsc.label_names_alphan[v] for v in effect_v_idx])


            # Y  ->  X2
            # X1 ->
        elif len(sv) > 1 and 'label' in cond_vtype:

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
            #c = dsc.labels[v_i]
            #cur_x_lab = dsc.labels[v_i]
            lab_dat = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled'])]
            all_dat = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
            n_lab = lab_dat.shape[0]
            n_ulab = all_dat.shape[0]

            ##############################
            #    median pwd for target x
            ##############################
            # match column names
            #ldc = [c for c in all_dat.columns if cur_x_lab in c]
            target_x_vals = all_dat[cur_x_lab].values
            median_pwd_target = get_median_pwd(torch.tensor(target_x_vals))

            ################################
            #  median pwd for conditional x
            ################################
            # match column names
            cond_x_vals = all_dat[concat_cond_lab].values
            median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))

            input_dim = len(concat_cond_lab) + len(cur_x_lab) + dsc.n_classes
            output_dim = len(cur_x_lab)

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

            #if dsc.feature_dim > 1:
            #    cur_x_lab = dsc.label_names_alphan[cur_x_lab]
            #else:
           #     cur_x_lab = cur_x_lab

            #tloader = SSLDataModule_X2_from_Y_and_X1(
            #    orig_data_df=dsc.merge_dat,
            #    tvar_names=cur_x_lab,  # change from cur_x_lab
            #    cvar_names=concat_cond_lab,
            #    label_var_name=label_name,
            #    labelled_batch_size=args.lab_bsize,
            #    unlabelled_batch_size=args.tot_bsize,
            #    use_bernoulli=args.use_bernoulli,
            #    causes_of_y=None,
            #    **vinfo_dict)

            # train

            #vn_concat = '_'.join([dsc.labels[v] for v in effect_v_idx])

            #model_name = create_model_name(vn_concat, algo_variant)

            #model_name = create_model_name(dsc.labels[v_i], algo_variant)
            #cond_vars = ''.join(cond_lab)

            #if args.estop_mmd_type == 'val':
            #    estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
            #    min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
                                                                       #dspec.save_folder)  # returns max checkpoint

            #elif args.estop_mmd_type == 'trans':
            #    estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
            #    min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,
                                                                         #dspec.save_folder)  # returns max checkpoint

            #callbacks = [min_mmd_checkpoint_callback, estop_cb]
            ##tb_logger = create_logger(model_name, args.d_n, s_i)
            #trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)
            #delete_old_saved_models(model_name, dspec.save_folder, s_i)  # delete old
            #trainer.fit(genx2_yx1, tloader)  # train here

            #mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)
            #if len(mod_names) > 1:
            #    print('error duplicate model names')
            #    assert 1 == 0
            #elif len(mod_names) == 1:
            #    genx2_yx1 = genx2_yx1.load_from_checkpoint(checkpoint_path=mod_names[0])
            #else:
            #    assert 1 == 0

            # dsc_generators[dsc.labels[v_i]]=copy.deepcopy(genx2_yx1)

            dsc_generators['effect_generator'] = copy.deepcopy(genx2_yx1)
            #dsc_generators['effect_generator'].conditional_on_label = True
            #dsc_generators['effect_generator'].conditional_feature_names = cond_lab


            dsc_generators['ordered_v']['effect'] = {}
            dsc_generators['ordered_v']['effect']['inputs'] = cond_lab
            dsc_generators['ordered_v']['effect']['input_features_alphan'] = concat_cond_lab
            dsc_generators['ordered_v']['effect']['input_label_alphan'] = label_name
            dsc_generators['ordered_v']['effect']['outputs'] = conditional_feature_names
            dsc_generators['ordered_v']['effect']['outputs_alphan'] = cur_x_lab



        #print('pausing here')


        for k in ['effect_generator',LABEL_NAME]:
            dsc_generators[k].configure_optimizers()

        print('this is where we do the big pause')


        #ok now we ready to do the gumbel

        ###################
        ###################
        # TRAINING GUMBEL #
        ###################
        ###################

        #the pass thru should be simpler:

        # 1. pass thru for p(Y|X_C)
        # 2. pass thru for p(X_E|Y,X_C,X_S)
        # 3. pass thru for p(X_E|\hat{Y},X_C,X_S) <----- unlabelled pass
        # 4. validation


        # Now we are using gumbel method to train all other generators
        # form dict comprising the idx of conditional features in the dataloader for each variable to be trained
        #all_keys = [k for k in dsc_generators.keys()]
        # make sure that label is removed from conditional_feature_names...
        #for k in all_keys:
        #    if hasattr(dsc_generators[k], 'conditional_feature_names'):
        #        cond_feat_var = dsc_generators[k].conditional_feature_names
        #        if dsc.label_name in cond_feat_var:
        #            new_fn = [f for f in cond_feat_var if f != dsc.label_name]  # new feature names
        #            dsc_generators[k].conditional_feature_names = new_fn  # overwrite conditional_feature_names w label var removed

            # GPU preamble
        has_gpu = torch.cuda.is_available()
        # gpu_kwargs={'gpus':torch.cuda.device_count(),'precision':16} if has_gpu else {}
        gpu_kwargs = {'gpus': torch.cuda.device_count()} if has_gpu else {}
        if has_gpu:
            for k in [LABEL_NAME,'effect_generator']:
                dsc_generators[k].cuda()










        for ttt in range(args.n_trials):  # n_trials = 1 usually. set > 1 to check stability of gumbel estimation
            # create new optimiser object to perform simultaneous update
            # when using gumbel noise

            all_parameters_labelled = []  # parameters of all generators including generator for Y

            for k in [LABEL_NAME] + ['effect_generator']:
                print(k)
                all_parameters_labelled = all_parameters_labelled + list(dsc_generators[k].parameters())

            combined_labelled_optimiser = optim.Adam(all_parameters_labelled, lr=args.lr)  # reset labelled opt

            all_parameters_unlabelled = []

            for k in ['effect_generator']:
                all_parameters_unlabelled = all_parameters_unlabelled + list(dsc_generators[k].parameters())

            combined_unlabelled_optimiser = optim.Adam(all_parameters_unlabelled, lr=args.lr)  # reset unlabelled opt

            # reset label generators
            dsc_generators[LABEL_NAME].apply(init_weights)

            # reset conditional generators
            dsc_generators['effect_generator'].apply(init_weights)

            mintemps_dict = {}
            optimal_mods_dict = OrderedDict()
            temps_tried = []
            val_losses_joint = []
            inv_val_accuracies = []
            val_bces = []

            rt_dict = {}  # for storing loss at each temp
            otemp_dict = {}  # for storing optimal temp value
            tcounter = 1


            #dsc_generators['ordered_v']['label']['inputs'] = cond_lab
            #dsc_generators['ordered_v']['label']['inputs_alphan'] = concat_cond_lab
            #dsc_generators['ordered_v']['label']['output'] = dsc.labels[v_i]

            #create list of alphan features
            all_feature_names_alphan = reduce_list([dsc.label_names_alphan[f] for f in dsc.feature_names])

            causes_of_y_feat_names=dsc_generators['ordered_v']['label']['inputs_alphan']
            # tensor for validating y BCE loss
            y_features = dsc.merge_dat[dsc.merge_dat.type == 'labelled'][causes_of_y_feat_names].values

            y_truelabel = dsc.merge_dat[dsc.merge_dat.type == 'labelled'][[LABEL_NAME]].values.flatten()
            y_truelabel = torch.tensor(y_truelabel)
            y_truelabel = torch.nn.functional.one_hot(y_truelabel).float()


            # check that y_features is not used for BCE validation!!!

            all_labelled_features = dsc.merge_dat[dsc.merge_dat.type == 'labelled'][all_feature_names_alphan]

            all_labelled_label = dsc.merge_dat[dsc.merge_dat.type == 'labelled'][[LABEL_NAME]].values.flatten()
            all_labelled_label = torch.tensor(all_labelled_label)
            all_labelled_label = torch.nn.functional.one_hot(all_labelled_label).float()



            # i think we gotta just use ```all_feature_names_alphan``` here???
            all_unlabelled_and_labelled=dsc.merge_dat[dsc.merge_dat.type.isin(['labelled','unlabelled'])][all_feature_names_alphan]

            # i think we gotta just use ```all_feature_names_alphan``` here???
            all_unlabelled=dsc.merge_dat[dsc.merge_dat.type.isin(['unlabelled'])][all_feature_names_alphan]




            all_validation_features=dsc.merge_dat[dsc.merge_dat.type.isin(['validation'])][all_feature_names_alphan]

            label_name_alphan=LABEL_NAME

            all_validation_label=dsc.merge_dat[dsc.merge_dat.type.isin(['validation'])][label_name_alphan].values.flatten()
            all_validation_label = torch.tensor(all_validation_label)
            all_validation_label = torch.nn.functional.one_hot(all_validation_label).float()
            # set up indexing for cause/spouse and effect features

            dsc_idx_dict = {}
            dsc_idx_dict['cause_spouse'] = np.array(dsc_generators['ordered_v_alphan']['cause'])
            dsc_idx_dict['effect'] = np.array(dsc_generators['ordered_v']['effect']['outputs_alphan'])
            dsc_idx_dict['all_features'] = np.array(all_feature_names_alphan)
            dsc_idx_dict['cs_idx'] = [np.where(c == dsc_idx_dict['all_features'])[0][0] for c in
                                      dsc_idx_dict['cause_spouse']]
            dsc_idx_dict['ef_idx'] = [np.where(c == dsc_idx_dict['all_features'])[0][0] for c in
                                      dsc_idx_dict['effect']]
            dsc_idx_dict['all_features'] = [np.where(c == dsc_idx_dict['all_features'])[0][0] for c in
                                      dsc_idx_dict['all_features']]


            # get median_pwd of cause/spouse
            median_pwd_dict={}
            median_pwd_dict['cause_spouse']=get_median_pwd(torch.tensor(all_unlabelled_and_labelled[dsc_idx_dict['cause_spouse']].values,device=device_string))
            median_pwd_dict['effect'] = get_median_pwd(torch.tensor(all_unlabelled_and_labelled[dsc_idx_dict['effect']].values, device=device_string))
            #median_pwd_dict['all_features'] = get_median_pwd(torch.tensor(all_unlabelled_and_labelled[dsc_idx_dict['all_features']].values, device=device_string))

            sigma_dict={}
            sigma_dict['cause_spouse']=[median_pwd_dict['cause_spouse'] * k for k in [1/4,1/2,1,2,4]]
            sigma_dict['effect'] = [median_pwd_dict['effect'] * k for k in [1 / 4, 1 / 2, 1, 2, 4]]
            #sigma_dict['all_features'] = [median_pwd_dict['all_features'] * k for k in [1 / 4, 1 / 2, 1, 2, 4]]
            #median_pwd_dict['all_features']
            #median_pwdall_unlabelled_and_labelled[dsc_idx_dict['cause_spouse']]

            # get median_pwd of effect


            if has_gpu:

                val_dloader_unlabelled = DataLoader(
                    torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled_and_labelled.values).cuda()),
                    batch_size=all_unlabelled_and_labelled.shape[0],
                    shuffle=False)

                val_dloader_labelled = DataLoader(
                    torch.utils.data.TensorDataset(torch.Tensor(all_validation_features.values).cuda(),
                                                   torch.Tensor(all_validation_label).cuda()),
                    batch_size=all_labelled_features.shape[0],
                    shuffle=False)

                y_features = torch.tensor(y_features).float().cuda()  # to predict Y from X and validate BCE loss
                y_truelabel = y_truelabel.cuda()
            else:
                val_dloader_unlabelled = DataLoader(
                    torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled_and_labelled.values)),
                    batch_size=all_unlabelled_and_labelled.shape[0],
                    shuffle=False)

                val_dloader_labelled = DataLoader(
                    torch.utils.data.TensorDataset(torch.Tensor(all_validation_features.values),
                                                   torch.Tensor(all_validation_label)),
                    batch_size=all_validation_label.shape[0],
                    shuffle=False)

                y_features = torch.tensor(y_features).float()  # to predict Y from X and validate BCE loss (NO cuda)
                y_truelabel = y_truelabel
            early_end = False
            epoch = 0
            templist = np.linspace(0.99, 0.1, 160)
            # templist = np.linspace(0.99, 0.9, 10)
            converged = False

            from torch.utils.tensorboard import SummaryWriter

            # default `log_dir` is "runs" - we'll be more specific here
            writer = SummaryWriter('lightning_logs/gumbel_training_mmd')
            feature_idx_subdict = {c: idx for idx, c in enumerate(all_unlabelled_and_labelled.columns) if
                                   all_unlabelled_and_labelled.columns[idx] == c}

            #causes_of_y = var_conditional_feature_names[labelled_key]

            #i = [dsc.label_names_alphan[k] for k in causes_of_y]

            # store causes of y column names here:
            causes_of_y_feat_names = dsc_generators['ordered_v']['label']['inputs_alphan']

            all_x_cols = [c for c in all_unlabelled_and_labelled.columns]

            # convert column names to index
            causes_of_y_idx_dl = []
            #dsc_generators['ordered_v']['label'] = {}
            #dsc_generators['ordered_v']['label']['inputs'] = cond_lab

            #dsc_generators['ordered_v']['label']['output'] = dsc.labels[v_i]
            unlabelled_val_losses=[]
            for k, cf in enumerate(all_x_cols):
                if cf in causes_of_y_feat_names:
                    causes_of_y_idx_dl.append(k)
            #templist=[0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90,
            #          0.89,0.88,0.87,0.86,0.85,0.84,0.83,0.82,0.81,0.80,
            #          0.79,0.77,0.77,0.76,0.75,0.74,0.73,0.72,0.71,0.70]
            for t_iter, temp in enumerate(templist):
                if converged == False:

                    # ...log the running loss
                    writer.add_scalar('temp',
                                      temp, t_iter)

                    if has_gpu:

                        ulab_dloader = DataLoader(
                            torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled.values).cuda()),
                            batch_size=args.tot_bsize,
                            shuffle=True)

                        lab_dloader = DataLoader(
                            torch.utils.data.TensorDataset(torch.Tensor(all_labelled_features.values).cuda(),
                                                           torch.Tensor(all_labelled_label).cuda()),
                            batch_size=args.lab_bsize,
                            shuffle=True)

                    else:

                        ulab_dloader = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(all_unlabelled.values)),
                                                  batch_size=args.tot_bsize,
                                                  shuffle=True)

                        lab_dloader = DataLoader(
                            torch.utils.data.TensorDataset(torch.Tensor(all_labelled_features.values),
                                                           torch.Tensor(all_labelled_label)),
                            batch_size=args.lab_bsize,
                            shuffle=True)

                    temps_tried.append(temp)
                    current_optimal_mods_dict = {}
                    n_labelled = all_labelled_features.shape[0]
                    n_unlabelled = all_unlabelled.shape[0]

                    ratio_n_unlabelled = n_labelled / n_unlabelled  # ratio of labelled to unlabelled < 1
                    ratio_n_labelled = int(n_unlabelled / n_labelled)  # ratio of unlabelled to labelled > 1

                    # train mode
                    dsc_generators[LABEL_NAME].train()
                    dsc_generators['effect_generator'].train()

                    for batch_idx, d in enumerate(lab_dloader):

                        cur_batch_features, cur_batch_label = d

                        # PREDICT Y FROM CAUSE/SPOUSE IN LABELLED SAMPLE

                        # PREDICT ALL EFFECTS FROM Y AND THE CAUSE/SPOUSE IN LABELLED SAMPLE

                        ground_truth_features=copy.deepcopy(cur_batch_features)
                        ground_truth_cs=ground_truth_features[:,dsc_idx_dict['cs_idx']]
                        ground_truth_ef = ground_truth_features[:, dsc_idx_dict['ef_idx']]
                        ground_truth_lab=cur_batch_label

                        #ground truth cause / spouse




                        #ground truth effect

                        #all_feature_names_alphan

                        #dsc_generators['ordered_v']['effect']['outputs_alphan']




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
                        y_generated = dsc_generators[LABEL_NAME].forward(input_for_y)
                        # hard=False
                        y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=True,
                                                                              tau=temp)  # I think use softmax label here?
                        # now put into ancestor dictionary
                        current_ancestor_dict[labelled_key] = y_gumbel_softmax
                        #ok get the feature names
                        effect_feature_inputs=dsc_generators['ordered_v']['effect']['input_features_alphan']
                        #get the label variable name
                        effect_label_input=dsc_generators['ordered_v']['effect']['input_label_alphan']
                        effect_combined_inputs = effect_feature_inputs + effect_label_input


                        try:
                            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in effect_combined_inputs]
                            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                        except:
                            #print('hit snag: make sure variable name alphan is fixed')
                            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in dsc_generators['ordered_v']['effect']['inputs']]
                            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)

                        generator_input = torch.cat(cur_feature_inputs_lc, 1)
                        # then add some noise
                        if has_gpu:
                            current_noise = torch.randn((generator_input.shape[0], len(dsc_generators['ordered_v']['effect']['outputs_alphan'])),device='cuda:0')
                        else:
                            current_noise = torch.randn((generator_input.shape[0], len(dsc_generators['ordered_v']['effect']['outputs_alphan'])))
                        # then concatenate the noise
                        generator_input_w_noise = torch.cat((current_noise, generator_input), 1)

                       
                        # then predict
                        predicted_mod_ef = dsc_generators['effect_generator'].forward(generator_input_w_noise)
                        # and put this one into ancestor dict
                        predicted_mod_lab=current_ancestor_dict[labelled_key]
                        ground_truth_cs = ground_truth_features[:, dsc_idx_dict['cs_idx']]
                        ground_truth_ef = ground_truth_features[:, dsc_idx_dict['ef_idx']]
                        ground_truth_lab = cur_batch_label
                        #get mmd loss on this one
                        joint_labelled_mmd_loss = mix_rbf_mmd2_joint_regress(ground_truth_ef,
                                                   predicted_mod_ef,
                                                   ground_truth_features,
                                                   ground_truth_features,
                                                   ground_truth_lab,
                                                   predicted_mod_lab,
                                                   sigma_list=sigma_dict['effect'],
                                                   sigma_list1=sigma_dict['cause_spouse'])

                        # add in EFFECT MMD only

                        effect_labelled_mmd_loss = mix_rbf_mmd2(ground_truth_ef,predicted_mod_ef,sigma_list=sigma_dict['effect'])



                        # add in EFFECT | Y MMD

                        y_given_cause_mmd_loss=mix_rbf_mmd2_joint(ground_truth_features,ground_truth_features,ground_truth_lab,predicted_mod_lab,sigma_list=sigma_dict['cause_spouse'])





                        #ce_loss = torch.nn.CrossEntropyLoss()

                        #bce_lab = ce_loss(ground_truth_lab, predicted_mod_lab.argmax(1).long())

                        #labelled_loss=(bce_lab+joint_labelled_mmd_loss)
                        labelled_loss=torch.sum(torch.stack([joint_labelled_mmd_loss, effect_labelled_mmd_loss,y_given_cause_mmd_loss]), dim=0)/3
                        #labelled_loss= torch.add(,,)/3
                        #torch.nn.CrossEntropyLoss(ground_truth_lab.long(),predicted_mod_lab.long())
                        labelled_loss*=predicted_mod_ef.shape[0]/n_labelled
                        combined_labelled_optimiser.zero_grad()
                        labelled_loss.backward()
                        combined_labelled_optimiser.step()
                        # ...log the running loss
                        writer.add_scalar('joint_labelled_mmd_loss_train',
                                          labelled_loss, t_iter)

                    # mmd loss loss on features x in training set
                    for batch_idx, d in enumerate(ulab_dloader):

                        #print('pausing here')
                        # code go here...
                        cur_batch = d[0]

                        ground_truth_features=copy.deepcopy(cur_batch)

                        # instantiate our ancestor dict that will be used for current batch
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

                        y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=True, tau=temp)
                        # anneal our way out of the temperature
                        # now put into ancestor dictionary
                        current_ancestor_dict[labelled_key] = y_gumbel_softmax

                        # ok get the feature names
                        effect_feature_inputs = dsc_generators['ordered_v']['effect']['input_features_alphan']
                        # get the label variable name
                        effect_label_input = dsc_generators['ordered_v']['effect']['input_label_alphan']
                        effect_combined_inputs = effect_feature_inputs + effect_label_input

                        try:
                            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in effect_combined_inputs]
                            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                        except:
                            #print('hit snag: make sure variable name alphan is fixed')
                            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in dsc_generators['ordered_v']['effect']['inputs']]
                            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)

                        generator_input = torch.cat(cur_feature_inputs_lc, 1)
                        # then add some noise
                        if has_gpu:
                            current_noise = torch.randn((generator_input.shape[0],
                                                         len(dsc_generators['ordered_v']['effect']['outputs_alphan'])), device='cuda:0')
                        else:
                            current_noise = torch.randn((generator_input.shape[0],
                                                         len(dsc_generators['ordered_v']['effect']['outputs_alphan'])))
                        # then concatenate the noise
                        generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
                        # then predict
                        predicted_mod_ef = dsc_generators['effect_generator'].forward(generator_input_w_noise)
                        # and put this one into ancestor dict
                        #predicted_mod_lab = current_ancestor_dict[labelled_key]
                        ground_truth_cs = ground_truth_features[:, dsc_idx_dict['cs_idx']]
                        ground_truth_ef = ground_truth_features[:, dsc_idx_dict['ef_idx']]
                        #ground_truth_lab = cur_batch_label
                        # get mmd loss on this one

                        joint_unlabelled_mmd_loss = mix_rbf_mmd2_joint_regress(ground_truth_ef,
                                                                               predicted_mod_ef,
                                                                               ground_truth_cs,
                                                                               ground_truth_cs,
                                                                               sigma_list=sigma_dict['effect'],
                                                                               sigma_list1=sigma_dict['cause_spouse'])

                        effect_labelled_mmd_loss = mix_rbf_mmd2(ground_truth_ef, predicted_mod_ef,
                                                                  sigma_list=sigma_dict['effect'])


                        unlabelled_loss=torch.sum(
                            torch.stack([joint_unlabelled_mmd_loss, effect_labelled_mmd_loss]),
                            dim=0) / 2
                        unlabelled_loss*= ground_truth_ef.shape[0] / n_unlabelled
                        combined_unlabelled_optimiser.zero_grad()
                        unlabelled_loss.backward()
                        combined_unlabelled_optimiser.step()
                        # ...log the running loss
                        writer.add_scalar('joint_unlabelled_mmd_loss_train',
                                          unlabelled_loss, t_iter)


                    #put this jones into eval mode!
                    dsc_generators[LABEL_NAME].eval()
                    dsc_generators['effect_generator'].eval()

                    with torch.no_grad():
                        # mmd loss on unlabelled features in training set, not using label
                        # features drawn from labelled + unlabelled data
                        for batch_idx, d in enumerate(val_dloader_unlabelled):
                            #unlabelled_val_losses = []

                            cur_batch = d[0]
                            current_ancestor_dict = {}
                            # variables where Y is not ancestor
                            # append them to current_ancestor_dict
                            cur_batch = d[0]

                            ground_truth_features = copy.deepcopy(cur_batch)

                            # instantiate our ancestor dict that will be used for current batch
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

                            y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=True, tau=temp)
                            # anneal our way out of the temperature
                            # now put into ancestor dictionary
                            current_ancestor_dict[labelled_key] = y_gumbel_softmax





                            # ok get the feature names
                            effect_feature_inputs = dsc_generators['ordered_v']['effect']['input_features_alphan']
                            # get the label variable name
                            effect_label_input = dsc_generators['ordered_v']['effect']['input_label_alphan']
                            effect_combined_inputs = effect_feature_inputs + effect_label_input

                            try:
                                cur_feature_inputs_lc = [current_ancestor_dict[f] for f in effect_combined_inputs]
                                cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                            except:
                                #print('hit snag: make sure variable name alphan is fixed')
                                cur_feature_inputs_lc = [current_ancestor_dict[f] for f in
                                                         dsc_generators['ordered_v']['effect']['inputs']]
                                cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)



                            generator_input = torch.cat(cur_feature_inputs_lc, 1)
                            # then add some noise
                            if has_gpu:
                                current_noise = torch.randn((generator_input.shape[0],
                                                             len(dsc_generators['ordered_v']['effect'][
                                                                     'outputs_alphan'])), device='cuda:0')
                            else:
                                current_noise = torch.randn((generator_input.shape[0],
                                                             len(dsc_generators['ordered_v']['effect'][
                                                                     'outputs_alphan'])))
                            # then concatenate the noise
                            generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
                            # then predict
                            predicted_mod_ef = dsc_generators['effect_generator'].forward(generator_input_w_noise)
                            # and put this one into ancestor dict
                            # predicted_mod_lab = current_ancestor_dict[labelled_key]
                            ground_truth_cs = ground_truth_features[:, dsc_idx_dict['cs_idx']]
                            ground_truth_ef = ground_truth_features[:, dsc_idx_dict['ef_idx']]
                            # ground_truth_lab = cur_batch_label
                            # get mmd loss on this one

                            joint_entire_unlabelled_mmd_loss = mix_rbf_mmd2_joint_regress(ground_truth_ef,
                                                                                   predicted_mod_ef,
                                                                                   ground_truth_cs,
                                                                                   ground_truth_cs,
                                                                                   sigma_list=sigma_dict['effect'],
                                                                                   sigma_list1=sigma_dict[
                                                                                       'cause_spouse'])

                            effect_labelled_mmd_loss = mix_rbf_mmd2(ground_truth_ef, predicted_mod_ef,
                                                                    sigma_list=sigma_dict['effect'])

                            unlabelled_loss_validation = torch.sum(
                                torch.stack([joint_entire_unlabelled_mmd_loss, effect_labelled_mmd_loss]),
                                dim=0) / 2

                            # ...log the running loss
                            writer.add_scalar('unlabelled_loss_validation',
                                              unlabelled_loss_validation, t_iter)

                            unlabelled_val_losses.append(unlabelled_loss_validation)




                        # joint mmd loss on labelled data, including label y
                        for batch_idx, d in enumerate(val_dloader_labelled):
                            cur_batch_features, cur_batch_label = d

                            # PREDICT Y FROM CAUSE/SPOUSE IN LABELLED SAMPLE

                            # PREDICT ALL EFFECTS FROM Y AND THE CAUSE/SPOUSE IN LABELLED SAMPLE

                            ground_truth_features=copy.deepcopy(cur_batch_features)
                            ground_truth_cs=ground_truth_features[:,dsc_idx_dict['cs_idx']]
                            ground_truth_ef = ground_truth_features[:, dsc_idx_dict['ef_idx']]

                            ground_truth_feat= ground_truth_features[:, dsc_idx_dict['ef_idx']]

                            #all_features
                            ground_truth_lab=cur_batch_label

                            #ground truth cause / spouse




                            #ground truth effect

                            #all_feature_names_alphan

                            #dsc_generators['ordered_v']['effect']['outputs_alphan']




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
                            y_generated = dsc_generators[LABEL_NAME].forward(input_for_y)
                            # hard=False
                            y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=True,
                                                                                  tau=temp)  # I think use softmax label here?

                            val_lab_soft_label_estimate = torch.nn.functional.gumbel_softmax(y_generated, hard=False,
                                                                                             tau=temp)



                            # now put into ancestor dictionary
                            current_ancestor_dict[labelled_key] = y_gumbel_softmax
                            #ok get the feature names
                            effect_feature_inputs=dsc_generators['ordered_v']['effect']['input_features_alphan']
                            #get the label variable name
                            effect_label_input=dsc_generators['ordered_v']['effect']['input_label_alphan']
                            effect_combined_inputs = effect_feature_inputs + effect_label_input
                            try:
                                cur_feature_inputs_lc = [current_ancestor_dict[f] for f in effect_combined_inputs]
                                cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                            except:
                                #print('hit snag: make sure variable name alphan is fixed')
                                cur_feature_inputs_lc = [current_ancestor_dict[f] for f in
                                                         dsc_generators['ordered_v']['effect']['inputs']]
                                cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)



                            generator_input = torch.cat(cur_feature_inputs_lc, 1)
                            # then add some noise
                            if has_gpu:
                                current_noise = torch.randn((generator_input.shape[0], len(dsc_generators['ordered_v']['effect']['outputs_alphan'])),device='cuda:0')
                            else:
                                current_noise = torch.randn((generator_input.shape[0], len(dsc_generators['ordered_v']['effect']['outputs_alphan'])))
                            # then concatenate the noise
                            generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
                            # then predict
                            predicted_mod_ef = dsc_generators['effect_generator'].forward(generator_input_w_noise)
                            # and put this one into ancestor dict
                            predicted_mod_lab=current_ancestor_dict[labelled_key]
                            ground_truth_cs = ground_truth_features[:, dsc_idx_dict['cs_idx']]
                            ground_truth_ef = ground_truth_features[:, dsc_idx_dict['ef_idx']]
                            ground_truth_lab = cur_batch_label
                            #get mmd loss on this one
                            joint_labelled_validation_mmd_loss = mix_rbf_mmd2_joint_regress(ground_truth_ef,
                                                       predicted_mod_ef,
                                                       ground_truth_cs,
                                                       ground_truth_cs,
                                                       ground_truth_lab,
                                                       predicted_mod_lab,
                                                       sigma_list=sigma_dict['effect'],
                                                       sigma_list1=sigma_dict['cause_spouse'])

                            effect_labelled_mmd_loss = mix_rbf_mmd2(ground_truth_ef, predicted_mod_ef,
                                                                    sigma_list=sigma_dict['effect'])

                            # add in EFFECT | Y MMD

                            y_given_cause_mmd_loss = mix_rbf_mmd2_joint(ground_truth_features, ground_truth_features,
                                                                        ground_truth_lab, predicted_mod_lab,
                                                                        sigma_list=sigma_dict['cause_spouse'])

                            ce_loss = torch.nn.CrossEntropyLoss()
                            val_bce_loss = ce_loss(val_lab_soft_label_estimate,ground_truth_lab.argmax(1))  # use this to get the val bce

                            # bce_lab = ce_loss(ground_truth_lab, predicted_mod_lab.argmax(1).long())

                            # labelled_loss=(bce_lab+joint_labelled_mmd_loss)
                            labelled_loss_validation = torch.sum(torch.stack(
                                [joint_labelled_validation_mmd_loss, effect_labelled_mmd_loss, y_given_cause_mmd_loss]), dim=0) / 3

                            # ...log the running loss
                            writer.add_scalar('labelled_loss_validation',
                                              labelled_loss_validation, t_iter)

                        #ce_loss=torch.nn.CrossEntropyLoss()

                        #bce_lab = ce_loss(ground_truth_lab, predicted_mod_lab.argmax(1).long())

                        #val_sum_mmd=labelled_loss_validation+unlabelled_loss_validation
                        val_sum_mmd=effect_labelled_mmd_loss + joint_entire_unlabelled_mmd_loss + val_bce_loss #try validating on unlabelled mmd only
                            # ...log the running loss
                        writer.add_scalar('val_sum_mmd',val_sum_mmd, t_iter)

                        print('epoch: {1}\ttemp: {2}\tval loss: {0}'.format(val_sum_mmd, epoch, str(temp)[:7]))
                        val_losses_joint.append(val_sum_mmd.cpu().detach().item())

                        # if loss does not improve after DELAYED_MIN_START epochs, model is optimal
                        epoch += 1

                        val_bces.append(val_bce_loss.cpu().detach().item())
                        #inv_val_accuracies.append(current_inverse_acc)
                        # need to set optimal_label_gen for first epoch
                        #if epoch == 1:
                       #     optimal_label_gen = copy.deepcopy(dsc_generators[dsc.label_var])
                       #     current_min_bce = loss_lab_bce
                        #    mintemp = temp

                        if epoch >= 2:
                           if val_bce_loss < min(val_bces[-epoch:-1]):
                                optimal_label_gen = copy.deepcopy(dsc_generators[dsc.label_var])
                                current_min_bce = val_bce_loss
                                mintemp = temp
                                print('\nmin bce: {0}\n'.format(current_min_bce))

                        #if DELAYED_MIN_START == epoch:
                        #    optimal_mods = copy.deepcopy(dsc_generators)
                        #    mintemp = temp
                        #    current_val_loss = val_losses_joint[-1]
                        #    minloss = current_val_loss
                        #    converged = False
                        current_val_loss = val_losses_joint[-1]
                        if current_val_loss == min(val_losses_joint):
                            optimal_mods = copy.deepcopy(dsc_generators)
                            mintemp = temp
                            minloss = current_val_loss
                            print('\nmin val loss: {0}\n'.format(current_val_loss))
                            # ...log the running loss
                            writer.add_scalar('min_val_loss',
                                              minloss, t_iter)

                        #if DELAYED_MIN_START < epoch:

                        if DELAYED_MIN_START < epoch:
                            PREVIOUS_TRAJECTORY = args.estop_patience  # set trajectory to number of epochs so we just take min val loss of all models
                            # get val_losses...
                            #PREVIOUS_TRAJECTORY = epoch  # try setting to epoch instead here

                            prev_k_vloss = val_losses_joint[-PREVIOUS_TRAJECTORY:]
                            # get most prev val loss
                            current_val_loss = val_losses_joint[-1]

                            if current_val_loss == max(prev_k_vloss):
                                converged = True
                                print('model now converged @ temp: {0}'.format(mintemp))
                                # normally write converged=True here
                            if current_val_loss == min(val_losses_joint):
                                optimal_mods = copy.deepcopy(dsc_generators)
                                mintemp = temp
                                minloss = current_val_loss
                                print('\nmin val loss: {0}\n'.format(current_val_loss))
                                # ...log the running loss
                                writer.add_scalar('min_val_loss',
                                                  minloss, t_iter)

                            # get current prediction for val acc on Y generator...
                        gone_thru_once = True
                    rt_dict[temp] = min(val_losses_joint)

        # setting optimal mods like so
        optimal_mods = copy.deepcopy(dsc_generators)
        # setting our gen for label variable to be whatever was decided according to min inverse acc

        if args.use_optimal_y_gen:
            print('overwriting p(Y|X) for optimal generator...')
            optimal_mods[dsc.label_var] = optimal_label_gen

        print('list of bce')
        print(val_bces)

        print('placing optimal mods into dsc_generators: ')

        for m in optimal_mods.keys():
            dsc_generators[m]=copy.deepcopy(optimal_mods[m])




        print('training complete: now synthesise synthetic data')

        if args.use_benchmark_generators and args.use_bernoulli:
            # we need to replace the generators in dsc_generators,
            for k in unlabelled_keys + [labelled_key]:
                # find model name...
                bmodel=create_model_name(k,'basic')
                model_to_search_for=dspec.save_folder+'/saved_models/'+bmodel+"*-s_i={0}-epoch*".format(s_i)
                candidate_models=glob.glob(model_to_search_for)
                dsc_generators[k]=dsc_generators[k].load_from_checkpoint(checkpoint_path=candidate_models[0])

        # generating synthetic data
        # generate 10000 samples


        n_samples=50000
        #n_samples = dsc.merge_dat.shape[0]
        synthetic_samples_dict=generate_samples_to_dict_tripartite(dsc,has_gpu,dsc_generators,device_string,n_samples,gumbel=True,tau=mintemp)
        joined_synthetic_data=samples_dict_to_df(dsc,synthetic_samples_dict,balance_labels=True,exact=False)

        algo_spec = copy.deepcopy(master_spec['algo_spec'])
        algo_spec.set_index('algo_variant', inplace=True)

        #synthetic_data_dir = algo_spec.loc[algo_variant].synthetic_data_dir

        synthetic_data_dir = algo_spec.loc[algo_variant].synthetic_data_dir
        save_synthetic_data(joined_synthetic_data,
                            d_n,
                            s_i,
                            master_spec,
                            dspec,
                            algo_spec,
                            synthetic_data_dir)

        et=time.time()
        total_time=et-st
        total_time/=60


        print('total time taken for n_iterations: {0}\t {1} minutes'.format(n_iterations,total_time))
        # dict of hyperparameters - parameters to be written as text within synthetic data plot later on
        hpms_dict = {'lr': args.lr,
                     'n_iterations': args.n_iterations,
                     'lab_bs': args.lab_bsize,
                     'ulab_bs': args.tot_bsize,
                     'nhidden_layer': args.nhidden_layer,
                     'neuron_in_hidden': args.n_neuron_hidden,
                     'use_bernoulli': args.use_bernoulli,
                     'time':total_time}
        if dsc.feature_dim==1:
            print('feature dim==1 so we are going to to attempt to plot synthetic v real data')
            #plot_synthetic_and_real_data(hpms_dict,dsc,args,s_i,joined_synthetic_data,synthetic_data_dir,dspec)
        else:

            #scols = [s.replace('_0', '') for s in joined_synthetic_data.columns]
            #joined_synthetic_data.columns = scols
            joined_synthetic_data.rename(columns={'Y_0':'Y'},inplace=True)
            #joined_synthetic_data=joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
            if 'y_given_x_bp' in dsc.merge_dat.columns:
                dsc.merge_dat = dsc.merge_dat[[c for c in joined_synthetic_data.columns]]
            if 'y_given_x_bp' in joined_synthetic_data.columns:
                joined_synthetic_data=joined_synthetic_data[[c for c in dsc.merge_dat.columns]]
            synthetic_and_orig_data = pd.concat([dsc.merge_dat, joined_synthetic_data], axis=0)

            dsc.merge_dat=synthetic_and_orig_data
            dsc.d_n=d_n
            dsc.var_types=dsc.variable_types
            dsc.var_names=dsc.labels
            dsc.feature_varnames=dsc.feature_names
            dsc.s_i=s_i
            #plot_2d_data_w_dag(dsc, s_i,synthetic_data_dir=synthetic_data_dir)

            print('data plotted')
