


############################
#
#     Causal GAN - Base Class
#     Trains a sequence of generators to form hierarchical representation of P(X,Y)
#
############################


import sys
sys.path.append('generative_models')
sys.path.append('py')
sys.path.append('py/generative_models/')

import copy
import argparse
from collections import OrderedDict
from generative_models.Generator_Y_from_X1 import *
from generative_models.Generator_X2_from_Y import *
from generative_models.Generator_X2_from_YX1 import *
from generative_models.Generator_Y import *
from generative_models.Generator_X1 import *
from generative_models.Generator_X_from_X import *
from gen_data_loaders import *
from generative_models.benchmarks_cgan import *
from parse_data import *
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_n', help='dataset number 1-5 OR MOG',type=str)
    parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ',type=int)
    parser.add_argument('--n_iterations', help='how many iterations to train classifier for',type=int,default=100)
    parser.add_argument('--lr',help='learning rate ie 1e-2, 1e-3,...',type=float,default=1e-3)
    parser.add_argument('--use_single_si',help='do we want to train on collection of si or only single instance',type=str,default='True')
    parser.add_argument('--use_bernoulli',help='use bernoulli for y given x',type=str,default='False')
    parser.add_argument('--use_benchmark_generators',help='using benchmark generators or not',type=str,default='False')
    parser.add_argument('--lab_bsize',help='label batch size',type=int,default=4)
    parser.add_argument('--tot_bsize',help='total batch size',type=int,default=128)
    parser.add_argument('--estop_patience',help='patience for training generators, stop trianing if no improve after this # epoch',type=int,default=10)
    parser.add_argument('--nhidden_layer',help='how many hidden layers in implicit model:1,3,5',type=int,default=1)
    parser.add_argument('--n_neuron_hidden',help='how many neurons in hidden layer if nhidden_layer==1',type=int,default=50)
    parser.add_argument('--estop_mmd_type',help='callback for early stopping. either use val mmd or trans mmd, val or trans respectively',default='val')
    parser.add_argument('--use_tuned_hpms',help='use tuned hyperparameters',default='False')



    args = parser.parse_args()
    args.use_single_si=str_to_bool(args.use_single_si)
    args.use_bernoulli=str_to_bool(args.use_bernoulli)
    args.use_benchmark_generators=str_to_bool(args.use_benchmark_generators)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)

    if args.use_tuned_hpms==True:
        print('args flag for use tuned hparams is TRUE, but hpms not available for CGAN method: running algorithm without any tuned hpms')

    st=time.time()
    algo_variant=''

    if args.use_bernoulli==False:
        algo_variant='basic_disjoint'
    elif args.use_bernoulli==True:
        algo_variant='marginal_disjoint'
    else:
        assert(1==0)

    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec=pd.read_excel('combined_spec.xls',sheet_name=None)
    #write dataset spec shorthand
    dspec=master_spec['dataset_spec']
    dspec.set_index("d_n",inplace=True) #set this index for easier
    #store index of pandas loc where we find the value
    dspec=dspec.loc[args.d_n] #use this as reerence..
    dspec.d_n= str(args.d_n) if dspec.d_n_type=='str' else int(args.d_n)
    algo_spec = master_spec['algo_spec']
    algo_spec.set_index('algo_variant', inplace=True)


    # GPU preamble
    has_gpu=torch.cuda.is_available()
    gpu_kwargs={'gpus':torch.cuda.device_count()} if has_gpu else {}
    device_string='cuda' if has_gpu else 'cpu'
    ngpu=torch.cuda.device_count()

    d_n=args.d_n
    n_iterations=args.n_iterations
    dn_log=dspec.dn_log

    #now we want to read in dataset_si
    csi=master_spec['dataset_si'][dspec.d_n].values
    candidate_si=csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si==True: #so we want to use single si, not entire range
        args.optimal_si_list=[args.s_i]
    else:
        args.optimal_si_list=[i for i in range(args.s_i)]

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


        #create generators and train individually, in order
        for k_n,v_i in enumerate(order_to_train):
            source_edges=dsc.dag.es.select(_target=v_i)
            source_vertices=[s_e.source_vertex for s_e in source_edges] #source vertex
            sv=[v.index for v in source_vertices]  #source idx
            vinfo_dict={} #reset this dict
            cur_variable=dsc.labels[v_i]
            source_variable=[dsc.labels[s] for s in sv]
            print('training on variable number: {0} of {1}'.format(k_n+1,len(order_to_train)))
            print('+---------------------------------------+')
            print('cur_var_name : {0}\tsource_var_name(s): {1}'.format(cur_variable,source_variable))

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


                tloader=SSLDataModule_Unlabel_X(dsc.merge_dat,target_x=cur_x_lab,batch_size=args.tot_bsize)
                model_name=create_model_name(dsc.labels[v_i],algo_variant)





                if args.estop_mmd_type == 'val':
                    estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback=return_chkpt_min_val_mmd(model_name, dspec.save_folder) #returns max checkpoint

                elif args.estop_mmd_type == 'trans':
                    estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback=return_chkpt_min_trans_mmd(model_name, dspec.save_folder) #returns max checkpoint


                callbacks=[min_mmd_checkpoint_callback,estop_cb]

                tb_logger = create_logger(model_name,d_n,s_i)
                trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=args.n_iterations)

                delete_old_saved_models(model_name,dspec.save_folder,s_i)

                trainer.fit(gen_x,tloader) #train here

                mod_names=return_saved_model_name(model_name,dspec.save_folder,dspec.dn_log,s_i)
                gen_x=gen_x.load_from_checkpoint(checkpoint_path=mod_names[0]) #loads correct model
                dsc_generators[dsc.labels[v_i]]=gen_x
                dsc_generators[dsc.labels[v_i]].conditional_on_label = False
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = []

            elif len(sv) == 0 and dsc.variable_types[v_i]=='label':
                #make generator for Y
                yv=dsc.merge_dat[dsc.merge_dat.type=='labelled'][dsc.labels[v_i]].mean()
                geny=Generator_Y(args.d_n,s_i,dspec.dn_log,yv)
                dsc_generators[dsc.labels[v_i]]=geny
                dsc_generators[dsc.labels[v_i]].conditional_feature_names=[]
                dsc_generators[dsc.labels[v_i]].conditional_on_label = False

            #########
            # X2->
            # X1->  Y
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


                concat_cond_lab=[]
                for c in cond_lab:
                    concat_cond_lab=concat_cond_lab+dsc.label_names_alphan[c]


                tloader=SSLDataModule_Y_from_X(dsc.merge_dat,
                                               tvar_name=cur_variable,
                                               cvar_name=concat_cond_lab,
                                               batch_size=args.lab_bsize)

                model_name=create_model_name(dsc.labels[v_i],algo_variant)
                cond_vars=''.join(cond_lab)
                min_bce_chkpt_callback=return_chkpt_min_bce(model_name,dspec.save_folder)
                estop_cb=return_early_stop_cb_bce(patience=args.estop_patience)
                callbacks=[min_bce_chkpt_callback,estop_cb]
                tb_logger = create_logger(model_name, d_n, s_i)
                trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)
                delete_old_saved_models(model_name, dspec.save_folder, s_i)
                trainer.fit(y_x1gen,tloader) #train here
                mod_names=return_saved_model_name(model_name,dspec.save_folder,d_n,s_i)
                if len(mod_names)>1:
                    print('error duplicate model names')
                    assert 1==0
                y_x1gen=y_x1gen.load_from_checkpoint(checkpoint_path=mod_names[0])
                dsc_generators[dsc.labels[v_i]]=copy.deepcopy(y_x1gen)
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab
                dsc_generators[dsc.labels[v_i]].conditional_on_label = False

                #modify original merge_dat data frame to get y|x on unlabelled data
                #we can use this later if we want to set use_bernoulli==True
                #for using the marginal distribution
                dsc.merge_dat['y_given_x_bp']=dsc.merge_dat[cur_variable] #cur_variable must be Y in this case
                concat_cond_lab=[]
                for c in cond_lab:
                    concat_cond_lab=concat_cond_lab+dsc.label_names_alphan[c]

                #get unlabelled data
                odf=dsc.merge_dat
                odf_unlabelled=odf[odf.type == 'unlabelled']
                x_unlab=odf_unlabelled[concat_cond_lab]
                # convert to numpy
                x_unlab=x_unlab.values
                # convert to torch tensor
                x_unlab= torch.Tensor(x_unlab)
                # now do prediction
                y_softmax = get_softmax(dsc_generators[dsc.labels[v_i]](x_unlab))
                # 1st col as bernoulli parameter
                y_bern_p=y_softmax[:,1]
                # sample from bernoulli dist
                y_hat_unlabelled=torch.bernoulli((y_bern_p))
                # convert to data frame
                y_df=pd.DataFrame(y_hat_unlabelled.cpu().detach().numpy())
                y_df.columns=['y_given_x_bp']
                y_df['ulab_idx']=odf_unlabelled.index.values
                y_df.set_index(['ulab_idx'],inplace=True)
                dsc.merge_dat['y_given_x_bp'].loc[y_df.index.values] = y_df['y_given_x_bp'].loc[y_df.index]
                dsc.causes_of_y=cond_lab


            #X from Y, ie Y->X
            elif len(sv) == 1 and dsc.variable_types[v_i] == 'feature' and cond_vtype == ['label']:
                cur_x_lab = dsc.labels[v_i]
                # get the matching data...
                all_vals = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
                # match column names
                # subset
                x_vals = all_vals[dsc.label_names[v_i]].values
                # get median pwd
                median_pwd = get_median_pwd(torch.tensor(x_vals))
                n_lab=dsc.merge_dat[dsc.merge_dat.type=='labelled'].shape[0]
                n_ulab = x_vals.shape[0]

                dict_for_mmd={'x1':1,'x3':34,'x5':[1,2,3,4,5]}

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
                model_name=create_model_name(dsc.labels[v_i],algo_variant)
                cond_vars = ''.join(cond_lab)

                if args.estop_mmd_type == 'val':
                    estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
                                                                           dspec.save_folder)  # returns max checkpoint

                elif args.estop_mmd_type == 'trans':
                    estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,
                                                                             dspec.save_folder)  # returns max checkpoint


                callbacks=[min_mmd_checkpoint_callback,estop_cb]
                tb_logger = create_logger(model_name,d_n,s_i)
                trainer = create_trainer(tb_logger,callbacks,gpu_kwargs,args.n_iterations)
                delete_old_saved_models(model_name, dspec.save_folder, s_i)
                trainer.fit(x2_y_gen, tloader)
                mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)

                if len(mod_names) > 1:
                    print(mod_names)
                    print('error duplicate model names')
                    assert 1 == 0
                elif len(mod_names)==1:
                    x2_y_gen = x2_y_gen.load_from_checkpoint(checkpoint_path=mod_names[0])
                else:
                    assert 1 == 0

                # training complete, save to list
                dsc_generators[dsc.labels[v_i]] = x2_y_gen
                dsc_generators[dsc.labels[v_i]].conditional_on_label = True
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = []


            elif (len(sv) >= 1) and (dsc.variable_types[v_i] == 'feature') and (np.all([c=='feature' for c in cond_vtype])):
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
                                            unlabel_batch_size=args.tot_bsize)

                tloader = SSLDataModule_X_from_X(orig_data_df=dsc.merge_dat,
                                                 tvar_names=cur_x_lab,
                                                 cvar_names=cond_x_lab,
                                                 cvar_types=cond_vtype,
                                                 labelled_batch_size=args.lab_bsize,
                                                 unlabelled_batch_size=args.tot_bsize,**vinfo_dict)
                model_name=create_model_name(dsc.labels[v_i],algo_variant)
                cond_vars = ''.join(cond_lab)

                if args.estop_mmd_type == 'val':
                    estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
                                                                           dspec.save_folder)  # returns max checkpoint

                elif args.estop_mmd_type == 'trans':
                    estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,
                                                                             dspec.save_folder)  # returns max checkpoint

                callbacks = [min_mmd_checkpoint_callback,estop_cb]
                tb_logger=create_logger(model_name,d_n,s_i)
                trainer=create_trainer(tb_logger,callbacks,gpu_kwargs,max_epochs=args.n_iterations)
                # deletions
                delete_old_saved_models(model_name, dspec.save_folder, s_i)
                # training
                trainer.fit(genx_x, tloader)
                mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)
                if len(mod_names) > 1:
                    print(mod_names)
                    print('error duplicate model names')
                    assert 1 == 0
                elif len(mod_names)==1:
                    genx_x = genx_x.load_from_checkpoint(checkpoint_path=mod_names[0])
                else:
                    assert 1 == 0
                # training complete, save to list
                dsc_generators[dsc.labels[v_i]] = copy.deepcopy(genx_x)
                dsc_generators[dsc.labels[v_i]].conditional_on_label=False
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab



                # Y  ->  X2
                # X1 ->
            elif len(sv) > 1 and dsc.variable_types[v_i] == 'feature' and 'label' in cond_vtype:

                conditional_feature_names=[]
                label_name=[]

                for cl,ct in zip(cond_lab,cond_vtype):
                    if ct=='feature':
                        conditional_feature_names.append(cl)
                    if ct=='label':
                        label_name.append(cl)

                mmd_vlabels = {}  # set dictionary for MMD
                dict_for_mmd= {}
                concat_cond_lab = []
                idx_feature=0
                for c in conditional_feature_names:

                    concat_cond_lab = concat_cond_lab + dsc.label_names_alphan[c]
                    mmd_vlabels[c]={'label_names_alphan':dsc.label_names_alphan[c]}

                    #get idx counter
                    idx_base=[i+idx_feature for i in range(len(dsc.label_names_alphan[c]))]
                    #increment current running idx
                    idx_feature=idx_base[-1]+1

                    mmd_vlabels[c].update({'dataloader_idx':idx_base})



                mmd_vlabels['vlist']=conditional_feature_names+label_name


                #get target variable names if multidimensional
                c=dsc.labels[v_i]
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
                mmd_vlabels[cur_x_lab]={}
                mmd_vlabels[cur_x_lab]['mpwd'] = median_pwd_target
                mmd_vlabels[cur_x_lab]['label_names_alphan'] = ldc
                mmd_vlabels[cur_x_lab]['is_target'] = True
                mmd_vlabels['target_variable']=cur_x_lab
                #mmd_vlabels[cur_x_lab]['mpwd'] = median_pwd_target

                #get idx of cond features for dataloader
                for c in conditional_feature_names:
                    relevant_columns=mmd_vlabels[c]['label_names_alphan']

                    cond_x_vals = all_dat[relevant_columns].values
                    median_pwd = get_median_pwd(torch.tensor(cond_x_vals))
                    mmd_vlabels[c]['mpwd']=median_pwd
                    mmd_vlabels[c]['sigma_list']=[median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]

                ################################
                #  median pwd for conditional x
                ################################
                # match column names
                cond_x_vals = all_dat[concat_cond_lab].values
                median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))





                input_dim=dsc.feature_dim+cond_x_vals.shape[1]+dsc.n_classes
                output_dim=dsc.feature_dim

                #dict_for_mmd = {'x1': 1, 'x3': 34, 'x5': [1, 2, 3, 4, 5]}




                genx2_yx1=Generator_X2_from_YX1(args.lr,
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
                                                unlabel_batch_size=args.tot_bsize,
                                                dict_for_mmd=mmd_vlabels)


                if dsc.feature_dim>1:
                    cur_x_lab=dsc.label_names_alphan[cur_x_lab]
                else:
                    cur_x_lab=cur_x_lab

                tloader=SSLDataModule_X2_from_Y_and_X1(
                        orig_data_df=dsc.merge_dat,
                        tvar_names=ldc, #change from cur_x_lab
                        cvar_names=concat_cond_lab,
                        label_var_name=label_name,
                        labelled_batch_size=args.lab_bsize,
                        unlabelled_batch_size=args.tot_bsize,
                        use_bernoulli = args.use_bernoulli,
                        causes_of_y=None,
                        **vinfo_dict)

                #train`
                model_name=create_model_name(dsc.labels[v_i],algo_variant)
                cond_vars=''.join(cond_lab)

                if args.estop_mmd_type == 'val':
                    estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_val_mmd(model_name,
                                                                           dspec.save_folder)  # returns max checkpoint

                elif args.estop_mmd_type == 'trans':
                    estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_trans_mmd(model_name,
                                                                             dspec.save_folder)  # returns max checkpoint



                callbacks = [min_mmd_checkpoint_callback, estop_cb]
                tb_logger = create_logger(model_name,args.d_n,s_i)
                trainer=create_trainer(tb_logger,callbacks,gpu_kwargs,max_epochs=args.n_iterations)
                delete_old_saved_models(model_name,dspec.save_folder,s_i) #delete old
                trainer.fit(genx2_yx1,tloader) #train here

                mod_names=return_saved_model_name(model_name,dspec.save_folder,dspec.dn_log,s_i)
                if len(mod_names)>1:
                    print('error duplicate model names')
                    assert 1==0
                elif len(mod_names)==1:
                    genx2_yx1 = genx2_yx1.load_from_checkpoint(checkpoint_path=mod_names[0])
                else:
                    assert 1==0

                #dsc_generators[dsc.labels[v_i]]=copy.deepcopy(genx2_yx1)


                dsc_generators[dsc.labels[v_i]]=copy.deepcopy(genx2_yx1)
                dsc_generators[dsc.labels[v_i]].conditional_on_label=True
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab



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


        n_samples=30000
        #n_samples = dsc.merge_dat.shape[0]
        synthetic_samples_dict=generate_samples_to_dict(dsc,has_gpu,dsc_generators,device_string,n_samples)
        joined_synthetic_data=samples_dict_to_df(dsc,synthetic_samples_dict,balance_labels=True,exact=False)



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
