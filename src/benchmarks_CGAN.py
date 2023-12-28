


############################
#
#     Causal GAN - Base Class
#     Trains a sequence of generators to form hierarchical representation of P(X,Y)
#
############################


import sys
sys.path.append('generative_models')
sys.path.append('src')
sys.path.append('src/generative_models/')

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
from benchmarks_utils import *
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
    #parser.add_argument('--estop_mmd_type',help='callback for early stopping. either use val mmd or trans mmd, val or trans respectively',default='val')
    parser.add_argument('--estop_mmd_type',help='callback for early stopping. either use val mmd or trans mmd, val or trans respectively ADD E[val + trans] val_trans',default='val')
    parser.add_argument('--use_tuned_hpms',help='use tuned hyperparameters',default='False')
    parser.add_argument('--precision',help='precision used by trainer, use 16 for fast on applicable gpu',default=32)
    parser.add_argument('--plot_synthetic_dist',help='plotting of synthetic data (take extra time), not necessary',default='False')
    #parser.add_argument('--precision',help='traainer precision ie 32,16',default='32')
    parser.add_argument('--compile_mmd_mode',help='compile mode for mmd losses',default='reduce-overhead')
    parser.add_argument('--ignore_plot_5',help='ignore_first_plot_five, if true then dont do any plot',default='False')
    parser.add_argument('--scale_for_indiv_plot',help='scale for pltoting',default=5)
    parser.add_argument('--synthesise_w_cs',help='synthesise new data using groudn truth cause/spouse values',default='False')
    parser.add_argument('--lambda_U',help='lambda_U for influence of unlabeleld data on loss',type=float,default=1.0)








    args = parser.parse_args()
    args.use_single_si=str_to_bool(args.use_single_si)
    args.use_bernoulli=str_to_bool(args.use_bernoulli)
    args.use_benchmark_generators=str_to_bool(args.use_benchmark_generators)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)
    args.ignore_plot_5 = str_to_bool(args.ignore_plot_5)
    args.plot_synthetic_dist = str_to_bool(args.plot_synthetic_dist)
    
    
    
    #plot_synthetic_dist
    
    args.synthesise_w_cs = str_to_bool(args.synthesise_w_cs)
    
    
    

    if args.use_tuned_hpms==True:
        print('args flag for use tuned hparams is TRUE, but hpms not available for CGAN method: running algorithm without any tuned hpms')

    st=time.time()
    algo_variant=''

    if args.use_bernoulli==False:
        algo_variant='basic'
    elif args.use_bernoulli==True:
        algo_variant='marginal'
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
        args.optimal_si_list=[i for i in range(100)]





    dict_of_precompiled=return_dict_of_precompiled_mmd(args.compile_mmd_mode)
















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

        # convert to unique elements

        # then remove variables shared bw parent&spouse to just parent

        for k in mb_label_var.keys():
            mb_label_var[k] = list(set(mb_label_var[k]))

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

        cause_spouse_v_idx=ce_dict['cause']+ce_dict['spouse']
        label_v_idx=ce_dict['lab']
        effect_v_idx=ce_dict['effect']

        #dsc_generators['ordered_v']={} #to be used later on for retrieving variable names, in correct order
        #dsc_generators['ordered_v_alphan']={}
        #we train as follows:

        #1. cause + spouse together
        #2. label
        #3. effect variables

        #we need to get the relvant idx for input of variables downstream

        #train the cause_spouse_v_idx ones













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
        
        





        dsc_idx_dict = {}
        dsc_idx_dict['cause_spouse'] = np.array(dsc_generators['ordered_v_alphan']['cause'])

        
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












        print(f'unlabelled_keys: {unlabelled_keys}')
        print(f'labelled_key: {labelled_key}')
        print(f'conditional_keys: {conditional_keys}')

        # get all keys together
        all_keys=unlabelled_keys+[labelled_key]+conditional_keys




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





        







        all_x_cols=[c for c in all_unlabelled.columns]

        #convert column names to index
        causes_of_y_idx_dl=[]

        for k,cf in enumerate(all_x_cols):
            if cf in causes_of_y_feat_names:
                causes_of_y_idx_dl.append(k)



            
            
            
            
            
            # dsc_generators[dsc.labels[v_i]] = gen_x
            # dsc_generators[dsc.labels[v_i]].conditional_on_label = False
            # dsc_generators[dsc.labels[v_i]].conditional_feature_names = []
            
        
        
        del trainer
        
        
        del tloader









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
                
                print('assuming we combine xc,xs for cgan joint method, skipping')






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
                trainer = create_trainer(tb_logger, callbacks, gpu_kwargs, max_epochs=args.n_iterations)#,gradient_clip_val=0.5)
                delete_old_saved_models(model_name, dspec.save_folder, s_i)
                trainer.fit(y_x1gen,tloader) #train here
                mod_names=return_saved_model_name(model_name,dspec.save_folder,d_n,s_i)
                if len(mod_names)>1:
                    print('error duplicate model names')
                    assert 1==0
                y_x1gen=type(y_x1gen).load_from_checkpoint(checkpoint_path=mod_names[0])
                dsc_generators[dsc.labels[v_i]]=copy.deepcopy(y_x1gen)
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab
                dsc_generators[dsc.labels[v_i]].conditional_on_label = False
                
                
                
                dsc_generators['ordered_v']['label']={}
                dsc_generators['ordered_v']['label']['inputs']=cond_lab
                dsc_generators['ordered_v']['label']['inputs_alphan'] = concat_cond_lab
                dsc_generators['ordered_v']['label']['output']=dsc.labels[v_i]
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
                
                cur_x_lab=dsc.label_names[v_i]
                
                # get the matching data...
                all_vals = dsc.merge_dat[dsc.merge_dat.type.isin(['labelled', 'unlabelled'])]
                # match column names
                # subset
                
                
                x_vals=torch.tensor(all_vals[cur_x_lab].to_numpy('float32'),device=torch.device('cuda'))
                #get median pwd
                median_pwd=get_median_pwd(x_vals).item()
                
                
                #x_vals = all_vals[dsc.label_names[v_i]].values
                # get median pwd
                #median_pwd = get_median_pwd(torch.tensor(x_vals))
                n_lab=dsc.merge_dat[dsc.merge_dat.type=='labelled'].shape[0]
                n_ulab = x_vals.shape[0]

                dict_for_mmd={'x1':1,'x3':34,'x5':[1,2,3,4,5]}
                
                
                # get data loader
                tloader = SSLDataModule_X_from_Y(orig_data_df=dsc.merge_dat,
                                                 tvar_name=dsc.label_names[v_i],
                                                 cvar_name=cond_lab,
                                                 cvar_type='label',
                                                 labelled_batch_size=args.lab_bsize,
                                                 unlabelled_batch_size=args.tot_bsize,
                                                 **vinfo_dict)
                
                tloader.setup()
                
                x_l=tloader.x_l
                y_l=tloader.y_l
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
                                               unlabel_batch_size=args.tot_bsize,
                                               x_l=x_l,
                                               y_l=y_l)# this is it

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


                elif args.estop_mmd_type == 'val_trans':
                    estop_cb = return_early_stop_min_val_trans_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_val_trans_mmd(model_name,
                                                                             dspec.save_folder)  # returns max checkpoint


                callbacks=[min_mmd_checkpoint_callback,estop_cb]
                
                x2_y_gen.set_precompiled(dict_of_precompiled)
                
                
                profiler=None
                #profiler = AdvancedProfiler(dirpath='/media/krillman/240GB_DATA/codes2/SSL_GCM/src/profiling',filename='profile_xe_from_y.log')
                
                tb_logger = create_logger(model_name,d_n,s_i)
                trainer = create_trainer(tb_logger,callbacks,gpu_kwargs,args.n_iterations,precision=args.precision,profiler=profiler)
                delete_old_saved_models(model_name, dspec.save_folder, s_i)
                trainer.fit(x2_y_gen, tloader)
                mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, s_i)

                if len(mod_names) > 1:
                    print(mod_names)
                    print('error duplicate model names')
                    assert 1 == 0
                elif len(mod_names)==1:
                    x2_y_gen = type(x2_y_gen).load_from_checkpoint(checkpoint_path=mod_names[0])
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
                #target_x_vals = all_dat[ldc].values
                #median_pwd_target = get_median_pwd(torch.tensor(target_x_vals))


                target_x_vals=torch.tensor(all_dat[ldc].to_numpy('float32'),device=torch.device('cuda'))
                #get median median_pwd_target
                median_pwd_target=get_median_pwd(target_x_vals).item()
                

                ################################
                #  median pwd for conditional x
                ################################
                # match column names
                #ldc = [c for c in lab_dat.columns if any(c in cond_x_lab)]
                #cond_x_vals = all_dat[cond_x_lab].values
                #median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))


                
                cond_x_vals=torch.tensor(all_dat[cond_x_lab].to_numpy('float32'),device=torch.device('cuda'))
                #get median pwd
                median_pwd_cond=get_median_pwd(cond_x_vals).item()
                

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



                elif args.estop_mmd_type == 'val_trans':
                    estop_cb = return_early_stop_min_val_trans_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_val_trans_mmd(model_name,
                                                                             dspec.save_folder)  # returns max checkpoint







                callbacks = [min_mmd_checkpoint_callback,estop_cb]
                tb_logger=create_logger(model_name,d_n,s_i)
                trainer=create_trainer(tb_logger,callbacks,gpu_kwargs,max_epochs=args.n_iterations,precision=args.precision)
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
                    genx_x = type(genx_x).load_from_checkpoint(checkpoint_path=mod_names[0])
                else:
                    assert 1 == 0
                # training complete, save to list
                dsc_generators[dsc.labels[v_i]] = copy.deepcopy(genx_x)
                dsc_generators[dsc.labels[v_i]].conditional_on_label=False
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = cond_lab



                # Y  ->  X2
                # X1 ->
            elif len(sv) > 1 and dsc.variable_types[v_i] == 'feature' and 'label' in cond_vtype:
                
                
                from IPython.core.debugger import set_trace
                
                #set_trace()
                
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
                #target_x_vals = all_dat[ldc].values
                
                
                
                
                
                
                
                x_vals=torch.tensor(all_dat[ldc].to_numpy('float32'),device=torch.device('cuda'))
                #get median pwd
                median_pwd_target=get_median_pwd(x_vals).item()
                
                
                
                
                
                #median_pwd_target = get_median_pwd(torch.tensor(target_x_vals))
                mmd_vlabels[cur_x_lab]={}
                mmd_vlabels[cur_x_lab]['mpwd'] = median_pwd_target
                mmd_vlabels[cur_x_lab]['label_names_alphan'] = ldc
                mmd_vlabels[cur_x_lab]['is_target'] = True
                mmd_vlabels['target_variable']=cur_x_lab
                #mmd_vlabels[cur_x_lab]['mpwd'] = median_pwd_target

                #get idx of cond features for dataloader
                for c in conditional_feature_names:
                    relevant_columns=mmd_vlabels[c]['label_names_alphan']

                    #cond_x_vals = all_dat[relevant_columns].values
                    #median_pwd = get_median_pwd(torch.tensor(cond_x_vals))
                    
                    
                                    
                
                    
                    cond_x_vals=torch.tensor(all_dat[relevant_columns].to_numpy('float32'),device=torch.device('cuda'))
                    #get median pwd
                    median_pwd=get_median_pwd(cond_x_vals).item()
                    
                        
                        
                        
                    
                    
                    
                    
                    
                    
                    
                    mmd_vlabels[c]['mpwd']=median_pwd
                    mmd_vlabels[c]['sigma_list']=[median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]

                ################################
                #  median pwd for conditional x
                ################################
                # match column names
                
                
                
                
                
                                
                cond_x_vals=torch.tensor(all_dat[concat_cond_lab].to_numpy('float32'),device=torch.device('cuda'))
                #get median pwd
                median_pwd_cond=get_median_pwd(cond_x_vals).item()
                
                    
            
                
                
                
                # cond_x_vals = all_dat[concat_cond_lab].values
                # median_pwd_cond = get_median_pwd(torch.tensor(cond_x_vals))





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
                                                label_batch_size=32,#args.lab_bsize,
                                                unlabel_batch_size=args.tot_bsize,
                                                labmda_U=args.lambda_U)
                                                #dict_for_mmd=mmd_vlabels)


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
                
                
                elif args.estop_mmd_type == 'val_trans':
                    estop_cb = return_early_stop_min_val_trans_mmd(patience=args.estop_patience)
                    min_mmd_checkpoint_callback = return_chkpt_min_val_trans_mmd(model_name,
                                                                             dspec.save_folder)  # returns max checkpoint

                
                profiler = None #AdvancedProfiler(dirpath='/media/krillman/240GB_DATA/codes2/SSL_GCM/src/profiling',filename='profile_x2y.log')

                genx2_yx1.set_precompiled(dict_of_precompiled)


                callbacks = [min_mmd_checkpoint_callback, estop_cb]
                tb_logger = create_logger(model_name,args.d_n,s_i)
                trainer=create_trainer(tb_logger,callbacks,gpu_kwargs,max_epochs=args.n_iterations,profiler=profiler,precision=args.precision)
                delete_old_saved_models(model_name,dspec.save_folder,s_i) #delete old
                trainer.fit(genx2_yx1,tloader) #train here

                mod_names=return_saved_model_name(model_name,dspec.save_folder,dspec.dn_log,s_i)
                if len(mod_names)>1:
                    print('error duplicate model names')
                    assert 1==0
                elif len(mod_names)==1:
                    genx2_yx1 = type(genx2_yx1).load_from_checkpoint(checkpoint_path=mod_names[0])
                else:
                    assert 1==0

                #dsc_generators[dsc.labels[v_i]]=copy.deepcopy(genx2_yx1)


                dsc_generators[dsc.labels[v_i]]=copy.deepcopy(genx2_yx1)
                dsc_generators[dsc.labels[v_i]].conditional_on_label=True
                dsc_generators[dsc.labels[v_i]].conditional_feature_names = conditional_feature_names #change from cond_lab, cond_lab contaminated by label var...



        print('training complete: now synthesise synthetic data')

        if args.use_benchmark_generators and args.use_bernoulli:
            # we need to replace the generators in dsc_generators,
            for k in unlabelled_keys + [labelled_key]:
                # find model name...
                bmodel=create_model_name(k,'basic')
                model_to_search_for=dspec.save_folder+'/saved_models/'+bmodel+"*-s_i={0}-epoch*".format(s_i)
                candidate_models=glob.glob(model_to_search_for)
                dsc_generators[k]=type(dsc_generators[k]).load_from_checkpoint(checkpoint_path=candidate_models[0])

        # generating synthetic data
        # generate 10000 samples


        n_samples=int(30000*min(dspec.n_unlabelled/1000,5)) #try to set this to deal wtih very large unalbeleld size...
        #n_samples = dsc.merge_dat.shape[0]
        
        
        n_samples=int(30000*5)#*min(dspec.n_unlabelled/1000,5)) #try to set this to deal wtih very large unalbeleld size...
        
        
        
            
            
            
            
            
        
        
        
        
        
        if not args.synthesise_w_cs:
        
            synthetic_samples_dict = generate_samples_to_dict_tripartite(dsc=dsc, has_gpu=has_gpu, dsc_generators=dsc_generators, device_string=device_string,n_samples=n_samples,use_gt_cspouse=False)
            joined_synthetic_data = samples_dict_to_df(dsc, synthetic_samples_dict, balance_labels=True,exact=False,extra_sample_frac=1.0,resample=True)
        
        
        
        
        
        
        
        
        if args.synthesise_w_cs:
        
        
        
            synthetic_samples_dict = generate_samples_to_dict_tripartite(dsc=dsc, has_gpu=has_gpu, dsc_generators=dsc_generators, device_string=device_string,n_samples=n_samples,use_gt_cspouse=True) #use cause spouse variables as antecedents rather than synthyesising from the generator
            joined_synthetic_data = samples_dict_to_df(dsc, synthetic_samples_dict, balance_labels=True,exact=False,extra_sample_frac=1.0,resample=False)
        
        
    #def generate_samples_to_dict_tripartite(dsc, has_gpu, dsc_generators, device_string,n_samples=10000,gumbel=False,tau=None):
        
        
        #synthetic_samples_dict = generate_samples_to_dict(dsc, has_gpu, dsc_generators, device_string, n_samples,tau=0.5)
        #joined_synthetic_data = samples_dict_to_df(dsc, synthetic_samples_dict, balance_labels=True,exact=False,extra_sample_frac=1.0,resample=False)

        
        
        
        
        
        
        #set_trace()
        
        
        algo_spec = copy.deepcopy(master_spec['algo_spec'])
        #algo_spec.set_index('algo_variant', inplace=True)

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
        
        
        
        
        
        
        
        
        
        
        
        #set_trace()
        
        
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
        
        #del optimal_mods
        
        #del GumbelModuleCombined
        
       # del old_dsc_generators
        
        #del data_module
        
        #del gumbel_module
        
        del trainer
        
        del dsc
        
        
        #del unlabelled
        
        
        del joined_synthetic_data
        
        del synthetic_samples_dict
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        