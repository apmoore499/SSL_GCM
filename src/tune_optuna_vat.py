import optuna
from pytorch_lightning import LightningModule, Trainer



from benchmarks_SSL_GAN import *






from benchmarks_VAT import *





























# class SGANClassifier(pl.LightningModule):
#     def __init__(self, lr, d_n, s_i, alpha, input_dim,output_dim,disc_layers=[100,5],gen_layers=[100,5],dn_log,tot_bsize=None,best_value=None):




def run_vat_model(d_n,#=d_n,
                     s_i,#=s_i,
                     n_iterations,#=n_iterations,
                     tot_bsize,#=tot_bsize,
                     estop_patience,#=estop_patience,
                     metric,#=metric,
                     precision,#=precision,
                     gen_layers,
                     lab_bsize,#=lab_bsize,
                     lr,
                     n_trials=5,
                     min_epochs=10):#=lr)
    

    
    

    #add logic for setting hidden_layers




    # - delete this one?????????????

    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)
    # write dataset spec shorthand
    dspec = master_spec['dataset_spec']
    dspec.set_index("d_n", inplace=True)  # set this index for easier
    # store index of pandas loc where we find the value
    dspec = dspec.loc[d_n]  # use this as reerence..
    dspec.d_n = str(d_n) if dspec.d_n_type == 'str' else int(d_n)


    # now we want to read in dataset_si
    csi = master_spec['dataset_si'][dspec.d_n].values
    candidate_si = csi[~np.isnan(csi)]
    optimal_si_list = [int(s) for s in candidate_si]
    #if use_single_si == True:  # so we want to use single si, not entire range
    optimal_si_list = [s_i]

    #gpu_kwargs = get_gpu_kwargs(args)


    result_dict = {}
    results_list = []

    #for k, s_i in enumerate(optimal_si_list):
    result_dict[s_i] = 0
    results_list = []

    orig_data = load_data(d_n=d_n, s_i=s_i, dataset_folder=dspec.save_folder)  # load data


    model_name='VIRTUAL_ADVER_VAT_OPTUNA_VERIFY'



    # model_init_args={
    #     'd_n':args.d_n,
    #     's_i':si_iter,
    #     'input_dim':dspec.input_dim,
    #     'output_dim':2,
    #     'dn_log':dspec.dn_log
    # }



    model_init_args = {
        'd_n': d_n,
        's_i': s_i,
        'dn_log': dspec.dn_log,
        'input_dim':dspec['input_dim'],
        'output_dim': 2, #output dim for class label
        'tot_bsize':tot_bsize,
        'lr':lr,
        'current_model_name':model_name,
        'gen_layers':gen_layers,
        

    }

    VAT_default_args={
            'xi':10.0,
            'eps':1.0,
            'ip':1,
            'alpha':0.01,
            'lr':0.01,
            'lmda':0.01
            }
    
    model_init_args.update(VAT_default_args)
    
    
    
    current_model_name=model_init_args['current_model_name']

    # if args.use_tuned_hpms==True:
    #     #get json file that has been tuned
    #     #load
    #     params_dict_fn = f'{dspec.save_folder}/{model_name}.json'
    #     # export to json
    #     input_f=open(params_dict_fn,'r')
    #     tuned_hpms=json.load(input_f)
    #     model_init_args.update(tuned_hpms)


    dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns



    ssld = SSLDataModule(orig_data, tot_bsize=tot_bsize,lab_bsize=lab_bsize,precision=precision)#model_init_args['lab_bsize'])
    ssld.setup()  # initialise the data
    # get the data for validation
    val_features = ssld.data_validation[0][0].cuda().float()
    val_lab = ssld.data_validation[0][1].cuda().float()

    optimal_model = None
    optimal_trainer = None

    # START TIME
    st = time.time()

    gpu_kwargs={'precision':precision}
    
    
    for t in range(n_trials):
        print(f'doing s_i: {s_i}\t t: {t}\t of: {n_trials}')


        # TRAINING CALLBACKS
        callbacks = []
        max_pf_checkpoint_callback = return_chkpt_max_val_acc(model_init_args['current_model_name'],
                                                                dspec.save_folder)  # returns max checkpoint

        if metric == 'val_bce':
            estop_cb = return_estop_val_bce(patience=estop_patience)
        elif metric == 'val_acc':
            estop_cb = return_estop_val_acc(patience=estop_patience)

        callbacks.append(max_pf_checkpoint_callback)
        callbacks.append(estop_cb)

        # TENSORBOARD LOGGER
        tb_logger = get_default_logger(current_model_name, d_n, s_i, t)


        DETERMINISTIC_FLAG=False
        #set_trace()
        # TRAINER
        
        class dargs:
            def __init__(self) -> None:
                pass
        
        args=dargs()
        
        args.n_iterations=n_iterations
        args.d_n=d_n
        trainer = get_default_trainer(args, tb_logger, callbacks, DETERMINISTIC_FLAG, min_epochs=min_epochs, **gpu_kwargs)



        with trainer.init_module():
            # models created here will be on GPU and in float16
            # CREATE MODEL
            current_model = VATClassifier(**model_init_args)  # define model



        # INITIALISE WEIGHTS
        current_model.apply(init_weights_he_kaiming)  # re init model and weights




        from IPython.core.debugger import set_trace
        
        
        #set_trace()

        # DELETE OLD SAVED MODELS
        clear_saved_models(current_model_name, dspec.save_folder, s_i)#nuclear=True)

        # TRAIN
        trainer.fit(current_model, ssld)

        # LOAD OPTIMAL MODEL FROM CURRENT TRAINING

        #optimal model in 32 bit float for val metrics
        current_model = load_optimal_model(dspec, current_model).cuda().float()
        
        if optimal_model is not None:
            optimal_model = optimal_model.cuda().float()
        
        

        # COMPARE TO OVERALL OPTIMAL MODEL FROM THIS RUN
        optimal_model, optimal_trainer,optimal_acc = return_optimal_model(current_model,
                                                                trainer,
                                                                optimal_model,
                                                                optimal_trainer,
                                                                val_features.float(),
                                                                val_lab.float(),
                                                                metric=metric)

        del trainer


        if optimal_acc==1.0:
            break
    # END TIME
    #et = time.time()
    #print('time taken: {0} minutes'.format((et - st) / 60.))

    # DELETE OLD SAVED MODELS
    clear_saved_models(current_model.model_name, dspec.save_folder, s_i)

    # CREATE NAME TO SAVE MODEL
    model_save_fn = create_model_save_name(optimal_model, optimal_trainer, dspec)

    # SAVE THE TRAINER
    optimal_trainer.save_checkpoint(model_save_fn)

    # EVALUATE ON DATA
    res=evaluate_on_test_and_unlabel_for_optuna(dspec, args, s_i, current_model, optimal_model, orig_data, optimal_trainer)

    #print('plotting decision boundaries (plotly)')


    res.update(optimal_acc=optimal_acc)
    
    return(res)



        
from functools import partial



import numpy as np

def objective(trial,sel_si):

    #estop_patience=10
    metric='val_acc'

    n_iterations=100

    precision=32
    lr=1e-3
    tot_bsize=256
    lab_bsize=4
    n_trials=5
    estop_patience=5
    # Define hyperparameters to tune
    #lr = trial.suggest_loguniform('lr',1e-3,1e-1)
    #tot_bsize = trial.suggest_categorical('lab_bsize',[4,16,32])
    #lab_bsize = trial.suggest_categorical('tot_bsize',[16,32,64,128,256])
    
 
    
    params={
        
        "nhidden":trial.suggest_categorical('nhidden',[3,1]),
        "hidden_size":trial.suggest_categorical('hidden_size',[50,100,200]),
        #"d_n":trial.suggest_categorical('d_n',['n36_gaussian_mixture_d2_5000','n36_gaussian_mixture_d3_5000','n36_gaussian_mixture_d4_5000','n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d7_5000','n36_gaussian_mixture_d2_10000','n36_gaussian_mixture_d3_10000','n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d5_10000','n36_gaussian_mixture_d6_10000','n36_gaussian_mixture_d7_10000']),
        #"d_n":trial.suggest_categorical('d_n',['n36_gaussian_mixture_d2','n36_gaussian_mixture_d3','n36_gaussian_mixture_d4','n36_gaussian_mixture_d5','n36_gaussian_mixture_d6','n36_gaussian_mixture_d7''n36_gaussian_mixture_d2_5000','n36_gaussian_mixture_d3_5000','n36_gaussian_mixture_d4_5000','n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d7_5000','n36_gaussian_mixture_d2_10000','n36_gaussian_mixture_d3_10000','n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d5_10000','n36_gaussian_mixture_d6_10000','n36_gaussian_mixture_d7_10000'])
        "d_n":trial.suggest_categorical('d_n',['n36_gaussian_mixture_d2_10000']),#,'n36_gaussian_mixture_d3','n36_gaussian_mixture_d4','n36_gaussian_mixture_d5','n36_gaussian_mixture_d6','n36_gaussian_mixture_d7','n36_gaussian_mixture_d2_5000','n36_gaussian_mixture_d3_5000','n36_gaussian_mixture_d4_5000','n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d7_5000','n36_gaussian_mixture_d2_10000','n36_gaussian_mixture_d3_10000','n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d5_10000','n36_gaussian_mixture_d6_10000','n36_gaussian_mixture_d7_10000'])
    
    
    }
    
    #d_n = trial.suggest_categorical('d_n',[d_n])

    #$last_size = trial.suggest_categorical('last_size',[5,100])

    #hlayers=[hidden_size]*(nhidden)# + [last_size]
    hlayers=[params['hidden_size']]*(params['nhidden']) #+ [last_size]
    #sel_si=[1,2,3,4,5,6,7]#,8,9,10] #do a single si
    
    
    optimal_accs=[]
    ulab_accs=[]
    for s_i in sel_si[0]:
    
    
        res=run_vat_model(d_n=params['d_n'],
                                lr=lr,
                                s_i=s_i,
                                n_iterations=n_iterations,
                                tot_bsize=tot_bsize,
                                estop_patience=estop_patience,
                                metric=metric,
                                precision=precision,
                                lab_bsize=lab_bsize,
                            gen_layers=hlayers,
                            n_trials=n_trials)
                            
                            
        
                            
        optimal_accs.append(res['optimal_acc'])
        ulab_accs.append(res['unlabel_acc'])
        
        
    trial.set_user_attr("unlabel_acc_best", ulab_accs)


    return torch.tensor(optimal_accs).mean()#,torch.tensor(ulab_accs).mean()


from IPython.core.debugger import set_trace


#model_name='VAT_OPTUNA_VERIFY'
#model_name='SSL_GAN_OPTUNA_VERIFY'
model_name='VIRTUAL_ADVER_VAT_OPTUNA_VERIFY'

import pytorch_lightning

pytorch_lightning.seed_everything(1234)

#for d_n in []:
for d_n in ['n36_gaussian_mixture_d2_10000']:#,'n36_gaussian_mixture_d3','n36_gaussian_mixture_d4','n36_gaussian_mixture_d5','n36_gaussian_mixture_d6','n36_gaussian_mixture_d7','n36_gaussian_mixture_d2_5000','n36_gaussian_mixture_d3_5000','n36_gaussian_mixture_d4_5000','n36_gaussian_mixture_d5_5000','n36_gaussian_mixture_d6_5000','n36_gaussian_mixture_d7_5000','n36_gaussian_mixture_d2_10000','n36_gaussian_mixture_d3_10000','n36_gaussian_mixture_d4_10000','n36_gaussian_mixture_d5_10000','n36_gaussian_mixture_d6_10000','n36_gaussian_mixture_d7_10000']:

    study = optuna.create_study(direction='maximize')

    study.set_metric_names(['optimal_acc'])
    
    si_all=list(range(100))
    
    sel_si=torch.randperm(100)[:2].cpu().numpy()
    
    sel_si=[[si_all[s] for s in sel_si]]

    study.enqueue_trial({"hidden_size": 50,"nhidden":1,'d_n':d_n})
    study.enqueue_trial({"hidden_size": 50,"nhidden":3,'d_n':d_n})
    study.enqueue_trial({"hidden_size": 100,"nhidden":1,'d_n':d_n})
    study.enqueue_trial({"hidden_size": 100,"nhidden":3,'d_n':d_n})
    study.enqueue_trial({"hidden_size": 200,"nhidden":1,'d_n':d_n})
    study.enqueue_trial({"hidden_size": 200,"nhidden":3,'d_n':d_n})
    
    
    
    
    from functools import partial


    objective = partial(objective, sel_si = sel_si)# param2 = param2)
    study.optimize(objective, n_trials = 6)
    
    
    #objective = partial(objective, d_n = d_n)#, param2 = param2)
    #study.optimize(objective,n_trials=1)


    
    #study.optimize(objective, n_trials = 5)

    results_df=study.trials_dataframe()


    ttc=results_df['user_attrs_unlabel_acc_best']#.values
    ttc=[np.mean(t) for t in ttc]
    
    
    # if type(ttc[0])==list:
        
        
        
    
    
    #     ttc=[torch.hstack(t).mean().cpu().item() for t in ttc]
        
    # elif type(ttc)==float:
        
        
        
    # else:
        
    #     set_trace()
    results_df['unlabelled_pc_acc']=ttc
    results_df.drop(columns=['user_attrs_unlabel_acc_best'],inplace=True)


    pd_out_fn=os.path.join(f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{d_n}/saved_models/optuna_{model_name}_compiled_results.csv')
    results_df.to_csv(pd_out_fn)
    
    from IPython.core.debugger import set_trace
    
    
    set_trace()


# import optuna
# from pytorch_lightning import LightningModule, Trainer

# def objective(trial):
#     # Define hyperparameters to tune
#     #lr = trial.suggest_loguniform('lr',1e-3,1e-1)
#     #lab_bsize = trial.suggest_categorical('batch_size_lab',[4,16,32])
#     lr = 1e-2
#     tot_bsize = trial.suggest_categorical('batch_size_ulab',[32,64,128])#,256,512,1024,2048])
#     lab_bsize = 16


#     if args.estop_mmd_type == 'val':
#         estop_cb = return_early_stop_min_val_mmd(patience=estop_patience)
#         min_mmd_checkpoint_callback=return_chkpt_min_val_mmd(model_name, dspec.save_folder) #returns max checkpoint
#     elif args.estop_mmd_type == 'trans':
#         estop_cb = return_early_stop_min_trans_mmd(patience=estop_patience)
#         min_mmd_checkpoint_callback=return_chkpt_min_trans_mmd(model_name, dspec.save_folder) #returns max checkpoint
        
        
        
#     # Create nwe model with these hparam!
#     callbacks=[min_mmd_checkpoint_callback,estop_cb]
#     tb_logger = create_logger(model_name,d_n,s_i)
#     trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=args.n_iterations)
#     delete_old_saved_models(model_name,dspec.save_folder,s_i)


#     genx2_yx1=Generator_X2_from_YX1(lr,
#                                     d_n,
#                                     s_i,
#                                     dspec.dn_log,
#                                     input_dim=input_dim,
#                                     output_dim=output_dim,
#                                     median_pwd_tx=median_pwd_target,
#                                     median_pwd_cx=median_pwd_cond,
#                                     num_hidden_layer=args.nhidden_layer,
#                                     middle_layer_size=args.n_neuron_hidden,
#                                     n_lab=n_lab,
#                                     n_ulab=n_ulab,
#                                     label_batch_size=args.lab_bsize,
#                                     unlabel_batch_size=args.tot_bsize,
#                                     dict_for_mmd=mmd_vlabels)




#     tloader=SSLDataModule_X2_from_Y_and_X1(
#         orig_data_df=dsc.merge_dat,
#         tvar_names=ldc, #change from cur_x_lab
#         cvar_names=concat_cond_lab,
#         label_var_name=label_name,
#         labelled_batch_size=lab_bsize,
#         unlabelled_batch_size=tot_bsize,
#         use_bernoulli = args.use_bernoulli,
#         causes_of_y=None,
#         **vinfo_dict)


#     trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=15)
#     trainer.fit(genx2_yx1,tloader)

#     return genx2_yx1.vmmd_losses[-1]








# study = optuna.create_study(direction='minimize')



# study.optimize(objective,n_trials=20)

# study.best_params
