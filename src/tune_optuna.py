import optuna
from pytorch_lightning import LightningModule, Trainer

def objective(trial):
	# Define hyperparameters to tune
	lr = trial.suggest_loguniform('lr',1e-3,1e-1)
	tot_bsize = trial.suggest_categorical('batch_size',[256,512])
	if args.estop_mmd_type == 'val':
	    estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
	    min_mmd_checkpoint_callback=return_chkpt_min_val_mmd(model_name, dspec.save_folder) #returns max checkpoint
	elif args.estop_mmd_type == 'trans':
	    estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
	    min_mmd_checkpoint_callback=return_chkpt_min_trans_mmd(model_name, dspec.save_folder) #returns max checkpoint
	# Create nwe model with these hparam!
	callbacks=[min_mmd_checkpoint_callback,estop_cb]
	tb_logger = create_logger(model_name,d_n,s_i)
	trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=args.n_iterations)
	delete_old_saved_models(model_name,dspec.save_folder,s_i)
	gen_x=Generator_X1(lr,args.d_n,
	                               s_i,
	                               dspec.dn_log,
	                               input_dim=dsc.feature_dim,
	                               median_pwd=median_pwd,
	                               num_hidden_layer=args.nhidden_layer,
	                               middle_layer_size=args.n_neuron_hidden,
	                               label_batch_size=args.lab_bsize,
	                               unlabel_batch_size=tot_bsize)
	trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=10)
	trainer.fit(gen_x,tloader)
	return gen_x.vmmd_losses[-1]


study = optuna.create_study(direction='minimize')



study.optimize(objective,n_trials=10)

study.best_params









import optuna
from pytorch_lightning import LightningModule, Trainer

def objective(trial):
    # Define hyperparameters to tune
    #lr = trial.suggest_loguniform('lr',1e-3,1e-1)
    #lab_bsize = trial.suggest_categorical('batch_size_lab',[4,16,32])
    lr = 1e-2
    tot_bsize = trial.suggest_categorical('batch_size_ulab',[32,64,128])#,256,512,1024,2048])
    lab_bsize = 16


    if args.estop_mmd_type == 'val':
        estop_cb = return_early_stop_min_val_mmd(patience=args.estop_patience)
        min_mmd_checkpoint_callback=return_chkpt_min_val_mmd(model_name, dspec.save_folder) #returns max checkpoint
    elif args.estop_mmd_type == 'trans':
        estop_cb = return_early_stop_min_trans_mmd(patience=args.estop_patience)
        min_mmd_checkpoint_callback=return_chkpt_min_trans_mmd(model_name, dspec.save_folder) #returns max checkpoint
    # Create nwe model with these hparam!
    callbacks=[min_mmd_checkpoint_callback,estop_cb]
    tb_logger = create_logger(model_name,d_n,s_i)
    trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=args.n_iterations)
    delete_old_saved_models(model_name,dspec.save_folder,s_i)


    genx2_yx1=Generator_X2_from_YX1(lr,
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




    tloader=SSLDataModule_X2_from_Y_and_X1(
        orig_data_df=dsc.merge_dat,
        tvar_names=ldc, #change from cur_x_lab
        cvar_names=concat_cond_lab,
        label_var_name=label_name,
        labelled_batch_size=lab_bsize,
        unlabelled_batch_size=tot_bsize,
        use_bernoulli = args.use_bernoulli,
        causes_of_y=None,
        **vinfo_dict)


    trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=15)
    trainer.fit(genx2_yx1,tloader)

    return genx2_yx1.vmmd_losses[-1]








study = optuna.create_study(direction='minimize')



study.optimize(objective,n_trials=20)

study.best_params
