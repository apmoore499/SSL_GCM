
import sys
sys.path.append('generative_models')
sys.path.append('src')
sys.path.append('src/generative_models/')


from benchmarks_utils import *
import argparse
import time

n_classes=2


get_softmax = torch.nn.Softmax(dim=0)

NUM_WORKERS=0
import pytorch_lightning as pl
from typing import Optional
from benchmarks_cgan import *
from Generator_Y import *

#n_classes=2







class MMDGAN(pl.LightningModule):
    def __init__(self,
                 lr, 
                 d_n, 
                 dn_log,
                 s_i, 
                 median_pwd,
                 input_dim,
                 n_class=2,
                 alpha_ul=1.0, 
                 y_l=None,
                 tot_bsize=None,
                 num_unlabelled=None,
                 current_model_name='MMD_GAN',
                 gen_layers=[100,5],
                 device_str='cuda',
                 precision=16):
        super().__init__()

        #self.save_hyperparameters('alpha_ul','input_dim','current_model_name','dn_log','lr','median_pwd','n_class','s_i')
        self.save_hyperparameters()#'alpha_ul','input_dim','current_model_name','dn_log','lr','median_pwd','n_class','s_i')
        
        
        
        self.gen = make_mlp(input_dim=input_dim+n_class,
                            hidden_layers=gen_layers,
                            output_dim=input_dim)


        self.val_accs = []

        self.model_name = current_model_name
        
        
        self.hparams['dn_log']=int(self.hparams['dn_log'])
        self.hparams['s_i']=int(self.hparams['s_i'])#,device=torch.device('cuda'))

        

        self.vmmd_losses=[]
        
        
        if type(y_l)==torch.tensor:
            y_l=y_l.cpu().numpy()
            
        if type(y_l)==list:
            y_l=np.array(y_l)
            
        self.y_labelled_for_resample=y_l

        self.sel_device=device_str

        self.precision=precision
        
        
        
        
        




        self.noise_placeholder_entire=torch.zeros((num_unlabelled+1000,input_dim),device=torch.device('cuda'))
        self.noise_placeholder_train=torch.zeros((tot_bsize,input_dim),device=torch.device('cuda'))

    def set_precompiled(self,dop):
        
        sel_device=self.sel_device
                
        sel_dtype=torch.float32
        if self.precision==16:
            sel_dtype=torch.float16
            
        if 'gpu' in sel_device or 'cuda' in sel_device:
            sel_device=torch.device('cuda')
        else:
            sel_device=torch.device('cpu')
            
        
        self.sigma_list=torch.tensor([self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]],dtype=sel_dtype,device=sel_device)
        self.y_labelled_for_resample=torch.tensor(self.y_labelled_for_resample.cpu().numpy(),dtype=sel_dtype,device=sel_device)
        #self.sigma_list_cond_x=torch.tensor([self.hparams.median_pwd_cx * x for x in [0.125, 0.25, 0.5, 1, 2]],dtype=sel_dtype,device=sel_device)
        self.n_labelled_for_train=self.y_labelled_for_resample.shape[0]    
        
        #rbf_kern=dop['mix_rbf_mmd2']
        #X=torch.randn((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
        #dummy=rbf_kern(X,X)
        
        #self.rbf_kern=rbf_kern
        
        
        
        self.dop=dop
        
        
        return self
        
    def delete_compiled_modules(self):
        
        del self.dop
        
        return self



    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x

    def training_step(self, batch, batch_idx):
        #opt_d,opt_g=self.optimizers()

        
        x_l, y_l, x_ul = batch
        # sample noise
        #z_u = torch.randn_like(x_ul)
        
        
                
        self.noise_placeholder_entire.normal_()
        self.noise_placeholder_train.normal_()
        
        
        
        
        z_u=self.noise_placeholder_entire[:x_ul.shape[0],]#.normal_()
        
        
        #y_labelled_for_resample
        
        #resample y_l.....
        
        n_ul = x_ul.shape[0]
        dummy_label_weights = torch.ones(self.n_labelled_for_train)
        resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_ul, replacement=True)
        
        y_rs=self.y_labelled_for_resample[resampled_i]
        
        
        input_ul=torch.cat((z_u,y_rs),1).float()
        
        # generate x for unlabelled
        x_hat_ul = self.gen(input_ul)
        
        
        
        #generate x for labelled
        
        z_l = self.noise_placeholder_train[:x_l.shape[0],]
        input_l=torch.cat((z_l,y_l),1).float()
        x_hat_l = self.gen(input_l)
        
        
        all_x=torch.cat((x_l,x_ul),dim=0)
        
        all_x_hat=torch.cat((x_hat_l,x_hat_ul),dim=0)
        
        
        #labelled MMD
        labelled_loss=self.dop['mix_rbf_mmd2_joint_1_feature_1_label'](x_hat_l,x_l,y_l,y_l,self.sigma_list)

        #unlabelled MMD
        marginal_loss=self.dop['mix_rbf_mmd2'](all_x_hat,all_x,self.sigma_list)


        total_loss=labelled_loss+marginal_loss*self.hparams['alpha_ul']
        
        

        self.log_dict({"labelled_loss": labelled_loss, "marginal_loss": marginal_loss,"total_loss":total_loss},prog_bar=True)#"temp":self.temp})

        return(total_loss)













    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        
        #sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]
        
        
        
        #--------------------------------------
        #   VALIDATION DATA (LABELLED)
        #--------------------------------------
        
        
        val_feat=batch[0].squeeze(0).view((-1,self.hparams['input_dim']))#.float()
        val_y=batch[1].squeeze(0).float()
        
        self.noise_placeholder_entire.normal_()
        self.noise_placeholder_train.normal_()
        
        z_l=self.noise_placeholder_train[:val_feat.shape[0],]#torch.randn_like(val_feat) #sample noise
        input_l=torch.cat((z_l,val_y),1)#.float() #concatentate noise with label info
        x_hat_l=self.gen(input_l) #generate x samples random
        
        val_mmd_loss=self.dop['mix_rbf_mmd2_joint_1_feature_1_label'](x_hat_l,val_feat,val_y,val_y,self.sigma_list).clone()
                
        
        
        
        #--------------------------------------
        #   VALIDATION DATA (UNLABELLED)
        #--------------------------------------
        
        
        trans_feat=batch[2].squeeze(0).reshape((-1,self.hparams['input_dim']))

        
        #marginal_loss=self.dop['mix_rbf_mmd2'](all_x_hat,all_x,self.sigma_list)
        n_ul = trans_feat.shape[0]
        dummy_label_weights = torch.ones(self.n_labelled_for_train)
        resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_ul, replacement=True)
        
        y_rs=self.y_labelled_for_resample[resampled_i]
        
        #z_u=torch.randn_like(trans_feat) #sample noise
        z_u=self.noise_placeholder_entire[:trans_feat.shape[0],]
        
        input_ul=torch.cat((z_u,y_rs),1)#.float()
        
        # generate x for unlabelled
        x_hat_ul = self.gen(input_ul)
        
        #pull in x_l also...
        x_all=torch.cat((val_feat,trans_feat),dim=0)
        x_hat_all=torch.cat((x_hat_l,x_hat_ul),dim=0)
        
        
        trans_mmd_loss=self.dop['mix_rbf_mmd2'](x_hat_all,x_all,self.sigma_list).clone()
        
        
        
        val_trans_mmd=val_mmd_loss+self.hparams.alpha_ul*trans_mmd_loss
        
        
        #--------------------------------------
        #   LOG LOSSES
        #--------------------------------------

        self.log("val_mmd", val_mmd_loss)
        self.log("trans_mmd",trans_mmd_loss)
        self.log("val_trans_mmd",val_trans_mmd)
        

        self.log("s_i",self.hparams.s_i)
        self.log("d_n",self.hparams.dn_log)
        
        
        



# -----------------------------------
#     SEMI SUPERVISED LEARNING DATA MODULE
# -----------------------------------


class SSLDataModule(pl.LightningDataModule):
    def __init__(self, orig_data,lab_bsize,precision,tot_bsize: int = 64):
        super().__init__()
        self.orig_data = orig_data
        self.tot_bsize = tot_bsize
        
        self.lab_bsize=lab_bsize
        self.precision=str(precision)

    def setup(self, stage: Optional[str] = None):

        orig_data = self.orig_data

        # ----------#
        # Training Labelled
        # ----------#
        X_train_lab = orig_data['label_features']
        y_train_lab = orig_data['label_y']#torch.argmax(orig_data['label_y'], 1)

        # ----------#
        # Training Unlabelled
        # ----------#
        X_train_ulab = orig_data['unlabel_features']
        y_train_ulab = orig_data['unlabel_y']#torch.argmax(orig_data['unlabel_y'], 1)

        # -------------#
        # Validation
        # -------------#

        X_val = orig_data['val_features']
        #y_val = torch.argmax(orig_data['val_y'], 1)
        
        y_val = orig_data['val_y']

        # -------------#
        # Setting up resampling
        # -------------#

        n_unlabelled = X_train_ulab.shape[0]
        n_labelled = X_train_lab.shape[0]
        dummy_label_weights = torch.ones(n_labelled)
        resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_unlabelled, replacement=True)

        # ulab_mix is the data train!
        vfeat = X_val.unsqueeze(0)
        vlab = y_val.unsqueeze(0)
        tfeat=X_train_ulab.unsqueeze(0)
        
        if self.precision=='16':
            X_train_lab = orig_data['label_features'].cuda().half()
            y_train_lab =orig_data['label_y'].cuda().half()
            
            X_train_lab_rs = X_train_lab[resampled_i]
            y_train_lab_rs = y_train_lab[resampled_i]
            
            X_val=vfeat.cuda().half()
            y_val=vlab.cuda().half()
            
            
            X_train_ulab = orig_data['unlabel_features'].cuda().half()
            y_train_ulab = orig_data['unlabel_features'].cuda().half()
            
            
        elif self.precision=='32':
            X_train_lab = orig_data['label_features'].cuda().float()
            y_train_lab = orig_data['label_y'].cuda().float()
            
            X_train_lab_rs = X_train_lab[resampled_i]
            y_train_lab_rs = y_train_lab[resampled_i]
            
            vfeat=vfeat.cuda().float()
            vlab=vlab.cuda().float()
            
        
            X_train_ulab = orig_data['unlabel_features'].cuda().float()
            y_train_ulab = orig_data['unlabel_features'].cuda().float()
            
        
        self.data_train = torch.utils.data.TensorDataset(X_train_lab_rs,
                                                         y_train_lab_rs,
                                                         X_train_ulab)

        self.data_validation = torch.utils.data.TensorDataset(vfeat, vlab,tfeat)
        self.nval = vlab.shape[0]

        return (self)


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.tot_bsize, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.nval)



















#dict_of_precompiled['mix_rbf_kernel']=torch.compile(mix_rbf_kernel_class().to(torch.float16).cuda(),fullgraph=True,mode='max-autotune')
# dict_of_precompiled['mix_rbf_mmd2']=torch.compile(mix_rbf_mmd2_class().to(torch.float16).cuda(),fullgraph=True,mode='max-autotune')
# dict_of_precompiled['mix_rbf_mmd2_joint_1_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='max-autotune')
# dict_of_precompiled['mix_rbf_mmd2_joint_regress_2_feature']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature().to(torch.float16).cuda(),fullgraph=True,mode='max-autotune')
# dict_of_precompiled['mix_rbf_mmd2_joint_regress_2_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='max-autotune')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_n', help='dataset number 1-5 OR MOG',type=str)
    parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ',type=int)
    parser.add_argument('--n_iterations', help='how many iterations to train classifier for',type=int,default=100)
    parser.add_argument('--lr',help='learning rate ie 1e-2, 1e-3,...',type=float,default=1e-3)
    parser.add_argument('--lambda_U',help='lambda_U for influence of unlabeleld data on loss',type=float,default=1.0)
    parser.add_argument('--use_single_si',help='do we want to train on collection of si or only single instance',type=str,default='True')
    parser.add_argument('--use_bernoulli',help='use bernoulli for y given x',type=str,default='False')
    parser.add_argument('--use_benchmark_generators',help='using benchmark generators or not',type=str,default='False')
    parser.add_argument('--lab_bsize',help='label batch size',type=int,default=4)
    parser.add_argument('--tot_bsize',help='total batch size',type=int,default=128)
    parser.add_argument('--estop_patience',help='patience for training generators, stop trianing if no improve after this # epoch',type=int,default=10)
    parser.add_argument('--nhidden_layer',help='how many hidden layers in implicit model:1,3,5',type=int,default=1)
    parser.add_argument('--n_neuron_hidden',help='how many neurons in hidden layer if nhidden_layer==1',type=int,default=50)
    parser.add_argument('--estop_mmd_type',help='callback for early stopping. either use val mmd or trans mmd, val or trans respectively',default='val_trans')
    parser.add_argument('--use_tuned_hpms',help='use tuned hyperparameters',default='False')
    parser.add_argument('--plot_synthetic_dist',help='plotting of synthetic data (take extra time), not necessary',default='False')
    parser.add_argument('--precision',help='traainer precision ie 32,16',default='32')
    parser.add_argument('--compile_mmd_mode',help='compile mode for mmd losses',default='reduce-overhead')
    



    args = parser.parse_args()
    
    print(args)
    
    
    
    print('use single si')
    print(args.use_single_si)
    
    args.use_single_si=str_to_bool(args.use_single_si)
    
    print('use single si')
    print(args.use_single_si)
    
    args.use_bernoulli=str_to_bool(args.use_bernoulli)
    args.use_benchmark_generators=str_to_bool(args.use_benchmark_generators)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)
    args.plot_synthetic_dist=str_to_bool(args.plot_synthetic_dist)
    
    
    dict_of_precompiled=return_dict_of_precompiled_mmd(args.compile_mmd_mode)
    

    if args.use_tuned_hpms==True:
        print('args flag for use tuned hparams is TRUE, but hpms not available for CGAN method: running algorithm without any tuned hpms')
    st=time.time()
    
    
    #we have dataloader, now we have to get the generator:
    DETERMINISTIC_FLAG=False
    model_name = 'MMD_GAN'
    current_model_name=model_name
    algo_variant= 'mmd_gan'


    #args.use_single_si=str_to_bool(args.use_single_si)

    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec=pd.read_excel('combined_spec.xls',sheet_name=None)
    #write dataset spec shorthand
    dspec=master_spec['dataset_spec']
    dspec.set_index("d_n",inplace=True) #set this index for easier
    #store index of pandas loc where we find the value
    dspec=dspec.loc[args.d_n] #use this as reerence..
    dspec.d_n= str(args.d_n) if dspec.d_n_type=='str' else int(args.d_n)


    d_n=args.d_n
    n_iterations=args.n_iterations
    dn_log=dspec.dn_log
    #SAVE_FOLDER=dspec.save_folder
    #SAVE_DIR=dspec.save_folder
    lr = float(args.lr)

    #now we want to read in dataset_si
    csi=master_spec['dataset_si'][dspec.d_n].values
    candidate_si=csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si==True: #so we want to use single si, not entire range
        args.optimal_si_list=[args.s_i]

    #gpu_kwargs = get_gpu_kwargs(args)
    if torch.cuda.is_available():
        has_gpu=True
        device_str='cuda'
    else:
        device_str='cpu'
        
        print('warning using cpu for our model, likely slow')

    #if torch.cuda.device_count() > 1:
    #    gpu_kwargs['accelerator'] = "dp"

    #precision_float=


    result_dict = {}
    results_list = []

    for k, si_iter in enumerate(args.optimal_si_list):
        result_dict[si_iter] = 0
        results_list = []

        orig_data = load_data(d_n=args.d_n, s_i=si_iter, dataset_folder=dspec.save_folder)  # load data

        dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns



        ssld = SSLDataModule(orig_data, tot_bsize=args.tot_bsize,lab_bsize=args.lab_bsize,precision=args.precision)#model_init_args['lab_bsize'])
        ssld.setup()  # initialise the data
        # get the data for validation
        val_features = ssld.data_validation[0][0].to(torch.device(device_str)).float()
        val_lab = ssld.data_validation[0][1].to(torch.device(device_str)).float()

        optimal_model = None
        optimal_trainer = None

        # START TIME
        st = time.time()


        # so first we create Y generator
        yvals=orig_data['label_y'][:,1].float()

        y_gen=Generator_Y(d_n,
                          si_iter,
                            dn_log,
                            yvals)

        feature_dim=orig_data['label_features'].shape[1] #columns in features
        all_x=torch.cat((orig_data['label_features'],orig_data['unlabel_features']),0).to(torch.device(device_str))
        median_pwd=get_median_pwd(all_x).item()

        gpu_kwargs={'precision':args.precision}

        #for mis-specified model, we should have Y as root cause
        #and concat all other features into effect...

        st=time.time()
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        #mmdgan_d = MMDDataModule(orig_data,labelled_batch_size=args.lab_bsize,unlabelled_batch_size=args.tot_bsize)

        #m.setup()
        model_init_args = {
            'd_n': args.d_n,
            's_i': si_iter,
            'dn_log': dspec.dn_log,
            'input_dim':dspec['input_dim'],
            'lr':args.lr,
            'alpha_ul':1.0,
            'y_l':ssld.orig_data['label_y'],
            'median_pwd':median_pwd,
            'n_class':2,
            'current_model_name': current_model_name,
            'gen_layers': [100],
            'device_str':device_str,
            'precision':float(args.precision),
            'tot_bsize':args.tot_bsize,
            'num_unlabelled':all_x.shape[0],
        }
        


        
        if args.estop_mmd_type == 'val_trans':
            estop_cb = return_early_stop_min_val_trans_mmd(patience=args.estop_patience)
            min_mmd_checkpoint_callback = return_chkpt_min_val_trans_mmd(current_model_name,
                                                                        dspec.save_folder)  # returns max checkpoint


        #min_mmd_checkpoint_callback = return_early_stop_min_val_trans_mmd(model_name, dspec.save_folder)  # returns max checkpoint
        #estop_cb = return_early_stop_cb()

        #max_pf_checkpoint_callback=return_chkpt_max_acc(model_name,dspec.save_folder) #returns max checkpoint

        
        
        mmd_gan=MMDGAN(**model_init_args)  
        
        #callbacks=[]


        callbacks=[min_mmd_checkpoint_callback,estop_cb]

        tb_logger = create_logger(model_name,d_n,si_iter)
        
        profiler=None
        
        trainer = create_trainer(tb_logger, callbacks, gpu_kwargs,max_epochs=args.n_iterations,profiler=profiler)

        delete_old_saved_models(model_name,dspec.save_folder,si_iter)

        mmd_gan.set_precompiled(dict_of_precompiled)


        trainer.fit(model=mmd_gan, datamodule=ssld)


        mod_names = return_saved_model_name(model_name, dspec.save_folder, dspec.dn_log, si_iter)

        if len(mod_names) > 1:
            print(mod_names)
            print('error duplicate model names')
            assert 1 == 0
        elif len(mod_names)==1:
            #set_trace()
            
            mmd_gan = type(mmd_gan).load_from_checkpoint(checkpoint_path=mod_names[0])
        else:
            assert 1 == 0


        #set_trace()








        #tuner=pl.tuner.Tuner(trainer)
        
        #tuner.lr_find(gen_x)
        
        #import torch
        #import torchvision.models as models
        #from torch.profiler import profile, record_function, ProfilerActivity
        
        #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #with record_function("model_inference"):
                #model(inputs)

        #trainer.fit(gen_x,tloader) #train here


        # TENSORBOARD LOGGER
        #tb_logger = get_default_logger(current_model_name, args.d_n, si_iter, t=0)



        #set_trace()
        # TRAINER
       # trainer = get_default_trainer(args, tb_logger, callbacks, DETERMINISTIC_FLAG, min_epochs=10, **gpu_kwargs)



        #with trainer.init_module():
            # models created here will be on GPU and in float16
            # CREATE MODEL
            #current_model = SGANClassifier(**model_init_args)  # define model
        
        # INITIALISE WEIGHTS
        #mmd_gan.apply(init_weights_he_kaiming)  # re init model and weights






        # DELETE OLD SAVED MODELS
        #clear_saved_models(model_name=current_model_name, save_dir=dspec.save_folder, s_i=si_iter)

        # TRAIN








        et=time.time()

        print('time taken: {0} minutes'.format((et-st)/60.))
        
        
        
        #generating synthetic data
        
        
        
        #set_trace()
                    
        n_samples=int(30000*min(dspec.n_unlabelled/1000,5)) #try to set this to deal wtih very large unalbeleld size...

        #now we want to synthesise...
        #n_samples=10000
        y_synthetic=y_gen(n_samples)
        noise=torch.randn(n_samples,feature_dim)
        #now cat them
        gen_input=torch.cat((noise,y_synthetic),1).to(torch.device(device_str))
        x_synthetic=mmd_gan(gen_input)
        #ok now we want to save these
        pkl_dn='{0}/d_n_{1}_s_i_{2}_*.pickle'.format(dspec.save_folder,args.d_n,args.s_i)
        pkl_candidate=glob.glob(pkl_dn)


        #load dataset pkl class to get feature variable names etc
        if len(pkl_candidate)==1:
            pkl_n = pkl_candidate[0]
            with open(pkl_n, 'rb') as f:
                data_pkl = pickle.load(f)
                
                
                
                

        #loading for legacy dataset
        elif len(pkl_candidate)==0:
            dsc_loader = eval(dspec.dataloader_function)  # within the spec
            dsc = dsc_loader(args, s_i, dspec)
            dsc = manipulate_dsc(dsc, dspec)  # adding extra label column and convenient features for complex data mod later on
            data_pkl=dsc
        elif len(pkl_candidate)>1:
            print('error multiple candidates for dataset')
            assert(1==0)
            
        #set_trace()
        
        data_pkl.feature_dim = dspec.feature_dim
        #create names
        data_pkl.all_varnames=data_pkl.merge_dat.drop(columns=['type']).columns.values.tolist()
        data_pkl.feature_varnames=[x for x in data_pkl.all_varnames if x!=data_pkl.class_varname]
        #rename synthetic data
        newlabs=[]
        featlabs=data_pkl.feature_varnames
        feat_df=pd.DataFrame(x_synthetic.cpu().detach().numpy())
        feat_df.columns=featlabs
        ysynth_val=y_synthetic[:,1] #argmax of one-hot vector
        ysynth_val=ysynth_val.cpu().detach().numpy().reshape((-1,1))
        lab_df=pd.DataFrame(ysynth_val)
        lab_df.columns=[data_pkl.class_varname]
        joined_synthetic=pd.concat((feat_df,lab_df),axis=1)


        joined_synthetic_data=joined_synthetic[data_pkl.all_varnames]


        
        balance_labels=True
        exact=False
        dsc=data_pkl
        dsc.label_var=dsc.class_varname
        
        s_i=si_iter
        
        if balance_labels==True and exact==False:
            #continue
        
            #get proportion in labelled....
            
            


            #make folder if not exist
            target_path='{0}/synthetic_data_mmd_gan/'.format(dspec.save_folder)
            dir_exists=os.path.exists(target_path)
            if not dir_exists:
                os.mkdir(target_path)

            
            
            algo_spec=master_spec['algo_spec'].copy()
            this_algo_spec=algo_spec.set_index('algo_variant')#,inplace=True)
            synthetic_data_dir=this_algo_spec.loc[algo_variant].synthetic_data_dir
    #        joined_synthetic_data.to_csv("{0}/{3}/synthetic_data_d_n_{1}_s_i_{2}.csv".format(dspec.save_folder,d_n,s_i,synthetic_data_dir))


            
            print('balancing labels in same proportion as exist in training dataset. ie: not necessarily exact')

            orig_lidx=np.where(np.array(dsc.variable_types)=='label')[0][0]

            orig_lname=dsc.class_varname
            
            
            labelled_partition=dsc.merge_dat[dsc.merge_dat.type=='labelled']

            #orig_labels=dsc.merge_dat[orig_lname]
            
            orig_labels=labelled_partition[orig_lname]
            orig_p0=sum(orig_labels==0)/labelled_partition.shape[0]
            orig_p1=sum(orig_labels==1)/labelled_partition.shape[0]

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
            joined_synthetic_data['type'] = 'synthetic'
            
            
            dsc.label_names_alphan={}

            for l in dsc.labels:
                dsc.label_names_alphan[l]=[l]
            dsc.label_names = [[d] for d in dsc.labels]

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

            
            
            #n_samples=int(30000*min(dspec.n_unlabelled/1000,5)) #try to set this to deal wtih very large unalbeleld size...
            ##n_samples = dsc.merge_dat.shape[0]
            #synthetic_samples_dict=generate_samples_to_dict(dsc,has_gpu,dsc_generators,device_string,n_samples)
            #joined_synthetic_data=samples_dict_to_df(dsc,synthetic_samples_dict,balance_labels=True,exact=False)



            #synthetic_data_dir = algo_spec.loc[algo_variant].synthetic_data_dir
            save_synthetic_data(joined_synthetic_data,d_n,s_i,master_spec,dspec,this_algo_spec,synthetic_data_dir)

            et=time.time()
            total_time=et-st
            total_time/=60


            print(f'total time taken for n_iterations: {n_iterations}\t {total_time:.4f} minutes')
            
            
            rem_runs=100-s_i
            eta=rem_runs*total_time
            
            
            print(f'estimated time remain:\t{eta:.4f}\tmin')
            
            # dict of hyperparameters - parameters to be written as text within synthetic data plot later on
            
            if dsc.feature_dim==1:
                print('feature dim==1 so we are going to to attempt to plot synthetic v real data')
                hpms_dict = {'lr': args.lr,
                            'n_iterations': args.n_iterations,
                            'lab_bs': args.lab_bsize,
                            'ulab_bs': args.tot_bsize,
                            'nhidden_layer': args.nhidden_layer,
                            'neuron_in_hidden': args.n_neuron_hidden,
                            'use_bernoulli': args.use_bernoulli,
                            'time':total_time}
                plot_synthetic_and_real_data(hpms_dict,dsc,args,s_i,joined_synthetic_data,synthetic_data_dir,dspec)
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
                
                
                if 'feature_varnames' not in dir(dsc):
                    dsc.feature_varnames=dsc.feature_names
                    
                    
                dsc.s_i=s_i
                
                if args.plot_synthetic_dist or s_i in [0,1,2,3,4]: #plot first 5 by default
                    plot_2d_data_w_dag(dsc, s_i,synthetic_data_dir=synthetic_data_dir)

                    print('data plotted')
                    
                else:
                    print('data not plotted')
                
                
                #import signal
                    
                #exit without waiting for sync
                # https://stackoverflow.com/questions/905189/why-does-sys-exit-not-exit-when-called-inside-a-thread-in-python


                #import os
                #os._exit(0)
                #os.kill(os.getpid(), signal.SIGINT)
                    
                    
            print('-------------------------')
            print('run finished')
            print('-------------------------')
            
            #sys.exit()
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        
        else:
            assert False,'not implemented for balance labels=True and eaxct=False...'


