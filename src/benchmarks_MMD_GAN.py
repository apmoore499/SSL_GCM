

from benchmarks_utils import *
import argparse
import time

n_classes=2


get_softmax = torch.nn.Softmax(dim=0)
import sys
sys.path.append('generative_models')
sys.path.append('py')
sys.path.append('py/generative_models/')
NUM_WORKERS=0
import pytorch_lightning as pl
from typing import Optional
from benchmarks_cgan import *
from Generator_Y import *

n_classes=2

class MMDGAN(pl.LightningModule):
    def __init__(self,
                 lr,
                 d_n,
                 s_i,
                 dn_log,
                 input_dim,
                 output_dim,
                 median_pwd,
                 label_batch_size=4,
                 unlabel_batch_size=256):
        super().__init__()

        self.save_hyperparameters()

        self.gen=get_standard_net(input_dim,output_dim)

     
        self.vmmd_losses=[]

    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x

    def training_step(self, batch, batch_idx,optimizer_idx):
        

        labelled=batch['loader_labelled']
        unlabelled=batch['loader_unlabelled']

        sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]

        if optimizer_idx==0: #labelled loader
            x,y=labelled
            #sample noise
            x=x.reshape((-1,self.hparams['output_dim']))
            z=torch.randn_like(x)
            
            #y=torch.nn.functional.one_hot(y)
            
            #cat input...
            gen_input=torch.cat((z,y),1).float()
            #prediction
            x_hat=self.gen(gen_input)
            y=y.float()
            loss=mix_rbf_mmd2_joint(x_hat,x,y,y,sigma_list=sigma_list)
            self.log('labelled_mmd_loss', loss)
            return(loss)

        if optimizer_idx==1:
            #set_trace()
            x,y=unlabelled
            x=x.reshape((-1,self.hparams['output_dim']))
            z=torch.randn_like(x)
            #y=torch.nn.functional.onehot(y)
            gen_input=torch.cat((z,y),1).float()
            #prediction
            x_hat=self.gen(gen_input)
            loss=mix_rbf_mmd2(x_hat,x,sigma_list=sigma_list)
            self.log('unlabelled_mmd_loss', loss)
            return(loss)


    def configure_optimizers(self):
        self.g_optim_one = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        self.g_optim_two = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        return self.g_optim_one, self.g_optim_two

    def validation_step(self, batch, batch_idx):
        
        sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]
        
        #set_trace()
        val_feat=batch[0].squeeze(0).reshape((-1,self.hparams['output_dim']))
        val_y=batch[1].squeeze(0)
        trans_feat=batch[2].squeeze(0).reshape((-1,self.hparams['output_dim']))
        trans_y=batch[3].squeeze(0)

        
        val_y_oh=val_y.float()
        trans_y_oh=trans_y.float()
        
        #joint mmd on validation data
        #x,y=batch #entire batch 
        noise=torch.randn_like(val_feat) #sample noise
        gen_input=torch.cat((noise,val_y_oh),1).float() #concatentate noise with label info
        x_hat=self.gen(gen_input) #generate x samples random
        #get rbf mmd2 joint
        val_mmd_loss=mix_rbf_mmd2_joint(x_hat,val_feat,val_y_oh,val_y_oh,sigma_list=sigma_list)
        
        #joint mmd transduction
        
        #x,y=batch #entire batch 
        noise=torch.randn_like(trans_feat)  #sample noise
        gen_input=torch.cat((noise,trans_y_oh),1).float() #concatentate noise with label info
        x_hat=self.gen(gen_input) #generate x samples random
        #get rbf mmd2 joint
        trans_mmd_loss=mix_rbf_mmd2_joint(x_hat,trans_feat,trans_y_oh,trans_y_oh,sigma_list=sigma_list)      

        self.log("val_mmd", val_mmd_loss)
        self.log("trans_mmd",trans_mmd_loss)
        
        self.vmmd_losses.append(val_mmd_loss.detach().item())
        
        #get min one..
        
        #print(self.vmmd_losses)
        

        self.log("hp_metric", min(self.vmmd_losses))
        #set_trace()
        print('val mmd loss: {0}'.format(val_mmd_loss))
        print('t mmd loss: {0}'.format(trans_mmd_loss))
        
        self.log("s_i",self.hparams.s_i)


        self.log("d_n",self.hparams.dn_log)
        return(self)

class MMDDataModule(pl.LightningDataModule):
    def __init__(self, orig_data,
                 labelled_batch_size: int = 4,
                 unlabelled_batch_size: int = 128):
        super().__init__()
        self.orig_data = orig_data
        self.labelled_batch_size = labelled_batch_size
        self.unlabelled_batch_size = unlabelled_batch_size

    def setup(self, stage: Optional[str] = None):
        orig_data = self.orig_data
        # ----------#
        # Training Labelled
        # ----------#
        X_train_lab = orig_data['label_features']
        y_train_lab = torch.argmax(orig_data['label_y'], 1)

        # ----------#
        # Training Unlabelled
        # ----------#
        X_train_ulab = orig_data['unlabel_features']
        y_train_ulab = orig_data['unlabel_y']

        # Validation Sets

        # -------------#
        # Validation
        # -------------#

        X_val = orig_data['val_features']
        y_val = orig_data['val_y']

        # -------------#
        # Setting up resampling
        # -------------#

        n_unlabelled = X_train_ulab.shape[0]

        # sample with replacement indices of X_train_lab

        # resampled_i=np.random.choice(label_i,n_unlabelled)

        # use torch.multinomial... to get resampled_i

        n_labelled = X_train_lab.shape[0]
        n_unlabelled = X_train_ulab.shape[0]

        # dummy tensor of l;abelled representyingh probabiliity weights, just use tensor.ones

        dummy_label_weights = torch.ones(n_labelled)

        resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_unlabelled+n_labelled, replacement=True)

        y_train_lab_rs = orig_data['label_y'][resampled_i]


        self.data_train_labelled=torch.utils.data.TensorDataset(X_train_lab,
                                                           orig_data['label_y'])


        self.data_train_unlabelled=torch.utils.data.TensorDataset(torch.cat((X_train_lab,X_train_ulab),0),
                                                             y_train_lab_rs)

        vfeat = X_val.unsqueeze(0)
        vlab = y_val.unsqueeze(0)
        tfeat = X_train_ulab.unsqueeze(0)
        tlab = y_train_ulab.unsqueeze(0)

        self.data_validation = torch.utils.data.TensorDataset(tfeat, tlab, vfeat, vlab)
        self.nval = vlab.shape[0]

    def train_dataloader(self):
        labelled_loader=torch.utils.data.DataLoader(self.data_train_labelled,batch_size=self.labelled_batch_size,num_workers=NUM_WORKERS,shuffle=True)
        unlabelled_loader=torch.utils.data.DataLoader(self.data_train_unlabelled,batch_size=self.unlabelled_batch_size,num_workers=NUM_WORKERS,shuffle=True)


        loaders = {"loader_labelled":labelled_loader,
                   "loader_unlabelled":unlabelled_loader}

        return loaders

    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.nval)


#we have dataloader, now we have to get the generator:

model_name = 'MMD_GAN'
algo_variant= 'mmd_gan'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_n', help='dataset number 1-5 OR MOG', type=str)
    parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ', type=int)
    parser.add_argument('--n_iterations', help='how many iterations to train classifier for', type=int)
    parser.add_argument('--lr', help='learning rate ie 1e-2, 1e-3,...', type=float)
    parser.add_argument('--use_single_si', help='do we want to train on collection of si or only single instance',
                        type=str)
    parser.add_argument('--tot_bsize', help='unlabelled + labelled batch size for training', type=int)
    parser.add_argument('--lab_bsize', help='labelled data batch size for training', type=int)
    parser.add_argument('--n_trials', help='how many trials to do', type=int, default=10)
    parser.add_argument('--use_gpu', help='use gpu or not', type=str, default='False')
    parser.add_argument('--estop_patience', help='early stopping patience', type=int, default=10)
    parser.add_argument('--metric', help='which metric to select best model. bce or acc', type=str, default='val_acc')
    parser.add_argument('--min_epochs', help='min epochs to train for', type=int, default=10)
    parser.add_argument('--use_tuned_hpms', help='use tuned hyper params or not', type=str, default='False')

    args = parser.parse_args()
    args.use_single_si=str_to_bool(args.use_single_si)

    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec=pd.read_excel('combined_spec.xls',sheet_name=None)
    #write dataset spec shorthand
    dspec=master_spec['dataset_spec']
    dspec.set_index("d_n",inplace=True) #set this index for easier
    #store index of pandas loc where we find the value
    dspec=dspec.loc[args.d_n] #use this as reerence..
    dspec.d_n= str(args.d_n) if dspec.d_n_type=='str' else int(args.d_n)

    #gpu_kwargs={'gpus':torch.cuda.device_count(),'precision':16} if has_gpu else {}
    #gpu_kwargs={'gpus':torch.cuda.device_count()} if has_gpu else {}


    d_n=args.d_n
    n_iterations=args.n_iterations
    dn_log=dspec.dn_log
    SAVE_FOLDER=dspec.save_folder
    SAVE_DIR=dspec.save_folder
    lr = float(args.lr)

    #now we want to read in dataset_si
    csi=master_spec['dataset_si'][dspec.d_n].values
    candidate_si=csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si==True: #so we want to use single si, not entire range
        args.optimal_si_list=[args.s_i]

    gpu_kwargs = get_gpu_kwargs(args)
    if args.use_gpu and torch.cuda.is_available():
        has_gpu=True

    if torch.cuda.device_count() > 1:
        gpu_kwargs['accelerator'] = "dp"

    results_list=[]
    for k, s_i in enumerate(args.optimal_si_list):
        for t in range(args.n_trials):

            print('doing s_i: {0}'.format(s_i))
            st = time.time()



            orig_data = load_data(d_n=d_n, s_i=s_i, dataset_folder=SAVE_FOLDER)

            #orig data is a ditcionary of 6 types:
            # label_features
            # unlabel_features
            # val_features
            # label_y
            # unlabel_y
            # val_y

            # so first we create Y generator
            yvals=orig_data['label_y'][:,1].float()


            y_gen=Generator_Y(d_n,s_i,
                              dn_log,
                              yvals)

            feature_dim=orig_data['label_features'].shape[1] #columns in features

            #n_classes should == 2

            input_dim=feature_dim+n_classes

            output_dim=feature_dim

            #get median pwd


            all_x=torch.cat((orig_data['label_features'],orig_data['unlabel_features']),0)
            median_pwd=get_median_pwd(all_x)


            #for mis-specified model, we should have Y as root cause
            #and concat all other features into effect...

            st=time.time()
            # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
            mmdgan_d = MMDDataModule(orig_data,labelled_batch_size=args.lab_bsize,unlabelled_batch_size=args.tot_bsize)

            #m.setup()


            mmd_gan=MMDGAN(args.lr,
                           args.d_n,
                           args.s_i,
                            dn_log,
                           input_dim,
                           output_dim,
                           median_pwd)

            min_mmd_checkpoint_callback = return_chkpt_min_mmd(model_name, dspec.save_folder)  # returns max checkpoint
            estop_cb = return_early_stop_cb()

            #max_pf_checkpoint_callback=return_chkpt_max_acc(model_name,SAVE_FOLDER) #returns max checkpoint

            tb_logger = pl_loggers.TensorBoardLogger("lightning_logs/",
                                                     name=combined_name(model_name,d_n,s_i),
                                                       version=0)

            trainer = pl.Trainer(log_every_n_steps=1,
                                 check_val_every_n_epoch=1,
                                 max_epochs=n_iterations,
                                 callbacks=[min_mmd_checkpoint_callback,estop_cb],
                                 reload_dataloaders_every_epoch=True,
                                logger=tb_logger)

            #before we start training, we have to delete old saved models
            delete_old_saved_models(model_name,save_dir=SAVE_FOLDER,s_i=s_i)

            trainer.fit(mmd_gan, mmdgan_d) #train here

            et=time.time()

            print('time taken: {0} minutes'.format((et-st)/60.))

#now we want to synthesise...
n_samples=10000
y_synthetic=y_gen(n_samples)
noise=torch.randn(n_samples,feature_dim)
#now cat them
gen_input=torch.cat((noise,y_synthetic),1)
x_synthetic=mmd_gan(gen_input)
#ok now we want to save these
pkl_dn='{0}/d_n_{1}_s_i_{2}_*.pickle'.format(dspec.save_folder,args.d_n,s_i)
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




#make folder if not exist
target_path='{0}/synthetic_data_mmd_gan/'.format(dspec.save_folder)
dir_exists=os.path.exists(target_path)
if not dir_exists:
    os.mkdir(target_path)

algo_spec=master_spec['algo_spec']
algo_spec.set_index('algo_variant',inplace=True)
synthetic_data_dir=algo_spec.loc[algo_variant].synthetic_data_dir
joined_synthetic_data.to_csv("{0}/{3}/synthetic_data_d_n_{1}_s_i_{2}.csv".format(dspec.save_folder,d_n,s_i,synthetic_data_dir))

