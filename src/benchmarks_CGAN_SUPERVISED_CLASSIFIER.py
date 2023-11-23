from benchmarks_utils import *

import sys
import argparse
from typing import Optional
import time
import pickle
import igraph

n_classes = 2
torch.set_float32_matmul_precision('medium') #try with 4090

# https://github.com/PyTorchLightning/pytorch-lightning/issues/10182
# turning off the warnings:
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

from pytorch_lightning import Trainer, seed_everything
RANDOM_SEED=999
seed_everything(RANDOM_SEED, workers=True)

#dataset placeholder
class ds_ph:
    def __init__(self):
        print('new class created')

def str_to_bool(in_str):
    if in_str=='True':
        return(True)
    if in_str=='False':
        return(False)
    else:
        assert(1==0)
        return(None)

#early stopping callback
def return_early_stop_train_loss(patience=100):
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=patience, verbose=False, mode="min")
    return(early_stop_callback)

# REAL datasets
class ds:
    def __init__(self, adj_mat, labels):
        self.adj_mat = adj_mat
        self.labels = labels
        self.dag = igraph.Graph.Adjacency(self.adj_mat)

has_gpu = torch.cuda.is_available()
#gpu_kwargs = {'gpus': torch.cuda.device_count()} if has_gpu else {}

gpu_kwargs = {}

class CGANSupervisedClassifier(pl.LightningModule):
    def __init__(self, lr, d_n, s_i, input_dim,dn_log):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = get_standard_net(input_dim=input_dim,
                                           output_dim=n_classes)
        self.lfunc = torch.nn.CrossEntropyLoss()
        self.val_accs=[]

    def forward(self, x):
        classification = self.classifier(x)
        return classification

    def training_step(self, batch, batch_idx):
        x_l, y_l = batch

        pred_y = get_softmax(self.classifier(x_l))

        loss = self.lfunc(pred_y, y_l.flatten())

        self.log('train_loss', loss)
        return (loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams['lr'])
        return optimizer

    def validation_step(self, batch, batch_idx):
        val_feat = batch[0].squeeze(0)
        val_y = batch[1].squeeze(0)
        y_hat = self.classifier(val_feat)
        v_acc = get_accuracy(y_hat, val_y.flatten())

        self.log("val_acc", v_acc)
        self.log("d_n", self.hparams['dn_log'])
        self.log("s_i", s_i)
        self.val_accs.append(v_acc)


    def predict_test(self,features,label):
        prediction=self.classifier(features)
        p_acc = get_accuracy(prediction, label)
        return(p_acc)

class SSLDataModule(pl.LightningDataModule):
    def __init__(self, orig_data, batch_size: int = 64):
        super().__init__()
        self.orig_data = orig_data
        self.batch_size = batch_size

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
        y_train_ulab = torch.argmax(orig_data['unlabel_y'], 1)

        # -------------#
        # Validation
        # -------------#

        X_val = orig_data['val_features']
        y_val = torch.argmax(orig_data['val_y'], 1)

        # -------------#
        # Setting up resampling
        # -------------#

        n_unlabelled = X_train_ulab.shape[0]
        n_labelled = X_train_lab.shape[0]
        dummy_label_weights = torch.ones(n_labelled)
        resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_unlabelled, replacement=True)
        X_train_lab_rs = X_train_lab[resampled_i]
        y_train_lab_rs = y_train_lab[resampled_i]
        # ulab_mix is the data train!
        self.data_train = torch.utils.data.TensorDataset(X_train_lab_rs,
                                                         y_train_lab_rs,
                                                         X_train_ulab)
        vfeat = X_val.unsqueeze(0)
        vlab = y_val.unsqueeze(0)
        self.data_validation = torch.utils.data.TensorDataset(vfeat, vlab)
        self.nval = vlab.shape[0]

        return (self)

    def train_dataloader(self):
        if has_gpu:
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True,num_workers=8)
        else:
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):

        if has_gpu:
            return DataLoader(self.data_validation, batch_size=self.nval, pin_memory=True,num_workers=8)
        else:
            return DataLoader(self.data_validation, batch_size=self.nval)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_n', help='dataset number 1-5 OR MOG',type=str)
    parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ',type=int)
    parser.add_argument('--n_iterations', help='how many i`terations to train classifier for',type=int,default=100)
    parser.add_argument('--nsamps', help='how many artifical samples to include',type=int,default=300)
    parser.add_argument('--lr', help='learning rate ie 1e-2, 1e-3,...',type=float,default=1e-3)
    parser.add_argument('--algo_variant', help='which algo used to train generators: disjoint,gumbel,...',type=str)
    parser.add_argument('--use_single_si',help='want to use single si or entire range',type=str,default='True')
    parser.add_argument('--n_trials',help='number trials for each si',type=int,default=1)
    parser.add_argument('--tot_bsize', help='unlabelled + labelled batch size for training', type=int,default=128)
    parser.add_argument('--lab_bsize', help='labelled data batch size for training', type=int,default=4)
    parser.add_argument('--estop_patience', help='labelled data batch size for training', type=int,default=5)

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
    algospec=master_spec['algo_spec']
    algospec.set_index("algo_variant",inplace=True) #set this index for easier
    synthetic_data_dir=algospec.loc[args.algo_variant,'synthetic_data_dir']
    model_name=algospec.loc[args.algo_variant,'model_save_name']

    #now we want to read in dataset_si
    csi=master_spec['dataset_si'][dspec.d_n].values
    candidate_si=csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si==True: #so we want to use single si, not entire range
        #args.optimal_si_list=[args.optimal_si_list[args.s_i]]
        args.optimal_si_list = [args.s_i]


    st = time.time()

    origs=[]
    wsynth=[]

    for k, s_i in enumerate(args.optimal_si_list):
        print('doing s_i idx of 100: {0}'.format(k))
        print(f'doing s_i: {s_i}')
        target_fn = '{0}/{3}/synthetic_data_d_n_{1}_s_i_{2}.csv'.format(dspec.save_folder, args.d_n, s_i,synthetic_data_dir)
        synthetic_data = pd.read_csv(target_fn, index_col=0)
        synthetic_data = synthetic_data.fillna(0)

        # 2. read in pkl

        pkl_dn='{0}/d_n_{1}_s_i_{2}_*.pickle'.format(dspec.save_folder,args.d_n,s_i)

        pkl_candidate=glob.glob(pkl_dn)

        if len(pkl_candidate)==0:
            #create it
            # 2.1 if pkl no exist then create it

            #get all 8 tensor names
            dtypes=['val_features',
                    'val_y',
                    'label_features',
                    'label_y',
                    'unlabel_features',
                    'unlabel_y',
                    'test_features',
                    'test_y']
            get_ddir = lambda in_str : '{0}/d_n_{1}_s_i_{2}_{3}.pt'.format(dspec.save_folder,args.d_n,s_i,in_str)
            #get all names
            pt_names ={d:get_ddir(d) for d in dtypes}
            #read in as pt tensor
            pt_tensors = {d:torch.load(pt_names[d]) for d in pt_names.keys()}
            #convert to npy
            np_tensors = {d:pt_tensors[d].cpu().detach().numpy() for d in pt_tensors.keys()}
            #convert to pandas df
            df_tensors = {d:pd.DataFrame(np_tensors[d]) for d in np_tensors.keys()}
            #subset Y
            df_tensors['val_y']=df_tensors['val_y'][[1]]
            df_tensors['label_y'] = df_tensors['label_y'][[1]]
            df_tensors['unlabel_y'] = df_tensors['unlabel_y'][[1]]
            df_tensors['test_y'] = df_tensors['test_y'][[1]]
            dt=df_tensors
            dt['val']=pd.concat([dt['val_features'],dt['val_y']],axis=1)
            dt['label'] = pd.concat([dt['label_features'], dt['label_y']], axis=1)
            dt['unlabel']= pd.concat([dt['unlabel_features'], dt['unlabel_y']], axis=1)
            dt['test'] = pd.concat([dt['test_features'], dt['test_y']], axis=1)
            dataset_names=dspec.feature_names.split(',') + ['Y']
            dt['val'].columns=dataset_names
            dt['label'].columns=dataset_names
            dt['unlabel'].columns=dataset_names
            dt['test'].columns = dataset_names
            dt['val']['type']='validation'
            dt['label']['type'] = 'labelled'
            dt['unlabel']['type'] = 'unlabelled'
            dt['test']['type'] = 'test'
            print('pausing here')
            merge_dat=pd.concat([dt['label'],dt['unlabel'],dt['val'],dt['test']],axis=0,ignore_index=True)
            #create class from this one

            data_pkl=ds_ph()
            data_pkl.merge_dat=merge_dat
            data_pkl.class_varname='Y'

        elif len(pkl_candidate)==1:
            #load it
            pkl_n=pkl_candidate[0]
            with open(pkl_n, 'rb') as f:
                data_pkl = pickle.load(f)


        else:
            #we have multiple possibilities for data class
            assert(1==0)


        pkl_df = data_pkl.merge_dat

        # 3. match columns
        c_name = data_pkl.class_varname
        dcols = [c for c in data_pkl.merge_dat.drop(columns=['type']).columns]
        is_lab = [c_name == d for d in dcols]
        lab_transform = lambda lab: 'label' if lab else 'feature'
        feat_or_lab = [lab_transform(c) for c in is_lab]

        data_pkl.variable_types = feat_or_lab

        # now get dict of each vtype being either label or feature
        var_ddict = {}
        feature_variables = [dcols[idx] for idx in np.where([f == 'feature' for f in feat_or_lab])[0]]
        feature_labels = [dcols[idx] for idx in np.where([f == 'label' for f in feat_or_lab])[0]]

        lab_dat = pkl_df[pkl_df['type'] == 'labelled']
        ulab_dat = pkl_df[pkl_df['type'] == 'unlabelled']
        val_dat = pkl_df[pkl_df['type'] == 'validation']
        test_dat = pkl_df[pkl_df['type'] == 'test']
        orig_data = {}

        orig_data['label_features'] = torch.Tensor(lab_dat[feature_variables].to_numpy(dtype=np.float32))
        orig_data['label_y'] = torch.Tensor(lab_dat[feature_labels].to_numpy(dtype=np.float32))
        orig_data['unlabel_features'] = torch.Tensor(ulab_dat[feature_variables].to_numpy(dtype=np.float32))
        orig_data['unlabel_y'] = torch.Tensor(ulab_dat[feature_labels].to_numpy(dtype=np.float32))
        orig_data['val_features'] = torch.Tensor(val_dat[feature_variables].to_numpy(dtype=np.float32))
        orig_data['val_y'] = torch.Tensor(val_dat[feature_labels].to_numpy(dtype=np.float32))
        orig_data['test_features'] = torch.Tensor(test_dat[feature_variables].to_numpy(dtype=np.float32))
        orig_data['test_y'] = torch.Tensor(test_dat[feature_labels].to_numpy(dtype=np.float32))


        synthcols=[c for c in synthetic_data.columns]
        rep_cols=[]
        REPLACING_X_0=False
        for c in synthcols:
            if c[0]=='X' and REPLACING_X_0:
                retval=c.replace('_0','')
            elif c[0]=='Y':
                retval=c.replace('_0','')
            else:
                retval=c
            rep_cols.append(retval)

        synthetic_data.columns=rep_cols
        input_dim = orig_data['label_features'].shape[1]  # columns in features





        results_list = []
        for t in range(args.n_trials):
            cgan_classifier = CGANSupervisedClassifier(lr=args.lr, d_n=args.d_n, s_i=s_i,
                                                    input_dim=dspec.input_dim,dn_log=dspec.dn_log)  # define model

            cgan_classifier.apply(init_weights_he_kaiming)

            max_pf_checkpoint_callback = return_chkpt_max_val_acc(model_name, dspec.save_folder)  # returns max checkpoint
            estop_cb = return_estop_val_acc(args.estop_patience)
            tb_logger = pl_loggers.TensorBoardLogger("lightning_logs/",
                                                    name=combined_name(model_name, args.d_n, s_i),
                                                    version=0)
            #import pytorch_lightning
            #profiler_simple_first=pytorch_lightning.profilers.SimpleProfiler(dirpath='/media/krillman/240GB_DATA/codes2/SSL_GCM/profiler',filename='simple_profile_results.txt')
            #profiler_simple_second=pytorch_lightning.profilers.SimpleProfiler(dirpath='/media/krillman/240GB_DATA/codes2/SSL_GCM/profiler',filename='simple_profile_results_second.txt')

            trainer = pl.Trainer(log_every_n_steps=1,
                                check_val_every_n_epoch=1,
                                max_epochs=args.n_iterations,
                                callbacks=[max_pf_checkpoint_callback,estop_cb],
                                reload_dataloaders_every_n_epochs=1,
                                deterministic=True,
                                logger=tb_logger,
                                accelerator='gpu',
                                devices=1,
                                #profiler=profiler_simple_first,
                                **gpu_kwargs)


            # now load in synthetic data
            n_zeros=synthetic_data[synthetic_data[feature_labels[0]]==0].shape[0]
            n_ones=synthetic_data[synthetic_data[feature_labels[0]]==1].shape[0]
            if n_zeros==0 or n_ones==1:
                print('pausing zero case of 0 1')
            if args.nsamps>n_zeros and args.nsamps>n_ones:
                args.nsamps=min(n_zeros,n_ones)
                synth_c0 = synthetic_data[synthetic_data[feature_labels[0]] == 0].sample(args.nsamps)
                synth_c1 = synthetic_data[synthetic_data[feature_labels[0]] == 1].sample(args.nsamps)

                synthetic_data = pd.concat((synth_c0, synth_c1), 0, ignore_index=True)
            elif args.nsamps==-1: # -1 flag is for when you use ALL of the data
                synthetic_data=synthetic_data # we just do this instead


            synth_dd = {}

            if dspec.feature_dim==1:
                #rename...
                scols=[c for c in synthetic_data.columns]
                ncols=[]
                for s in scols:
                    n_name=s
                    if s[0]=='X' and '_' not in s:
                        n_name=s+'_0'
                    ncols.append(n_name)
                synthetic_data.columns=ncols

            #select feature / label vars respectively. this will also reorder the data.
            synth_dd['synthetic_features'] = torch.Tensor(synthetic_data[feature_variables].to_numpy(dtype=np.float32))
            synth_dd['synthetic_y'] = torch.Tensor(synthetic_data[feature_labels[0]].to_numpy(dtype=np.float32))

            ssld_orig = CGANSupervisedDataModule(orig_data,
                                                synth_dd,
                                                inclusions='orig_only',
                                                n_to_sample_for_orig='unlabelled', #unlabelled, labelled, baseline. baseline=2000 as per original synthetic data. unlabelled=as many as in this unlabelled data. so like 100,000 if so. labelled=just labelled cases ~ 40 cases.
                                                batch_size=args.tot_bsize) #actually just do tot_bsize cos same as orig, otherwise too slow 23_11_2023 AM
                                                
                                                #batch_size=32) #32 to keep same as orig for small datasets, 23_11_2023 AM
                                                #batch_size=args.tot_bsize)

            # before we start training, we have to delete old saved models
            clear_saved_models(model_name, save_dir=dspec.save_folder, s_i=s_i)

            from IPython.core.debugger import set_trace


            trainer.fit(cgan_classifier, ssld_orig)  # train here

            #set_trace()

            #load optimal model....
            cgan_fn = '{0}/saved_models/{2}-s_i={1}-*.ckpt'.format(dspec.save_folder, s_i,model_name)
            optimal_model_path=glob.glob(cgan_fn)[0]
            #set_trace()
            cgan_classifier=type(cgan_classifier).load_from_checkpoint(optimal_model_path)

            cgan_saved = glob.glob(cgan_fn)[0]
            cgan_saved = cgan_saved.split('.ckpt')[0][-4:]

            optimal_pre_synthetic=cgan_saved

            max_pf_checkpoint_callback = return_chkpt_max_val_acc(model_name, dspec.save_folder)  # returns max checkpoint
            estop_cb = return_estop_val_acc(args.estop_patience)


            trainer = pl.Trainer(log_every_n_steps=1,
                                check_val_every_n_epoch=1,
                                max_epochs=args.n_iterations,
                                callbacks=[max_pf_checkpoint_callback,estop_cb],
                                reload_dataloaders_every_n_epochs=1,
                                logger=tb_logger,
                                deterministic=True,
                                #profiler=profiler_simple_second,
                                **gpu_kwargs)

            ssld_synth = CGANSupervisedDataModule(orig_data,
                                                synth_dd,
                                                inclusions='orig_and_synthetic',
                                                batch_size=args.tot_bsize)
            trainer.fit(cgan_classifier, ssld_synth)  # train here

            current_model=cgan_classifier

            et = time.time()

            max_va = max(current_model.val_accs)
            results_list.append(max_va)

            model_to_search_for = dspec.save_folder + '/saved_models/' + model_name + "*-s_i={0}-epoch*".format(s_i)
            #set_trace()
            candidate_models = glob.glob(model_to_search_for)

            current_model = type(current_model).load_from_checkpoint(checkpoint_path=candidate_models[0])
            # store optimal model

            # get the data for validation
            #val_features = ssld_orig.data_validation[0][0]
            val_features = torch.tensor(ssld_orig.data_validation[0][0],device='cuda')
            #val_lab = ssld_orig.data_validation[0][1].flatten()

            val_lab = torch.tensor(ssld_orig.data_validation[0][1].flatten(),device='cuda')
            

            try:
                print('pausing here')
                with torch.no_grad():
                    optimal_pred = cgan_classifier.forward(val_features)#.cuda())
                optimal_acc = get_accuracy(optimal_pred, val_lab)#.cuda())
                with torch.no_grad():
                    current_pred = current_model.forward(val_features)#.cuda())
                current_acc = get_accuracy(current_pred, val_lab)

                if current_acc > optimal_acc:
                    optimal_model = copy.deepcopy(current_model)
                    optimal_trainer = copy.deepcopy(trainer)
                    print('optimal model overwritten')
                    print('old optimal: {0}'.format(optimal_acc))
                    print('new optimal: {0}'.format(current_acc))
                else:
                    print('optimal model NOT overwritten')
                    print('old optimal: {0}'.format(optimal_acc))
                    print('new optimal: {0}'.format(current_acc))
                del trainer
            except:
                optimal_model = copy.deepcopy(current_model)
                optimal_trainer = copy.deepcopy(trainer)
                print('optimal model created')
                del trainer

            et = time.time()

            print('time taken: {0} minutes'.format((et - st) / 60.))


            cgan_saved=glob.glob(cgan_fn)[0]
            cgan_saved=cgan_saved.split('.ckpt')[0][-4:]

        # delete old saved models cos we are storing the optimal
        SAVE_FOLDER=dspec.save_folder
        clear_saved_models(model_name, save_dir=SAVE_FOLDER, s_i=s_i)
        filepath = "{0}/saved_models/{1}-s_i={2}-epoch={3}-val_acc={4}.ckpt".format(SAVE_FOLDER,
                                                                                    model_name,
                                                                                    optimal_trainer.model.hparams[
                                                                                        's_i'],
                                                                                    10,
                                                                                    max(optimal_trainer.model.val_accs))
        optimal_trainer.save_checkpoint(filepath)
        
        
        test_features=torch.tensor(orig_data['test_features'],device='cuda')
        test_y=torch.tensor(orig_data['test_y'].flatten(),device='cuda')
        ulab_features=torch.tensor(orig_data['unlabel_features'],device='cuda')
        ulab_y=torch.tensor(orig_data['unlabel_y'].flatten(),device='cuda')

        with torch.no_grad():
            test_acc = optimal_model.predict_test(features=test_features,label=test_y)
            unlabel_acc = optimal_model.predict_test(features=ulab_features,label=ulab_y)

        #test_acc = optimal_model.predict_test(orig_data['test_features'].cuda(), orig_data['test_y'].flatten().cuda())
        #unlabel_acc = optimal_model.predict_test(orig_data['unlabel_features'].cuda(), orig_data['unlabel_y'].flatten().cuda())

        test_acc = np.array([test_acc.cpu().detach().item()])
        filepath = "{0}/saved_models/{1}-s_i={2}_test_acc.out".format(SAVE_FOLDER, model_name,
                                                                    optimal_trainer.model.hparams[
                                                                        's_i'])
        np.savetxt(filepath, test_acc)

        unlabel_acc = np.array([unlabel_acc.cpu().detach().item()])
        filepath = "{0}/saved_models/{1}-s_i={2}_unlabel_acc.out".format(SAVE_FOLDER, model_name,
                                                                        optimal_trainer.model.hparams[
                                                                            's_i'])
        np.savetxt(filepath, unlabel_acc)

        print(f'test_acc: {test_acc}')
        print(f'unlabel_acc: {unlabel_acc}')

        print('pausing here')
        print('plotting decision boundaries (plotly)')


        optimal_model.model_name=model_name

        # PLOT HARD DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, s_i, args, optimal_model, hard=True, output_html=False)

        # PLOT SOFT (CONTINUOUS) DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, s_i, args, optimal_model, hard=False, output_html=False)

        # DELETE OPTIMALS SO CAN RESTART IF DOING MULTIPLE S_I
        del optimal_trainer
        del optimal_model
