from benchmarks_utils import *
import sys

sys.path.append('./py/generative_models/')
from benchmarks_cgan import *

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from benchmarks_utils import *

import sys
import argparse
from typing import Optional
import time


import argparse

import torch
import torch.distributions as D
from torch.utils.data import DataLoader
from typing import Optional

torch.set_float32_matmul_precision('high') #try with 4090

DETERMINISTIC_FLAG=False


current_model_name='PARTIAL_SUPERVISED_CLASSIFIER'
class PartialSupervisedClassifier(pl.LightningModule):
    def __init__(self, lr, d_n, s_i, input_dim,dn_log,output_dim,tot_bsize=None,best_value=None):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = get_standard_net(input_dim=input_dim, output_dim=output_dim)
        self.val_accs = []

        self.model_name = current_model_name

    def forward(self, x):
        classification = self.classifier(x)
        return classification

    def predict(self, x):  # predicting on numpy array for decision boundary plot
        if type(x) != torch.Tensor:  # turn the numpy array into torch tensor
            x = torch.Tensor(x)
        pred = self.forward(x)  # perform classification using model
        retval = pred.argmax(
            1).cpu().detach().numpy()  # convert classification results to numpy, take argmax of binary classification predictions
        return (retval)  # return the value of self.forward, ie, the classification

    def training_step(self, batch, batch_idx):
        x_l = batch[0]
        
        y_l = batch[1]
        
        #x_u = batch[2]
        
        
        guess_label = self.classifier(x_l)
        loss_gl = torch.nn.functional.cross_entropy(guess_label.float(), y_l.float())
        loss = loss_gl
        self.log('train_loss', loss, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        # print(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams['lr'])
        return optimizer

    def validation_step(self, batch, batch_idx):
        val_feat = batch[0].squeeze(0)
        val_y = batch[1].squeeze(0)
        y_hat = self.classifier(val_feat)
        v_acc = get_accuracy(y_hat, val_y[:,1].flatten())
        loss_val_bce = torch.nn.functional.cross_entropy(y_hat, val_y)
        self.log('val_bce', loss_val_bce, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("val_acc", v_acc, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

        self.log("d_n", self.hparams['dn_log'])
        self.log("s_i", self.hparams['s_i'])
        self.val_accs.append(v_acc.item())




    def predict_test(self, features, label):
        prediction = self.classifier(features)
        p_acc = get_accuracy(prediction, label)
        return (p_acc)


# -----------------------------------
#     SEMI SUPERVISED LEARNING DATA MODULE
# -----------------------------------

class SSLDataModulePSup(pl.LightningDataModule):
    def __init__(self, orig_data, using_resampled_lab,lab_bsize,tot_bsize: int = 64):
        super().__init__()
        self.orig_data = orig_data
        self.tot_bsize = tot_bsize
        
        self.lab_bsize=lab_bsize
        
        self.using_resampled_lab=using_resampled_lab #whether we resample labels with replacement or just use straight label sample. resample = more stable training.
        
        

    def setup(self, stage: Optional[str] = None,precision='16'):

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
        if precision=='16':
            X_train_lab = orig_data['label_features'].cuda().half()
            y_train_lab =orig_data['label_y'].cuda().half()
            
            X_train_lab_rs = X_train_lab[resampled_i]
            y_train_lab_rs = y_train_lab[resampled_i]
            
            vfeat=vfeat.cuda().half()
            vlab=vlab.cuda().half()
            
            
        elif precision=='32':
            X_train_lab = orig_data['label_features'].cuda().float()
            y_train_lab = orig_data['label_y'].cuda().float()
            
            X_train_lab_rs = X_train_lab[resampled_i]
            y_train_lab_rs = y_train_lab[resampled_i]
            
            vfeat=vfeat.cuda().float()
            vlab=vlab.cuda().float()
            
            
        if self.using_resampled_lab:
            self.data_train = torch.utils.data.TensorDataset(X_train_lab_rs,y_train_lab_rs)
            self.batch_size=self.tot_bsize
        else:
            self.data_train = torch.utils.data.TensorDataset(X_train_lab,y_train_lab)
            self.lab_batch_size=self.lab_bsize
            
        #vfeat = X_val.unsqueeze(0)
        #vlab = y_val.unsqueeze(0)
        self.data_validation = torch.utils.data.TensorDataset(vfeat, vlab)
        self.nval = vlab.shape[0]

        return (self)

    def train_dataloader(self):
        #has_gpu=torch.cuda.is_available()
        if has_gpu:
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        has_gpu=torch.cuda.is_available()

        if has_gpu:
            return DataLoader(self.data_validation, batch_size=self.nval)
        else:
            return DataLoader(self.data_validation, batch_size=self.nval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_n', help='dataset number 1-5 OR MOG', type=str)
    parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ', type=int)
    parser.add_argument('--n_iterations', help='how many iterations to train classifier for', type=int)
    parser.add_argument('--lr', help='learning rate ie 1e-2, 1e-3,...', type=float)
    parser.add_argument('--use_single_si', help='do we want to train on collection of si or only single instance',type=str)
    parser.add_argument('--tot_bsize', help='unlabelled + labelled batch size for training', type=int)
    parser.add_argument('--lab_bsize', help='labelled data batch size for training', type=int)
    parser.add_argument('--n_trials', help='how many trials to do', type=int, default=10)
    parser.add_argument('--use_gpu', help='use gpu or not', type=str, default='False')
    parser.add_argument('--estop_patience', help='early stopping patience', type=int, default=10)
    parser.add_argument('--metric', help='which metric to select best model. bce or acc', type=str, default='val_acc')
    parser.add_argument('--use_tuned_hpms', help='use tuned hyper params or not', type=str, default='False')
    parser.add_argument('--min_epochs', help='min epochs to train for', type=int, default=10)
    parser.add_argument('--precision',help='what precision u want, ie 16, 32, 16-true etc',type=str,default='32')
    parser.add_argument('--plot_decision_boundary',help='plot the decision boundary ? or not',type=str,default='False')

    args = parser.parse_args()

    args.use_single_si = str_to_bool(args.use_single_si)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)
    args.plot_decision_boundary = str_to_bool(args.plot_decision_boundary)

    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)
    # write dataset spec shorthand
    dspec = master_spec['dataset_spec']
    dspec.set_index("d_n", inplace=True)  # set this index for easier
    # store index of pandas loc where we find the value
    dspec = dspec.loc[args.d_n]  # use this as reerence..
    dspec.d_n = str(args.d_n) if dspec.d_n_type == 'str' else int(args.d_n)
    # GPU preamble
    
    has_gpu = torch.cuda.is_available()

    # now we want to read in dataset_si
    csi = master_spec['dataset_si'][dspec.d_n].values
    candidate_si = csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si == True:  # so we want to use single si, not entire range
        # args.optimal_si_list=[args.optimal_si_list[args.s_i]]
        args.optimal_si_list = [args.s_i]

    #gpu_kwargs = get_gpu_kwargs(args)
    gpu_kwargs = {}

    result_dict = {}
    results_list = []

    for k, si_iter in enumerate(args.optimal_si_list):
        result_dict[si_iter] = 0
        results_list = []

        orig_data = load_data(d_n=args.d_n, s_i=si_iter, dataset_folder=dspec.save_folder)  # load data
        ssld = SSLDataModulePSup(orig_data, lab_bsize=args.lab_bsize,tot_bsize=args.tot_bsize,using_resampled_lab=True)

        ssld.setup()  # initialise the data

        dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns

        # get the data for validation
        val_features = torch.tensor(ssld.data_validation[0][0],device=torch.device('cuda'))
        val_lab = torch.tensor(ssld.data_validation[0][1],device=torch.device('cuda'))

        PSUP_default_args = {'lr': args.lr}

        model_init_args = {
            'd_n': args.d_n,
            's_i': si_iter,
            'dn_log': dspec.dn_log,
            'input_dim': dspec['input_dim'],
            'output_dim': 2,

        }

        model_init_args.update(PSUP_default_args)

        model_name='PARTIAL_SUPERVISED_CLASSIFIER'
        #if args.use_tuned_hpms==True:
            #get json file that has been tuned
            #load
        #    params_dict_fn = f'{dspec.save_folder}/{model_name}.json'
            # export to json
        #    input_f=open(params_dict_fn,'r')
        #    tuned_hpms=json.load(input_f)
        #    model_init_args.update(tuned_hpms)

        optimal_model = None
        optimal_trainer = None

        # START TIME
        st = time.time()
        
        
        # TENSORBOARD LOGGER
        
        
        extra_trainer_kwargs={'precision':args.precision}#,'gradient_clip_val':0.5}
        
        
        for t in range(args.n_trials):
            print(f'doing s_i: {si_iter}\t t: {t}\t of: {args.n_trials}')


            # TRAINING CALLBACKS
            callbacks = []
            max_pf_checkpoint_callback = return_chkpt_max_val_acc(current_model_name,
                                                                  dspec.save_folder)  # returns max checkpoint

            if args.metric == 'val_bce':
                estop_cb = return_estop_val_bce(patience=args.estop_patience)
            elif args.metric == 'val_acc':
                estop_cb = return_estop_val_acc(patience=args.estop_patience)

            callbacks.append(max_pf_checkpoint_callback)
            callbacks.append(estop_cb)

            tb_logger = get_default_logger(current_model_name, args.d_n, si_iter, t)
            

            # TRAINER
            trainer = get_trainer_psup(args, tb_logger, callbacks, DETERMINISTIC_FLAG, **extra_trainer_kwargs)

            with trainer.init_module():
                # models created here will be on GPU and in float16
                # CREATE MODEL
                current_model = PartialSupervisedClassifier(**model_init_args)  # define model

            # INITIALISE WEIGHTS
            current_model.apply(init_weights_he_kaiming)  # re init model and weights


            # DELETE OLD SAVED MODELS
            clear_saved_models(current_model.model_name, dspec.save_folder, si_iter)



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
                                                                  metric=args.metric)

            del trainer
            
            
            if optimal_acc==1.0: #break cos no need to keep going...
                break

        # END TIME
        et = time.time()
        print('time taken: {0} minutes'.format((et - st) / 60.))

        # DELETE OLD SAVED MODELS
        clear_saved_models(current_model.model_name, dspec.save_folder, si_iter)

        # CREATE NAME TO SAVE MODEL
        model_save_fn = create_model_save_name(optimal_model, optimal_trainer, dspec)

        # SAVE THE TRAINER
        optimal_trainer.save_checkpoint(model_save_fn)
        print('optimal model saved here')
        print(model_save_fn)

        # quick dirty prediction on the data

        print('unlabel prediction acc: ')
        print((optimal_model.predict(orig_data['unlabel_features'].to(optimal_model.device)) == orig_data['unlabel_y'].argmax(1).numpy()).mean())

        # EVALUATE ON DATA
        evaluate_on_test_and_unlabel(dspec, args, si_iter, current_model, optimal_model, orig_data, optimal_trainer)

        print('pausing here')

        if args.plot_decision_boundary:
            
            print('plotting decision boundaries (plotly)')
            
            # PLOT HARD DECISION BOUNDARY
            plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=True, output_html=False)

            # PLOT SOFT (CONTINUOUS) DECISION BOUNDARY
            plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=False, output_html=False)
            
        else:
            print('no plot decision boundary accoridng to args.plot_decision_boundary')

        # DELETE OPTIMALS SO CAN RESTART IF DOING MULTIPLE S_I
        del optimal_trainer
        del optimal_model

