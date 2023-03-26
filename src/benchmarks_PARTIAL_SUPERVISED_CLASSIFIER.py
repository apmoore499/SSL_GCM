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
has_gpu=False

import sqlite3
import os
import pandas as pd
import json

SQL_DB_NAME='ALL_RESULTS.db'


class PartialSupervisedClassifier(pl.LightningModule):
    def __init__(self, lr, d_n, s_i, input_dim,dn_log,output_dim,tot_bsize=None,best_value=None):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = get_standard_net(input_dim=input_dim, output_dim=output_dim)
        self.val_accs = []

        self.model_name = 'PARTIAL_SUPERVISED_CLASSIFIER'

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
        x_l, y_l, x_ul = batch
        guess_label = self.classifier(x_l)
        loss_gl = torch.nn.functional.cross_entropy(guess_label, y_l)
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
        v_acc = get_accuracy(y_hat, val_y)
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
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        else:
            return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):

        if has_gpu:
            return DataLoader(self.data_validation, batch_size=self.nval, pin_memory=True)
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
    parser.add_argument('--keep_SQL_records', help='keeping to SQL records', type=str, default='True')

    args = parser.parse_args()

    args.use_single_si = str_to_bool(args.use_single_si)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)
    args.keep_SQL_records = str_to_bool(args.keep_SQL_records)

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

    gpu_kwargs = get_gpu_kwargs(args)

    result_dict = {}
    results_list = []

    for k, si_iter in enumerate(args.optimal_si_list):
        result_dict[si_iter] = 0
        results_list = []

        orig_data = load_data(d_n=args.d_n, s_i=si_iter, dataset_folder=dspec.save_folder)  # load data
        ssld = SSLDataModule(orig_data, batch_size=args.tot_bsize)

        ssld.setup()  # initialise the data

        dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns

        # get the data for validation
        val_features = ssld.data_validation[0][0]
        val_lab = ssld.data_validation[0][1]

        PSUP_default_args = {'lr': args.lr}

        model_init_args = {
            'd_n': args.d_n,
            's_i': si_iter,
            'dn_log': int(dspec.dn_log),
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

        # check if table exist in SQL or not

        sql_device = get_sql_working_device()

        # replace 'CLASSIFIER' string
        if 'CGAN' in model_name:
            sql_model_name = model_name
        else:
            if '_CLASSIFIER' in model_name:
                sql_model_name = model_name.replace('_CLASSIFIER', '')

        cmd_str = f'''SELECT * FROM {sql_model_name} WHERE model_name = '{sql_model_name}' AND d_n = '{args.d_n}' AND device = '{sql_device}' AND s_i = {model_init_args['s_i']} AND batch_size = {args.tot_bsize} AND lr = {args.lr} '''
        conn = sqlite3.connect(SQL_DB_NAME, timeout=60.0)
        candidate_models = pd.read_sql_query(cmd_str, conn)
        if candidate_models.shape[0] > 0 and args.keep_SQL_records:
            print('already exists for this one: exiting')
            sys.exit()
        else:
            print('not already exist for this one: proceed')
            conn.close()

        # ID
        # from application

        # ")
        # "SELECT salary FROM SqliteDb_developers WHERE id = "
        # ID
        # from application

        # ")
        # cmd_str = f'''INSERT INTO {sql_model_name} VALUES ('{model_name}','{args.d_n}',{args.s_i},{args.lr},{args.tot_bsize},'{device}',{unlabel_acc},{test_acc},'{args_sql}','{model_init_args_sql}','{cgan_args_sql}')'''

        # WHERE
        # column_1 = 100;

        # WHERE
        # column_2
        # IN(1, 2, 3);

        # current_cursor = conn.cursor()
        # current_cursor.execute(cmd_str)
        # Save (commit) the changes
        # conn.commit()
        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        # conn.close()

        # START TIME
        st = time.time()

        for t in range(args.n_trials):
            print(f'doing s_i: {si_iter}\t t: {t}\t of: {args.n_trials}')

            # CREATE MODEL
            current_model = PartialSupervisedClassifier(**model_init_args)  # define model

            # INITIALISE WEIGHTS
            current_model.apply(init_weights_he_kaiming)  # re init model and weights

            # TRAINING CALLBACKS
            callbacks = []
            max_pf_checkpoint_callback = return_chkpt_max_val_acc(current_model.model_name,
                                                                  dspec.save_folder)  # returns max checkpoint

            if args.metric == 'val_bce':
                estop_cb = return_estop_val_bce(patience=args.estop_patience)
            elif args.metric == 'val_acc':
                estop_cb = return_estop_val_acc(patience=args.estop_patience)

            callbacks.append(max_pf_checkpoint_callback)
            callbacks.append(estop_cb)

            # TENSORBOARD LOGGER
            tb_logger = get_default_logger(current_model.model_name, args.d_n, si_iter, t)

            # TRAINER
            trainer = get_default_trainer(args, tb_logger, callbacks, DETERMINISTIC_FLAG, **gpu_kwargs)

            # DELETE OLD SAVED MODELS
            clear_saved_models(current_model.model_name, dspec.save_folder, si_iter)

            # TRAIN
            trainer.fit(current_model, ssld)

            # LOAD OPTIMAL MODEL FROM CURRENT TRAINING
            current_model = load_optimal_model(dspec, current_model)

            # COMPARE TO OVERALL OPTIMAL MODEL FROM THIS RUN
            optimal_model, optimal_trainer = return_optimal_model(current_model,
                                                                  trainer,
                                                                  optimal_model,
                                                                  optimal_trainer,
                                                                  val_features,
                                                                  val_lab,
                                                                  metric=args.metric)

            del trainer

        # END TIME
        et = time.time()
        print('time taken: {0} minutes'.format((et - st) / 60.))

        # DELETE OLD SAVED MODELS
        clear_saved_models(current_model.model_name, dspec.save_folder, si_iter)

        # CREATE NAME TO SAVE MODEL
        model_save_fn = create_model_save_name(optimal_model, optimal_trainer, dspec)

        # SAVE THE TRAINER
        optimal_trainer.save_checkpoint(model_save_fn)


        # quick dirty prediction on the data

        print('unlabel prediction acc: ')
        print((optimal_model.predict(orig_data['unlabel_features']) == orig_data['unlabel_y'].argmax(1).numpy()).mean())

        # EVALUATE ON DATA
        evaluate_on_test_and_unlabel(dspec, args, si_iter, current_model, optimal_model, orig_data, optimal_trainer)
        sql_res = return_test_ulab_for_sql(dspec, args, si_iter, current_model, optimal_model, orig_data,
                                           optimal_trainer)

        print('pausing here')
        print('plotting decision boundaries (plotly)')

        # PLOT HARD DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=True, output_html=False)

        # PLOT SOFT (CONTINUOUS) DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=False, output_html=False)

        # DELETE OPTIMALS SO CAN RESTART IF DOING MULTIPLE S_I
        del optimal_trainer
        del optimal_model


#now write to SQL database




args_sql=json.dumps(vars(args))


#model init args convert to not int64

model_init_args_sql=json.dumps(model_init_args)


try:
    cgan_args_sql = json.dumps(cgan_args)
except:
    print('cgan args not in namespace')
    cgan_args_sql=''
finally:
    next


#attempt to connect
conn = sqlite3.connect(SQL_DB_NAME,timeout=60.0)
current_cursor = conn.cursor()
test_acc=float(sql_res['test_acc'][0])
unlabel_acc=float(sql_res['unlabel_acc'][0])


device=get_sql_working_device()


# get time


time_of_execution=time.time()

#replace 'CLASSIFIER' string
if 'CGAN' in model_name:
    model_name=model_name
else:
    if '_CLASSIFIER' in model_name:
        model_name=model_name.replace('_CLASSIFIER','')

cmd_str=f'''INSERT INTO {model_name} VALUES ('{model_name}','{args.d_n}',{args.s_i},{args.lr},{args.tot_bsize},'{device}',{unlabel_acc},{test_acc},{time_of_execution},'{args_sql}','{model_init_args_sql}','{cgan_args_sql}')'''
current_cursor.execute(cmd_str)
# Save (commit) the changes
conn.commit()
# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()

#backup local SQL before exit??

# - model_init_args.dn_log
# - sql_res
# - include extra arg for SQL records
# - hopefully can just copy paste rest?



#current_cursor = conn.cursor()
#for m in model_names:
#    cmd_str=f'CREATE TABLE {m} (d_n text, s_i integer, args blob, model_init_args blob, cgan_args blob)'
#    current_cursor.execute(cmd_str)
#    print(f'db created for model: {m}')

#conn.close()

