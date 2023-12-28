import sys
sys.path.append('./src/generative_models/')
from benchmarks_cgan import *
from benchmarks_utils import *
import time
import argparse
import torch
from torch.utils.data import DataLoader
from typing import Optional
import json





import argparse

import torch
import torch.distributions as D
from torch.utils.data import DataLoader
from typing import Optional

torch.set_float32_matmul_precision('high') #try with 4090

DETERMINISTIC_FLAG=False




current_model_name=cur_model_name='LABEL_PROPAGATION'




class LabelPropagationClassifier(pl.LightningModule):
    def __init__(self, lr, d_n, s_i, alpha_pl, input_dim, output_dim, dn_log, lppclassifier=None):
        super().__init__()

        self.save_hyperparameters('lr', 'd_n', 's_i', 'alpha_pl', 'input_dim', 'output_dim', 'dn_log')

        self.classifier = get_standard_net(input_dim=input_dim,
                                           output_dim=output_dim)

        if lppclassifier is not None:
            self.classifier = lppclassifier.classifier

        # self.lfunc=torch.nn.CrossEntropyLoss()
        self.val_accs = []
        self.model_name = 'LABEL_PROPAGATION'

        self.T1 = 10
        self.T2 = 600

        self.alpha_f = 3

        self.scale_func = lambda x: (x - self.T1) / (self.T2 - self.T1) * self.alpha_f
        self.hparams['alpha_pl'] = 0  # start alpha_pl at 0

    def set_lppcclassifier(self, lppclassifier):
        self.classifier = lppclassifier.classifier
        return (self)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        classification = self.classifier(x)
        return classification

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y, x_ulab, ulab_idx = batch

        x_hat_labelled = get_softmax(self.classifier(x))
        y_labelled = y
        labelled_loss = torch.nn.functional.cross_entropy(x_hat_labelled, y_labelled)

        x_hat_unlabelled = get_softmax(self.classifier(x_ulab))
        pseudo_labels = torch.argmax(x_hat_unlabelled, 1)
        unlabelled_loss = torch.nn.functional.cross_entropy(x_hat_unlabelled, pseudo_labels)

        if self.T1 <= self.global_step < self.T2:
            x = self.hparams['alpha_pl']
            self.hparams['alpha_pl'] = self.scale_func(x)

        if self.T2 <= self.global_step:
            self.hparams['alpha_pl'] = self.alpha_f

        total_loss = labelled_loss + self.hparams['alpha_pl'] * unlabelled_loss  # calculate loss term
        loss = total_loss
        self.log('train_loss', loss, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

        return (loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams['lr'])
        return optimizer

    # def validation_step(self, batch, batch_idx):
    #     val_feat = batch[0].squeeze(0)
    #     val_y = batch[1].squeeze(0)
    #     y_hat = self.classifier(val_feat)
    #     v_acc = get_accuracy(y_hat, val_y)
    #     self.log("val_acc", v_acc, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
    #     v_bce = get_bce_w_logit(y_hat, torch.nn.functional.one_hot(val_y).float())
    #     self.log("val_bce", v_bce, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
    #     self.log("d_n", self.hparams['dn_log'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
    #     self.log("s_i", self.hparams['s_i'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
    #     self.val_accs.append(v_acc.item())
        
    def validation_step(self, batch, batch_idx):
        
        
        val_feat = batch[0].squeeze(0).float()
        val_y = batch[1].squeeze(0)[:,1].flatten()

        #set_trace()
        # get val loss
        y_hat = self.classifier(val_feat)
        v_acc = get_accuracy(y_hat, val_y)

        #v_bce = torch.nn.functional.cross_entropy(y_hat, val_y)
        #set_trace()
        lfunc=torch.nn.BCEWithLogitsLoss()
        
        #v_bce = get_bce_w_logit(torch.nn.functional.softmax(y_hat,dim=1), batch[1][:,:,:][0])
        #def get_bce_w_logit(pred, true):
        v_bce=lfunc(torch.nn.functional.softmax(y_hat,dim=1),batch[1][:,:,:][0].float())
        
        #if np.isnan(v_bce.detach().cpu().numpy()):
        #    set_trace()
        
        self.log("val_bce", v_bce)#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("val_acc", v_acc)#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

        self.log("d_n", self.hparams['dn_log'])#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("s_i", self.hparams['s_i'])#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.val_accs.append(v_acc.item())



        #print(f'val acc: {v_acc}')
        #print(f'val_bce: {v_bce}')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    def predict_test(self, features, label):
        prediction = self.classifier(features)
        p_acc = get_accuracy(prediction, label)
        return (p_acc)


# class PseudoLabelDataModule(pl.LightningDataModule):
#     def __init__(self, orig_data, batch_size: int = 64):
#         super().__init__()
#         self.orig_data = orig_data
#         self.batch_size = batch_size

#     def setup(self, stage: Optional[str] = None):

#         orig_data = self.orig_data

#         # ----------#
#         # Training Labelled
#         # ----------#
#         X_train_lab = orig_data['label_features']
#         y_train_lab = torch.argmax(orig_data['label_y'], 1)

#         # ----------#
#         # Training Unlabelled
#         # ----------#
#         X_train_ulab = orig_data['unlabel_features']
#         y_train_ulab = torch.argmax(orig_data['unlabel_y'], 1)

#         # -------------#
#         # Validation
#         # -------------#

#         X_val = orig_data['val_features']
#         y_val = torch.argmax(orig_data['val_y'], 1)

#         # -------------#
#         # Setting up resampling
#         # -------------#

#         n_unlabelled = X_train_ulab.shape[0]
#         n_labelled = X_train_lab.shape[0]
#         dummy_label_weights = torch.ones(n_labelled)
#         resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_unlabelled, replacement=True)
#         X_train_lab_rs = X_train_lab[resampled_i]
#         y_train_lab_rs = y_train_lab[resampled_i]
#         # ulab_mix is the data train!

#         ulab_idx = torch.Tensor([i for i in range(X_train_ulab.shape[0])])

#         # X_train_ulab_idx=
#         self.data_train = torch.utils.data.TensorDataset(X_train_lab_rs,
#                                                          y_train_lab_rs,
#                                                          X_train_ulab,
#                                                          ulab_idx)
#         vfeat = X_val.unsqueeze(0)
#         vlab = y_val.unsqueeze(0)

#         self.data_validation = torch.utils.data.TensorDataset(vfeat, vlab)
#         self.nval = vlab.shape[0]

#         return (self)

#     def train_dataloader(self):
#         if has_gpu:
#             return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
#         else:
#             return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):

#         if has_gpu:
#             return DataLoader(self.data_validation, batch_size=self.nval, pin_memory=True)
#         else:
#             return DataLoader(self.data_validation, batch_size=self.nval)








# -----------------------------------
#     SEMI SUPERVISED LEARNING DATA MODULE
# -----------------------------------


class PseudoLabelDataModule(pl.LightningDataModule):
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
            
        
        
        

        
        
        
        
        ulab_idx = torch.Tensor([i for i in range(X_train_ulab.shape[0])])

        # X_train_ulab_idx=
        self.data_train = torch.utils.data.TensorDataset(X_train_lab_rs,
                                                         y_train_lab_rs,
                                                         X_train_ulab,
                                                         ulab_idx)

        self.data_validation = torch.utils.data.TensorDataset(vfeat, vlab)
        self.nval = vlab.shape[0]

        return (self)


    def train_dataloader(self):
        #has_gpu=torch.cuda.is_available()
        if has_gpu:
            return DataLoader(self.data_train, batch_size=self.tot_bsize, shuffle=True)
        else:
            return DataLoader(self.data_train, batch_size=self.tot_bsize, shuffle=True)

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
    parser.add_argument('--min_epochs', help='min epochs to train for', type=int, default=10)
    parser.add_argument('--use_tuned_hpms', help='using tuned hyperparameters or not', type=str, default='False')
    parser.add_argument('--keep_SQL_records', help='keeping to SQL records', type=str, default='True')
    parser.add_argument('--precision',help='what precision u want, ie 16, 32, 16-true etc',type=str,default='32')
    parser.add_argument('--plot_decision_boundary',help='plot the decision boundary ? or not',type=str,default='False')


    args = parser.parse_args()

    args.use_single_si = str_to_bool(args.use_single_si)
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

    # now we want to read in dataset_si
    csi = master_spec['dataset_si'][dspec.d_n].values
    candidate_si = csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si == True:  # so we want to use single si, not entire range
        # args.optimal_si_list=[args.optimal_si_list[args.s_i]]
        args.optimal_si_list = [args.s_i]

    gpu_kwargs = get_gpu_kwargs(args)
    if args.use_gpu and torch.cuda.is_available():
        has_gpu = True
    else:
        has_gpu = False

    result_dict = {}
    results_list = []

    for k, si_iter in enumerate(args.optimal_si_list):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        result_dict[si_iter] = 0
        results_list = []

        orig_data = load_data(d_n=args.d_n, s_i=si_iter, dataset_folder=dspec.save_folder)  # load data













        pseudo_dm = PseudoLabelDataModule(orig_data, tot_bsize=args.tot_bsize,lab_bsize=args.lab_bsize,precision=args.precision)#model_init_args['lab_bsize'])
        pseudo_dm.setup()  # initialise the data
        # get the data for validation






        optimal_model = None
        optimal_trainer = None

        # START TIME
        st = time.time()


        dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns

        # get the data for validation
        val_features = pseudo_dm.data_validation[0][0].cuda().float()
        val_lab = pseudo_dm.data_validation[0][1].cuda().float()

        # START TIME
        st = time.time()
        PRECLASSIFIER_default_args = {'lr': args.lr}

        model_init_args = {
            'd_n': args.d_n,
            's_i': si_iter,
            'dn_log': dspec.dn_log,
            'input_dim': dspec['input_dim'],
            'output_dim': 2,
        }

        model_init_args.update(PRECLASSIFIER_default_args)








        label_prop_init_args = copy.deepcopy(model_init_args)

        label_prop_init_args.update({'alpha_pl': 0.05})  # prop of label to unlabel data

        optimal_model = None
        optimal_trainer = None

        # START TIME
        st = time.time()
        gpu_kwargs={'precision':args.precision}
        
        
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

            # TENSORBOARD LOGGER
            tb_logger = get_default_logger(current_model_name, args.d_n, si_iter, t)


            DETERMINISTIC_FLAG=False
            #set_trace()
            # TRAINER
            trainer = get_default_trainer(args, tb_logger, callbacks, DETERMINISTIC_FLAG, min_epochs=args.min_epochs, **gpu_kwargs)



            with trainer.init_module():
                # models created here will be on GPU and in float16
                # CREATE MODEL
                current_model = LabelPropagationClassifier(**label_prop_init_args)



            # INITIALISE WEIGHTS
            current_model.apply(init_weights_he_kaiming)  # re init model and weights










            # DELETE OLD SAVED MODELS
            clear_saved_models(current_model_name, dspec.save_folder, si_iter)

            # TRAIN
            trainer.fit(current_model, pseudo_dm)

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


            if optimal_acc==1.0:
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

        # EVALUATE ON DATA
        evaluate_on_test_and_unlabel(dspec, args, si_iter, current_model, optimal_model, orig_data, optimal_trainer)

        print('plotting decision boundaries (plotly)')


        # PLOT HARD DECISION BOUNDARY
        args.plot_decision_boundary=False
        
        #don't know why it's defualting to true...ujust leavve here.
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
        del current_model



        # for t in range(args.n_trials):
        #     # -----------------------------------
        #     #     Training LABEL PROPAGATION model
        #     # -----------------------------------

        #     # label_prop_init_args.update({'lppclassifier':optimal_preclassifier})

        #     # INITIALISE NEW MODEL
        #     current_model = LabelPropagationClassifier(**label_prop_init_args)

        #     # INITIALISE WEIGHTS
        #     current_model.apply(init_weights_he_kaiming)  # re init model and weights

        #     # TRAINING CALLBACKS
        #     callbacks = []
        #     max_pf_checkpoint_callback = return_chkpt_max_val_acc(current_model.model_name,
        #                                                           dspec.save_folder)  # returns max checkpoint

        #     if args.metric == 'val_bce':
        #         estop_cb = return_estop_val_bce(patience=args.estop_patience)
        #     elif args.metric == 'val_acc':
        #         estop_cb = return_estop_val_acc(patience=args.estop_patience)

        #     callbacks.append(max_pf_checkpoint_callback)
        #     callbacks.append(estop_cb)

        #     # TENSORBOARD LOGGER
        #     tb_logger = get_default_logger(current_model.model_name, args.d_n, si_iter, t)

        #     # TRAINER
        #     trainer = get_default_trainer(args, tb_logger, callbacks, DETERMINISTIC_FLAG, min_epochs=args.min_epochs,
        #                                   **gpu_kwargs)

        #     # DELETE OLD SAVED MODELS
        #     clear_saved_models(current_model.model_name, dspec.save_folder, si_iter)

        #     # TRAIN
        #     trainer.fit(current_model, pseudo_dm)

        #     # COMPARE TO OVERALL OPTIMAL MODEL FROM THIS RUN
        #     optimal_model, optimal_trainer = return_optimal_model(current_model,
        #                                                           trainer,
        #                                                           optimal_model,
        #                                                           optimal_trainer,
        #                                                           val_features,
        #                                                           val_lab,
        #                                                           metric=args.metric)

        #     del trainer

        # # END TIME
        # et = time.time()
        # print('time taken: {0} minutes'.format((et - st) / 60.))

        # # DELETE OLD SAVED MODELS
        # clear_saved_models(current_model.model_name, dspec.save_folder, si_iter)

        # # CREATE NAME TO SAVE MODEL
        # model_save_fn = create_model_save_name(optimal_model, optimal_trainer, dspec)

        # # SAVE THE TRAINER
        # optimal_trainer.save_checkpoint(model_save_fn)

        # # EVALUATE ON DATA
        # evaluate_on_test_and_unlabel(dspec, args, si_iter, current_model, optimal_model, orig_data, optimal_trainer)

        # print('pausing here')
        # print('plotting decision boundaries (plotly)')

        # #disable plotting
        # try:
        #     # PLOT HARD DECISION BOUNDARY
        #     plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=True, output_html=False)
        #     # PLOT SOFT (CONTINUOUS) DECISION BOUNDARY
        #     plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=False, output_html=False)
        # except:
        #     print('error for plotting for this one')
        # finally:
        #     print('restart run')

        # # DELETE OPTIMALS SO CAN RESTART IF DOING MULTIPLE S_I
        # del optimal_trainer
        # del optimal_model
        # del current_model



