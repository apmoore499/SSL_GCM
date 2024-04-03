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

# -----------------------------------
#     SGAN CLASSIFIER
# -----------------------------------

current_model_name='SSL_GAN'
DETERMINISTIC_FLAG=False

import argparse

import torch
import torch.distributions as D
from torch.utils.data import DataLoader
from typing import Optional

torch.set_float32_matmul_precision('high') #try with 4090

DETERMINISTIC_FLAG=False

class SGANClassifier(pl.LightningModule):
    def __init__(self, lr, d_n, s_i, alpha, input_dim,output_dim,dn_log,tot_bsize=None,best_value=None,current_model_name='SSL_GAN',disc_layers=[100,5],gen_layers=[100,5]):
        super().__init__()

        self.save_hyperparameters()
        
        
        
        self.gen = make_mlp(input_dim=input_dim,
                            hidden_layers=gen_layers,
                            output_dim=input_dim)
        
        
        self.disc = make_mlp(input_dim=input_dim,
                            hidden_layers=disc_layers,
                            output_dim=output_dim)
                

        self.aux_lfunc = torch.nn.CrossEntropyLoss()
        self.adv_lfunc = torch.nn.BCEWithLogitsLoss()
        
        self.smax = torch.nn.Softmax(dim=1)

        self.val_accs = []

        self.model_name = current_model_name
        
        
        self.automatic_optimization=False
        
        self.hparams['dn_log']=int(self.hparams['dn_log'])
        self.hparams['s_i']=int(self.hparams['s_i'])#,device=torch.device('cuda'))


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        classification = self.disc(x)
        return classification

    def predict(self, x):  # predicting on numpy array for decision boundary plot
        if type(x) != torch.Tensor: x = torch.Tensor(x)
        pred = self.forward(x)  # perform classification using model
        retval = pred.argmax(1).cpu().detach().numpy()  # convert classification results to numpy, take argmax of binary classification predictions
        return (retval)  # return the value of self.forward, ie, the classification

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        
        opt_d,opt_g=self.optimizers()

        
        x_l, y_l, x_ul = batch
        # sample noise
        z = torch.randn_like(x_ul)
        # generate fake sample
        synthetic_dat = self.gen(z)
        
        from IPython.core.debugger import set_trace
        #set_trace()
        
        
        #set_trace()
        
        
        # 2. get adversarial loss on real data
        #guess_label_real = self.smax(self.disc(x_l))+1e-6
        
        guess_label_real=self.disc(x_l)+1e-6
        auxSup_loss = self.aux_lfunc(guess_label_real, y_l)
        
        
        #put in eps so we don't get NaNs...
        guess_unlabel_real = self.disc(x_ul)+1e-6
        exp_ret = torch.exp(guess_unlabel_real)
        z_x = exp_ret.sum(dim=-1)+1e-6
        D_x_real = z_x / (z_x + 1)  # ie this is equivalent to Z(x)/(Z(x)+1)=D(x) from paper, we want this one to be 1 for discrimiantor!!
        #rl_tens = torch.ones_like(D_x_real)#.float()
        
        log_Dx = F.binary_cross_entropy_with_logits(D_x_real, torch.ones_like(D_x_real))+1e-6
        
        
        
        #guess_label_real=self.disc(x_l)+1e-6

        # 3. get adversarial loss on fake data
        guess_label_fake = self.disc(synthetic_dat.detach())#+1e-6 #nb detach is crucial in this context
        exp_ret = torch.exp(guess_label_fake)+1e-6
        z_gz = exp_ret.sum(dim=-1)
        D_gz = z_gz / (z_gz + 1)
        
        log_D_gz = F.binary_cross_entropy_with_logits(D_gz, torch.zeros_like(D_gz))
        
        errD_unsup = self.hparams['alpha'] * (log_Dx + log_D_gz+1e-6)
        
        
        disc_loss = errD_unsup + auxSup_loss
        disc_loss*=0.1
        
        opt_d.zero_grad()
        
        self.manual_backward(disc_loss)
        
        opt_d.step()
        
            
        # sample noise again, for freed computation graph..........
        #z = torch.randn_like(x_ul)
        # generate fake sample
        #synthetic_dat = self.gen(z)
        
        
        disc_pred_synthetic = self.disc(synthetic_dat)
        
        exp_ret = torch.exp(disc_pred_synthetic)+1e-6
        Z_Dgz = exp_ret.sum(dim=-1)#+1e-6
        D_gz = Z_Dgz / (Z_Dgz + 1)
        #rl_tens = torch.ones_like(disc_pred_synthetic[:, 0]).type_as(disc_pred_synthetic)
        
        log_D_gz = F.binary_cross_entropy_with_logits(D_gz, torch.ones_like(D_gz))+1e-6
        
        
        #gen_loss = self.adv_lfunc(adv_loss_synthetic, rl_tens)+1e-6

        
        gen_loss=log_D_gz*0.1
        
                    
        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()
        
        
        
            
        
        self.log_dict({"gen_train_loss": gen_loss, "disc_train_loss": disc_loss},prog_bar=True)#"temp":self.temp})


    def configure_optimizers(self):

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.hparams['lr'])
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.hparams['lr'])
        return [opt_d, opt_g], []

    def validation_step(self, batch, batch_idx):
        
        
        val_feat = batch[0].squeeze(0).float()
        val_y = batch[1].squeeze(0)[:,1].flatten()

        #set_trace()
        # get val loss
        y_hat = self.disc(val_feat)
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
        prediction = self.disc(features)
        p_acc = get_accuracy(prediction, label)
        return (p_acc)


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

        self.data_validation = torch.utils.data.TensorDataset(vfeat, vlab)
        self.nval = vlab.shape[0]

        return (self)


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.tot_bsize, shuffle=True)


    def val_dataloader(self):
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
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)
    args.keep_SQL_records = str_to_bool(args.keep_SQL_records)
    args.plot_decision_boundary = str_to_bool(args.plot_decision_boundary)







#  #-delete this section bewlow??????????????????
    args = parser.parse_args()


    args.use_single_si = str_to_bool(args.use_single_si)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)


# - delete this one?????????????

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
    has_gpu=False
    if args.use_gpu and torch.cuda.is_available():
        has_gpu=True

    result_dict = {}
    results_list = []

    for k, si_iter in enumerate(args.optimal_si_list):
        result_dict[si_iter] = 0
        results_list = []

        orig_data = load_data(d_n=args.d_n, s_i=si_iter, dataset_folder=dspec.save_folder)  # load data




        model_init_args = {
            'd_n': args.d_n,
            's_i': si_iter,
            'dn_log': dspec.dn_log,
            'input_dim':dspec['input_dim'],
            'output_dim': 2,
            'tot_bsize':args.tot_bsize,
            'lr':args.lr,
            'alpha':0.5

        }

        model_name='SSL_GAN'
        if args.use_tuned_hpms==True:
            #get json file that has been tuned
            #load
            params_dict_fn = f'{dspec.save_folder}/{model_name}.json'
            # export to json
            input_f=open(params_dict_fn,'r')
            tuned_hpms=json.load(input_f)
            model_init_args.update(tuned_hpms)


        dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns



        ssld = SSLDataModule(orig_data, tot_bsize=model_init_args['tot_bsize'],lab_bsize=args.lab_bsize,precision=args.precision)#model_init_args['lab_bsize'])
        ssld.setup()  # initialise the data
        # get the data for validation
        val_features = ssld.data_validation[0][0].cuda().float()
        val_lab = ssld.data_validation[0][1].cuda().float()

        optimal_model = None
        optimal_trainer = None

        # START TIME
        st = time.time()

        gpu_kwargs={'precision':args.precision}
        
        
        for t in range(args.n_trials):
            print(f'doing s_i: {si_iter}\t t: {t}\t of: {args.n_trials}')


            # TRAINING CALLBACKS
            callbacks = []
            max_pf_checkpoint_callback = return_chkpt_max_val_acc(current_model_name,dspec.save_folder)  # returns max checkpoint

            if args.metric == 'val_bce':
                estop_cb = return_estop_val_bce(patience=args.estop_patience)
            elif args.metric == 'val_acc':
                estop_cb = return_estop_val_acc(patience=args.estop_patience)

            callbacks.append(max_pf_checkpoint_callback)
            callbacks.append(estop_cb)

            # TENSORBOARD LOGGER
            tb_logger = get_default_logger(current_model_name, args.d_n, si_iter, t)



            #set_trace()
            # TRAINER
            trainer = get_default_trainer(args, tb_logger, callbacks, DETERMINISTIC_FLAG, min_epochs=args.min_epochs, **gpu_kwargs)



            with trainer.init_module():
                # models created here will be on GPU and in float16
                # CREATE MODEL
                current_model = SGANClassifier(**model_init_args)  # define model


            # INITIALISE WEIGHTS
            current_model.apply(init_weights_he_kaiming)  # re init model and weights






            # DELETE OLD SAVED MODELS
            clear_saved_models(current_model_name, dspec.save_folder, si_iter)

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
