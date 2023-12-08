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


try:
    import os
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]
except:
    print('try deleting slurm but no slurm')
finally:
    print('proceed w training')

    
import argparse

import torch
import torch.distributions as D
from torch.utils.data import DataLoader
from typing import Optional

torch.set_float32_matmul_precision('high') #try with 4090

DETERMINISTIC_FLAG=False

    
current_model_name=cur_model_name='TRIPLE_GAN'

# -----------------------------------
#     TRIPLE GAN CLASSIFIER
# -----------------------------------

class TripleGANClassifier(pl.LightningModule):
    def __init__(self, d_n, dn_log, s_i, lr, threshold_for_pseudo_dloss, alpha_p, alpha, rp_term, input_dim, latent_dim,
                 n_classes,output_dim,tot_bsize=None,best_value=None):
        super().__init__()

        self.save_hyperparameters()
        self.generator = get_standard_net(latent_dim + n_classes, input_dim)
        self.discriminator = get_standard_net(input_dim + n_classes, 2)
        self.classifier = get_standard_net(input_dim, n_classes)

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.rp_term = 0

        self.val_accs = []

        self.model_name = current_model_name
        
        
        self.hparams['dn_log']=int(self.hparams['dn_log'])
        
        self.hparams['s_i']=int(self.hparams['s_i'])
        
        
        
        
        
        self.automatic_optimization=False







    def compile_nets(self):
        
        self.generator = torch.compile(self.generator)
        self.discriminator = torch.compile(self.discriminator)
        self.classifier = torch.compile(self.classifier)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # here just use as classifier
        classification = self.classifier(x)
        return classification



    def training_step(self, batch, batch_idx):
        x_l, y_l, x_ul = batch
        D_x, D_y = x_l, y_l
        C_x = torch.cat((x_l, x_ul), 0)
        G_y = y_l
        
        
        
        
        
        
        
        from IPython.core.debugger import set_trace
        
        
        #set_trace()
        
        #G_y = torch.nn.functional.one_hot(G_y,num_classes=2)
        noise = torch.randn_like(D_x)
        G_input = torch.cat((noise, G_y), 1)
        G_x = self.generator(G_input)
        C_y = get_softmax(self.classifier(C_x))





        opt_d, opt_c, opt_g = self.optimizers()




        # optimizer order is:
        # 1. discriminator
        # 2. classifier
        # 3. generator

        # discriminator
        #if optimizer_idx == 0:
        D_d_input = torch.cat((D_x, D_y), 1)
        lossD_d = torch.log(get_softmax(self.discriminator(D_d_input)) + 1e-5)

        D_g_input = torch.cat((G_x, G_y), 1).detach()
        lossD_g = torch.log(1 - get_softmax(self.discriminator(D_g_input)) + 1e-5) #detach gradients here?

        D_c_input = torch.cat((C_x, C_y), 1).detach()
        lossD_c = torch.log(1 - get_softmax(self.discriminator(D_c_input)) + 1e-5)

        lossD = - torch.mean(lossD_d) + self.hparams['alpha'] * torch.mean(lossD_c) + (1 - self.hparams['alpha']) * torch.mean(lossD_g)
        loss = lossD
        
        opt_d.zero_grad()
        
        self.manual_backward(loss)
        
        opt_d.step()
        
        self.log('disc_train_loss', loss)

        #return (loss)

        # classifier
        #if optimizer_idx == 1:

        classified_labels = self.classifier(D_x)
        #set_trace()
        #true_labels = D_y.type(torch.LongTensor).to(self.device)

        # this is the cross-entropy term corresponding to labelled data
        R_L = self.ce_loss(classified_labels, torch.argmax(D_y,1))
        # each member of classified_labels is 1d tensor of probabilites

        # generator prediction conditional on class Y
        classified_gen_labels = self.classifier(G_x.detach())
        orig_gen_labels = torch.argmax(G_y.detach(), 1)

        # below is the cross-entropy term corresponding to data generated conditional on some label
        R_P = self.hparams['alpha'] * self.ce_loss(classified_gen_labels, orig_gen_labels)





        disc_input = torch.cat((C_x, C_y), 1)

        # partial loss coming from the discriminator
        partial_lossC = self.hparams['alpha'] * torch.mean(
            get_softmax(self.classifier(C_x)) * torch.log(1 - get_softmax(self.discriminator(disc_input)) + 1e-5))

        # check if epochs > threshold
        if self.trainer.current_epoch >= self.hparams['threshold_for_pseudo_dloss']:
            self.rp_term = self.hparams['alpha_p']

        # combine 3 components for the entire loss term
        # note that rp_term is 0 before threshold number of epochs reached
        lossC = partial_lossC + R_L + self.rp_term * R_P
        loss = lossC

        self.log('classifier_train_loss', loss)
        
        
        opt_c.zero_grad()
        
        self.manual_backward(loss)
        
        opt_c.step()
        
        
        
        
        
        
        
        
            
            
        disc_input = torch.cat((G_x, G_y), 1)
        lossG = (1 - self.hparams['alpha']) * torch.mean(torch.log(1 - get_softmax(self.discriminator(disc_input)) + 1e-5))
        loss = lossG
        self.log('generator_train_loss', loss)
        
        
        
        
        
        
                
        opt_g.zero_grad()
        
        self.manual_backward(loss)
        
        opt_g.step()
        
        
        
        
        
        
        
        #return (loss)

    def configure_optimizers(self):

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams['lr'])
        opt_c = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams['lr'])
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams['lr'])

        return opt_d, opt_c, opt_g

    # def validation_step(self, batch, batch_idx):
    #     val_feat = batch[0].squeeze(0)
    #     val_y = batch[1].squeeze(0)
    #     # get val loss
    #     y_hat = self.classifier(val_feat)
    #     v_acc = get_accuracy(y_hat, val_y)
    #     self.log("val_acc", v_acc, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

    #     # v_bce = torch.nn.functional.cross_entropy(y_hat, val_y)
    #     v_bce = get_bce_w_logit(y_hat, torch.nn.functional.one_hot(val_y,num_classes=2).float())
    #     self.log("val_bce", v_bce, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

    #     self.log("d_n", self.hparams['dn_log'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
    #     self.log("s_i", self.hparams['s_i'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
    #     self.val_accs.append(v_acc.item())


    def validation_step(self, batch, batch_idx):

        #validation data consists of validation features and labels
        val_feat=batch[0].squeeze(0)
        val_y = batch[1].squeeze(0)[:,1].flatten()

        #get val loss
        y_hat = self.classifier(val_feat)
        v_acc=get_accuracy(y_hat,val_y)
        
        #v_bce = get_bce_w_logit(torch.nn.functional.softmax(y_hat,dim=1)+1e-6, batch[1].squeeze(0))
        
        #set_trace()
        # get ce loss
        loss_val_bce = self.ce_loss(y_hat, (batch[1].squeeze(0)[:,1]).to(int))

        self.log('val_bce', loss_val_bce)#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("val_acc", v_acc)#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("d_n",self.hparams['dn_log'])#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("s_i",self.hparams['s_i'])#, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

        self.val_accs.append(v_acc.item())


















    def predict_test(self, features, label):
        prediction = self.classifier(features)
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
        #has_gpu=torch.cuda.is_available()
        return DataLoader(self.data_validation, batch_size=self.nval)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--d_n', help='dataset number 1-5 OR MOG', type=str)
    parser.add_argument('--n_iterations', help='how many iterations to train classifier for', type=int)
    
    parser.add_argument('--tot_bsize', help='unlabelled + labelled batch size for training', type=int)
    parser.add_argument('--lab_bsize', help='labelled data batch size for training', type=int)
    parser.add_argument('--use_gpu', help='use gpu or not', type=str, default='False')
    parser.add_argument('--estop_patience', help='early stopping patience', type=int, default=10)
    parser.add_argument('--metric', help='which metric to select best model. bce or acc', type=str, default='val_acc')
    parser.add_argument('--min_epochs', help='min epochs to train for', type=int, default=10)
    parser.add_argument('--keep_SQL_records', help='keeping to SQL records', type=str, default='True')
    
    parser.add_argument('--use_single_si', help='do we want to train on collection of si or only single instance',type=str,default='xxxx')
    parser.add_argument('--precision',help='what precision u want, ie 16, 32, 16-true etc',type=str,default='32')
    parser.add_argument('--n_trials', help='how many trials to do', type=int, default=10)
    parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ', type=int)
    parser.add_argument('--lr', help='learning rate ie 1e-2, 1e-3,...', type=float)
    
    parser.add_argument('--plot_decision_boundary',help='plot the decision boundary ? or not',type=str,default='False')
    parser.add_argument('--use_tuned_hpms', help='using tuned hyperparameters or not', type=str, default='False')



    args = parser.parse_args()

    args.use_single_si = str_to_bool(args.use_single_si)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)
    args.plot_decision_boundary = str_to_bool(args.plot_decision_boundary)


    print(args)

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

        orig_data = load_data(d_n=args.d_n, s_i=si_iter, dataset_folder=dspec.save_folder) #load data
        ssld = SSLDataModule(orig_data, tot_bsize=args.tot_bsize,lab_bsize=args.lab_bsize,precision=args.precision)#model_init_args['lab_bsize'])
        ssld.setup()  # initialise the data
        # get the data for validation
        val_features = ssld.data_validation[0][0].cuda().float()
        val_lab = ssld.data_validation[0][1].cuda().float()

        dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns

        # get the data for validation
        val_features = ssld.data_validation[0][0]
        val_lab = ssld.data_validation[0][1]

        TGAN_default_args = {
                     'threshold_for_pseudo_dloss': int(args.n_iterations*0.3),
                     'alpha_p': 0.03,
                     'alpha': 0.5,
                     'rp_term': 0,
                     'latent_dim': dspec.input_dim,
                     'n_classes': 2}

        model_init_args = {
            'd_n': args.d_n,
            's_i': si_iter,
            'lr':args.lr,
            'dn_log': dspec.dn_log,
            'input_dim':dspec['input_dim'],
            'output_dim': 2,

        }

        model_init_args.update(TGAN_default_args)

        model_name = 'TRIPLE_GAN'
        if args.use_tuned_hpms == True:
            # get json file that has been tuned
            # load
            params_dict_fn = f'{dspec.save_folder}/{model_name}.json'
            # export to json
            input_f = open(params_dict_fn, 'r')
            tuned_hpms = json.load(input_f)
            model_init_args.update(tuned_hpms)


        optimal_model = None
        optimal_trainer = None

        # START TIME
        st = time.time()
        
        
        DETERMINISTIC_FLAG=False
        
        
        gpu_kwargs={'precision':args.precision}
        
        
        
        
        
        
        
        current_model = TripleGANClassifier(**model_init_args)  # define model
        
        
        
        
        
        for t in range(args.n_trials):
            print(f'doing s_i: {si_iter}\t t: {t}\t of: {args.n_trials}')

            # TRAINING CALLBACKS
            callbacks=[]
            import benchmarks_utils
            max_pf_checkpoint_callback=benchmarks_utils.return_chkpt_max_val_acc(cur_model_name, dspec.save_folder) #returns max checkpoint


            if args.metric=='val_bce':
                estop_cb = benchmarks_utils.return_estop_val_bce(patience=args.estop_patience)
            elif args.metric=='val_acc':
                estop_cb = benchmarks_utils.return_estop_val_acc(patience=args.estop_patience)

            callbacks.append(max_pf_checkpoint_callback)
            callbacks.append(estop_cb)

            # TENSORBOARD LOGGER
            tb_logger=get_default_logger(cur_model_name,args.d_n,si_iter, t)
            DETERMINISTIC_FLAG=False
            # TRAINER
            trainer=get_default_trainer(args,tb_logger,callbacks,DETERMINISTIC_FLAG,min_epochs=args.min_epochs,**gpu_kwargs)
            
            
            
            # CREATE MODEL

            # INITIALISE WEIGHTS

            #with trainer.init_module():
                # models created here will be on GPU and in float16
                # CREATE MODEL
                


            #set_trace()


            # INITIALISE WEIGHTS
            #current_model.apply(init_weights_he_kaiming) # re init model and weights
            current_model.apply(init_weights_he_kaiming)  # re init model and weights
            
            
            
            print(current_model.global_step)

            

            # DELETE OLD SAVED MODELS
            clear_saved_models(cur_model_name,dspec.save_folder,si_iter)

            # TRAIN
            trainer.fit(current_model, ssld)

            
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
        print('time taken: {0} minutes'.format((et-st)/60.))

        # DELETE OLD SAVED MODELS
        clear_saved_models(current_model.model_name,dspec.save_folder,si_iter)

        # CREATE NAME TO SAVE MODEL
        model_save_fn=create_model_save_name(optimal_model,optimal_trainer,dspec)

        # SAVE THE TRAINER
        optimal_trainer.save_checkpoint(model_save_fn)

        # EVALUATE ON DATA
        evaluate_on_test_and_unlabel(dspec, args, si_iter,current_model,optimal_model,orig_data,optimal_trainer)

        print('pausing here')
        #print('plotting decision boundaries (plotly)')


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
