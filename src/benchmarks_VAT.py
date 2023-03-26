from benchmarks_utils import *
import sys
sys.path.append('./py/generative_models/')
from benchmarks_cgan import *

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


#-----------------------------------
#     VAT CLASSIFIER
#-----------------------------------

#based on github implementation of VAT in pytorch, external author
#https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjzvKyclbnyAhXVc30KHSAuA5sQFnoECAIQAQ&url=https%3A%2F%2Fgithub.com%2Flyakaap%2FVAT-pytorch&usg=AOvVaw22N8bw3RADMNF_9GGC7Sem

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class VATLoss(pl.LightningModule):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

class VATClassifier(pl.LightningModule):
    def __init__(self,d_n,s_i,xi,eps,ip,alpha,lr,lmda,input_dim,output_dim,dn_log,tot_bsize=None,best_value=None):
        super().__init__()
        
        self.save_hyperparameters()

        self.classifier=get_standard_net(input_dim,output_dim)

        self.vat_loss = VATLoss(xi=self.hparams['xi'], eps=self.hparams['eps'], ip=self.hparams['ip'])

        self.ce_loss=torch.nn.CrossEntropyLoss()

        self.val_accs=[]

        self.model_name='VAT'

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # here just use as classifier
        classification = self.classifier(x)
        return classification
    
    def training_step(self, batch, batch_idx):

        #batch is loaded as labelled and then unlabelled
        x_l,y_l,x_ul=batch

        #unlabelled loss
        lds = self.vat_loss(self.classifier, x_ul)

        #labelled loss
        output = self.classifier(x_l) #predict
        classification_loss = self.ce_loss(output, y_l.type(torch.LongTensor).to(self.device)) #get cross entropy loss

        #combine both loss terms
        loss = classification_loss + self.hparams['alpha'] * lds

        #log loss
        self.log('train_loss', loss, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        
        return(loss)
        
    def configure_optimizers(self):
        opt_c = torch.optim.Adam(self.classifier.parameters(),lr=self.hparams['lr']) #classifier optimiser
        return opt_c
    
    def validation_step(self, batch, batch_idx):

        #validation data consists of validation features and labels
        val_feat=batch[0].squeeze(0)
        val_y=batch[1].squeeze(0)

        #get val loss
        y_hat = self.classifier(val_feat)
        v_acc=get_accuracy(y_hat,val_y)

        # get ce loss
        loss_val_bce = self.ce_loss(y_hat, val_y)

        self.log('val_bce', loss_val_bce, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("val_acc", v_acc, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("d_n",self.hparams['dn_log'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("s_i",self.hparams['s_i'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

        self.val_accs.append(v_acc.item())

    def predict_test(self,features,label):
        prediction=self.classifier(features)
        p_acc = get_accuracy(prediction, label)
        return(p_acc)



#-----------------------------------
#     SEMI SUPERVISED LEARNING DATA MODULE
#-----------------------------------


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
    parser.add_argument('--d_n', help='dataset number 1-5 OR MOG',type=str)
    parser.add_argument('--s_i', help='which random draw of s_i in {0,...,99} ',type=int)
    parser.add_argument('--n_iterations', help='how many iterations to train classifier for',type=int)
    parser.add_argument('--lr',help='learning rate ie 1e-2, 1e-3,...',type=float)
    parser.add_argument('--use_single_si',help='do we want to train on collection of si or only single instance',type=str)
    parser.add_argument('--tot_bsize',help='unlabelled + labelled batch size for training',type=int)
    parser.add_argument('--lab_bsize', help='labelled data batch size for training', type=int)
    parser.add_argument('--n_trials', help='how many trials to do', type=int,default=10)
    parser.add_argument('--use_gpu',help='use gpu or not',type=str,default='False')
    parser.add_argument('--estop_patience', help='early stopping patience', type=int, default=10)
    parser.add_argument('--metric', help='which metric to select best model. bce or acc', type=str, default='val_acc')
    parser.add_argument('--min_epochs',help='min epochs to train for',type=int,default=10)
    parser.add_argument('--use_tuned_hpms', help='use tuned hyper params or not', type=str, default='False')

    args = parser.parse_args()

    args.use_single_si=str_to_bool(args.use_single_si)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)

    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec=pd.read_excel('combined_spec.xls',sheet_name=None)
    #write dataset spec shorthand
    dspec=master_spec['dataset_spec']
    dspec.set_index("d_n",inplace=True) #set this index for easier
    #store index of pandas loc where we find the value
    dspec=dspec.loc[args.d_n] #use this as reerence..
    dspec.d_n= str(args.d_n) if dspec.d_n_type=='str' else int(args.d_n)


    #now we want to read in dataset_si
    csi=master_spec['dataset_si'][dspec.d_n].values
    candidate_si=csi[~np.isnan(csi)]
    args.optimal_si_list = [int(s) for s in candidate_si]
    if args.use_single_si==True: #so we want to use single si, not entire range
        #args.optimal_si_list=[args.optimal_si_list[args.s_i]]
        args.optimal_si_list=[args.s_i]

    gpu_kwargs = get_gpu_kwargs(args)
    if args.use_gpu and torch.cuda.is_available():
        has_gpu=True
    else:
        has_gpu=False

    result_dict = {}
    results_list=[]

    for k, si_iter in enumerate(args.optimal_si_list):
        result_dict[si_iter] = 0
        results_list = []

        orig_data = load_data(d_n=args.d_n, s_i=si_iter, dataset_folder=dspec.save_folder) #load data
        ssld = SSLDataModule(orig_data, batch_size=args.tot_bsize)

        ssld.setup() #initialise the data

        dspec['input_dim'] = orig_data['label_features'].shape[1]  # columns

        # get the data for validation
        val_features = ssld.data_validation[0][0]
        val_lab = ssld.data_validation[0][1]

        VAT_default_args={
            'xi':10.0,
            'eps':1.0,
            'ip':1,
            'alpha':0.01,
            'lr':0.01,
            'lmda':0.01
        }

        model_init_args={
            'd_n':args.d_n,
            's_i':si_iter,
            'input_dim':dspec.input_dim,
            'output_dim':2,
            'dn_log':dspec.dn_log
        }

        model_init_args.update(VAT_default_args)

        model_name = 'VAT'
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

        for t in range(args.n_trials):
            print(f'doing s_i: {si_iter}\t t: {t}\t of: {args.n_trials}')

            # CREATE MODEL
            current_model = VATClassifier(**model_init_args)  # define model

            # INITIALISE WEIGHTS
            current_model.apply(init_weights_he_kaiming) # re init model and weights

            # TRAINING CALLBACKS
            callbacks=[]
            max_pf_checkpoint_callback=return_chkpt_max_val_acc(current_model.model_name, dspec.save_folder) #returns max checkpoint


            if args.metric=='val_bce':
                estop_cb = return_estop_val_bce(patience=args.estop_patience)
            elif args.metric=='val_acc':
                estop_cb = return_estop_val_acc(patience=args.estop_patience)

            callbacks.append(max_pf_checkpoint_callback)
            callbacks.append(estop_cb)

            # TENSORBOARD LOGGER
            tb_logger=get_default_logger(current_model.model_name, args.d_n,si_iter, t)

            # TRAINER
            trainer=get_default_trainer(args,tb_logger,callbacks,DETERMINISTIC_FLAG,min_epochs=args.min_epochs,**gpu_kwargs)

            # DELETE OLD SAVED MODELS
            clear_saved_models(current_model.model_name,dspec.save_folder,si_iter)

            # TRAIN
            trainer.fit(current_model, ssld)

            # LOAD OPTIMAL MODEL FROM CURRENT TRAINING
            current_model=load_optimal_model(dspec,current_model)

            # COMPARE TO OVERALL OPTIMAL MODEL FROM THIS RUN
            optimal_model,optimal_trainer=return_optimal_model(current_model,
                                                               trainer,
                                                               optimal_model,
                                                               optimal_trainer,
                                                               val_features,
                                                               val_lab,
                                                               metric=args.metric)


            del trainer

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
        print('plotting decision boundaries (plotly)')

        # PLOT HARD DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=True, output_html=False)

        # PLOT SOFT (CONTINUOUS) DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=False, output_html=False)


        # DELETE OPTIMALS SO CAN RESTART IF DOING MULTIPLE S_I
        del optimal_trainer
        del optimal_model

