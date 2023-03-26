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


# -----------------------------------
#     SSVAE CLASSIFIER
# -----------------------------------

# based on github implementation of VAT in pytorch, external author
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjzvKyclbnyAhXVc30KHSAuA5sQFnoECAIQAQ&url=https%3A%2F%2Fgithub.com%2Flyakaap%2FVAT-pytorch&usg=AOvVaw22N8bw3RADMNF_9GGC7Sem


def train_epoch(model, dataloader, loss_fn, optimizer, epoch, args):
    model.train()

    ELBO_loss = 0

    # with tqdm(total=len(dataloader), desc='epoch {} of {}'.format(epoch+1, args.n_epochs)) as pbar:

    for i, (data, _) in enumerate(dataloader):
        data = data.to(device)

        p_x_z, q_z_x = model(data)
        loss = loss_fn(p_x_z, q_z_x, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tracking
        # pbar.set_postfix(loss='{:.3f}'.format(loss.item()))
        # pbar.update()

        ELBO_loss += loss.item()

    # print('Epoch: {} Average ELBO loss: {:.4f}'.format(epoch+1, ELBO_loss / (len(dataloader))))


def loss_components_fn(x, y, z, p_y, p_z, p_x_yz, q_z_xy,device):
    # SSL paper eq 6 for an given y (observed or enumerated from q_y)
    #print('placeholder')
    #x=torch.nn.functional.softmax(x,dim=1)

    pxyz = torch.nn.functional.log_softmax(x).sum(1).to(device)
    qzxy = torch.nn.functional.log_softmax(x).sum(1).to(device)


    #print('printing device')
    #print(pxyz.device)
    y=y.cpu()
    #print(y.device)
    #print('now doing p_y')
    #print(p_y.log_prob(y).device)
    #print(p_z.log_prob(z).sum(1).device)
    #print(qzxy.device)

    ygrim=p_y.log_prob(y).to(device)


    retval = - pxyz - ygrim - p_z.log_prob(z).sum(1) + qzxy
    return(retval)



class SSLVAEClassifier(pl.LightningModule):
    def __init__(self, lmda,
                 lr,
                 d_n,
                 s_i,
                 x_dim,
                 y_dim,
                 z_dim,
                 dn_log,
                 n_classes,
                 alpha,tot_bsize=None,best_value=None):
        super().__init__()
        self.save_hyperparameters()

        self.p_y = D.OneHotCategorical(probs=1 / y_dim * torch.ones(1, y_dim))
        self.p_z = D.Normal(torch.tensor(0.), torch.tensor(1.))

        self.decoder = get_standard_net(input_dim=z_dim + y_dim, output_dim=x_dim)

        self.encoder_y = get_standard_net(input_dim=x_dim, output_dim=y_dim)

        self.encoder_z = get_standard_net(input_dim=x_dim + y_dim, output_dim=2 * z_dim)

        self.val_accs = []

        self.model_name='SSL_VAE'

    # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- SSL paper eq 4
    def encode_z(self, x, y):
        xy = torch.cat([x, y], dim=1)
        mu, logsigma = self.encoder_z(xy).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())

    # q(y|x) = Categorical(y|pi_phi(x)) -- SSL paper eq 4
    def encode_y(self, x):
        return D.OneHotCategorical(logits=self.encoder_y(x))

    # p(x|y,z) = Bernoulli
    def decode(self, y, z):
        yz = torch.cat([y, z], dim=1)
        return D.Bernoulli(logits=self.decoder(yz))

    # classification model q(y|x) using the trained q distribution
    def forward(self, x):
        classification = self.encode_y(x).probs
        return classification  # return pred labels = argmax

    def training_step(self, batch, batch_idx):
        # LABELLED LOSS
        x_l, y_l, x_ul = batch
        x_l = x_l.view(x_l.shape[0], -1)
        y_l = torch.nn.functional.one_hot(y_l, 2).to(self.device)

        q_y = self.encode_y(x_l)
        q_z_xy = self.encode_z(x_l, y_l)
        z_l = q_z_xy.rsample()
        p_x_yz = self.decode(y_l, z_l) #should be size 7?
        loss = loss_components_fn(x_l, y_l, z_l, self.p_y, self.p_z, p_x_yz, q_z_xy,device=self.device)
        # loss -= self.hparams.alpha * q_y.log_prob(y_l)  # SSL eq 9. CHECK!
        loss -= q_y.log_prob(y_l)

        # UNLABELLED LOSS
        x_ul = x_ul.view(x_ul.shape[0], -1)

        q_y = self.encode_y(x_ul)
        ul_loss = - q_y.entropy()
        for y in q_y.enumerate_support():
            q_z_xy = self.encode_z(x_ul, y)
            z_ul = q_z_xy.rsample()
            p_x_yz = self.decode(y, z_ul)
            L_xy = loss_components_fn(x_ul, y, z_ul, self.p_y, self.p_z, p_x_yz, q_z_xy,device=self.device)
            ul_loss += q_y.log_prob(y).exp() * L_xy

        loss += ul_loss * self.hparams['lmda']
        loss = loss.mean(0)
        self.log('train_loss', loss, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

        return (loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams['lr'])
        return optimizer

    def validation_step(self, batch, batch_idx):
        val_feat = batch[0].squeeze(0)
        val_y = batch[1].squeeze(0)

        y_hat = self.forward(val_feat)
        v_acc = get_accuracy(y_hat, val_y)
        self.log("val_acc", v_acc, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)

        v_bce = get_bce_w_logit(y_hat, torch.nn.functional.one_hot(val_y).float())

        self.log("val_bce", v_bce, on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("d_n", self.hparams['dn_log'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.log("s_i", self.hparams['s_i'], on_step=LOG_ON_STEP, on_epoch=CHECK_ON_TRAIN_END)
        self.val_accs.append(v_acc.item())

    def predict_test(self, features, label):
        prediction = self.forward(features)
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

    args.use_single_si = str_to_bool(args.use_single_si)
    args.use_tuned_hpms = str_to_bool(args.use_tuned_hpms)

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

    print(has_gpu)

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

        SSVAE_default_args = {'lmda': 0.001,
                              'lr': args.lr,
                              'x_dim': dspec['input_dim'],
                              'y_dim': 2,
                              'z_dim': 5,
                              'alpha': 0.05}

        model_init_args = {
            'd_n': args.d_n,
            's_i': si_iter,
            'dn_log': dspec.dn_log,
            'n_classes':2
        }

        model_init_args.update(SSVAE_default_args)

        model_name = 'SSL_VAE'
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
            current_model = SSLVAEClassifier(**model_init_args)  # define model

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
            trainer = get_default_trainer(args, tb_logger, callbacks, DETERMINISTIC_FLAG,min_epochs=args.min_epochs,**gpu_kwargs)

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

        # EVALUATE ON DATA
        evaluate_on_test_and_unlabel(dspec, args, si_iter, current_model, optimal_model, orig_data, optimal_trainer)

        print('pausing here')
        print('plotting decision boundaries (plotly)')

        # PLOT HARD DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=True, output_html=False)

        # PLOT SOFT (CONTINUOUS) DECISION BOUNDARY
        plot_decision_boundaries_plotly(dspec, si_iter, args, optimal_model, hard=False, output_html=False)

        # DELETE OPTIMALS SO CAN RESTART IF DOING MULTIPLE S_I
        del optimal_trainer
        del optimal_model
        del current_model

