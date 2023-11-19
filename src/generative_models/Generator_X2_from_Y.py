
import pytorch_lightning as pl

from benchmarks_cgan import *

n_classes=2

class Generator_X2_from_Y(pl.LightningModule):
    def __init__(self,
                 lr,
                 d_n,
                 s_i,
                 dn_log,
                 input_dim,
                 output_dim,
                 median_pwd,
                 num_hidden_layer,
                 middle_layer_size,
                 n_lab,
                 n_ulab,
                 label_batch_size=4,
                 unlabel_batch_size=256
                 ):
        super().__init__()

        self.save_hyperparameters()
        
        if self.hparams.num_hidden_layer==1:
            self.gen=get_one_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==3:
            self.gen=get_three_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==5:
            self.gen=get_standard_net(input_dim=input_dim,
                                      output_dim=output_dim)
            
        self.vmmd_losses=[]
        
        
        
        self.automatic_optimization=False

    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x



    def training_step(self, batch, batch_idx):
        
        labelled=batch['loader_labelled']
        unlabelled=batch['loader_unlabelled']

        sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]

        
        opt_lab, opt_ulab=self.optimizers()
        opt_lab.zero_grad()
        opt_ulab.zero_grad()

        #if optimizer_idx==0: #labelled loader
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
        # get batch size
        cbatch_size = float(z.shape[0])
        ratio_cbatch = cbatch_size / self.hparams.n_lab
        loss*=ratio_cbatch
        
        loss.backward()
        
        opt_lab.step()
        self.log('labelled_mmd_loss', loss)
            #return(loss)
            
        

        #if optimizer_idx==1:
        x,y=unlabelled
        x=x.reshape((-1,self.hparams['output_dim']))
        z=torch.randn_like(x)
        #y=torch.nn.functional.onehot(y)
        gen_input=torch.cat((z,y),1).float()
        #prediction
        x_hat=self.gen(gen_input)
        loss=mix_rbf_mmd2(x_hat,x,sigma_list=sigma_list)
        # get batch size
        cbatch_size = float(z.shape[0])
        ratio_cbatch = cbatch_size / self.hparams.n_ulab
        loss*=ratio_cbatch
        
        loss.backward()
        
        opt_ulab.step()
        
        self.log('unlabelled_mmd_loss', loss)
        
        
        #return(loss)


    def configure_optimizers(self):
        self.optim_labelled = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        self.optim_unlabelled = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        return self.optim_labelled, self.optim_unlabelled

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

