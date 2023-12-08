
from typing import List, Union
import pytorch_lightning as pl
#from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from benchmarks_cgan import *







class Generator_X1(pl.LightningModule):
    def __init__(self,
                 lr,
                 d_n,
                 s_i,
                 dn_log,
                 input_dim,
                 median_pwd,
                 num_hidden_layer,
                 middle_layer_size,
                 precision=32,
                 sel_device='gpu',
                 label_batch_size=4,
                 unlabel_batch_size=256):
        super().__init__()
        self.save_hyperparameters()
        output_dim=input_dim
        input_dim=input_dim
        self.input_dim=input_dim
        if self.hparams.num_hidden_layer==1:
            self.gen=get_one_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==3:
            self.gen=get_three_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==5:
            self.gen=get_standard_net(input_dim=input_dim,
                                      output_dim=output_dim)

        self.vmmd_losses=[]
        self.tmmd_losses=[]
        
        
        self.noise_placeholder_val=torch.zeros((12000,2),device=torch.device('cuda'))
        self.noise_placeholder_train=torch.zeros((256,2),device=torch.device('cuda'))
        
        
        
        # self.log("val_mmd", 1e6) #init!
        # self.log("trans_mmd", 1e6) #init!
        
        self.hparams.dn_log=torch.tensor(float(self.hparams.dn_log),device=torch.device('cuda'))
        self.hparams.s_i=torch.tensor(float(self.hparams.s_i),device=torch.device('cuda'))
        
        
        sel_dtype=torch.float32
        if precision==16:
            sel_dtype=torch.float16
            
        if 'gpu' in sel_device or 'cuda' in sel_device:
            sel_device=torch.device('cuda')
        else:
            sel_device=torch.device('cpu')
            
        
        sigma_list=torch.tensor([self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]],dtype=sel_dtype,device=sel_device)
            
        
        rbf_kern=torch.compile(mix_rbf_mmd2_class(sigma_list=sigma_list).to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
        #X=torch.randn((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
        #dummy=rbf_kern(X,X)
        
        self.rbf_kern=rbf_kern
        
    def delete_compiled_modules(self):
        
        del self.rbf_kern
        
        return self

    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x
    
    def training_step(self, batch, batch_idx):
        #set_trace()
        #sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]
        target_x = batch[0]
        #sample noise
        #z=torch.randn((x.shape[0],self.input_dim), device=self.device)
        #z=torch.randn_like(target_x)
        self.noise_placeholder_train.normal_()
        #cat input...
        #gen_input=z
        #prediction
        x_hat=self.gen(self.noise_placeholder_train[:target_x.shape[0]])
        loss=self.rbf_kern(x_hat,target_x)
        self.log('mmd_loss', loss)
        return(loss)


    def configure_optimizers(self):
        return torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)

    def validation_step(self, batch, batch_idx):
        
        
        #-----------------------
        
        #self.first_func()
        
        #median pwd of entire labelled + unlabelled
        sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]
        #get median pwd of val only and see if it maeks a difference
        val_feat=batch[0].squeeze(0)
        trans_feat=val_feat
        
        #self.second_func()
        
        trans_feat = batch[1].squeeze(0)
        
        

        
        # if val_feat.shape[0]>10000:
        #     val_feat=val_feat[torch.randperm(10000),:]
        self.noise_placeholder_train.normal_()
        self.noise_placeholder_val.normal_()

        
        
        
        #gen_input=torch.randn_like(val_feat) #sample noise
        
        
        #self.third_func()
        x_hat=self.gen(self.noise_placeholder_train[:val_feat.shape[0]]) #generate x samples random
        #get rbf mmd2 joint

        val_mmd_loss=mix_rbf_mmd2(x_hat,val_feat,sigma_list=sigma_list)
        
        #if x_hat.shape[0]>50000, too big:
        
        # if trans_feat.shape[0]>10000:
        #     trans_feat=trans_feat[torch.randperm(10000),:]



        #gen_input = torch.randn_like(trans_feat)  # sample noise
        x_hat = self.gen(self.noise_placeholder_val[:trans_feat.shape[0]])  # generate x samples random
        trans_mmd_loss=mix_rbf_mmd2(x_hat,trans_feat,sigma_list=sigma_list)
        
        
        val_trans_mmd_loss=torch.mean(val_mmd_loss + trans_mmd_loss)
        

        #-----------------------
        
        
        
        
        
        
        
        
        self.log("val_mmd", val_mmd_loss)
        self.log("trans_mmd", trans_mmd_loss)
        self.log("val_trans_mmd",val_trans_mmd_loss)
        
        
        
        
        
        
        
        #self.vmmd_losses.append(val_mmd_loss)
        #self.tmmd_losses.append(trans_mmd_loss)
        
        

        #print('t mmd loss: {0}'.format(trans_mmd_loss))
        self.log("s_i",self.hparams.s_i)
        self.log("d_n",self.hparams.dn_log)
        return(self)
