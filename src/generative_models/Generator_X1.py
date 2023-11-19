
import pytorch_lightning as pl

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

    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x
    
    def training_step(self, batch, batch_idx):
        #set_trace()
        sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]
        target_x = batch[0]
        #sample noise
        #z=torch.randn((x.shape[0],self.input_dim), device=self.device)
        z=torch.randn_like(target_x)
        #cat input...
        gen_input=z
        #prediction
        x_hat=self.gen(gen_input)
        loss=mix_rbf_mmd2(x_hat,target_x,sigma_list=sigma_list)
        self.log('mmd_loss', loss)
        return(loss)


    def configure_optimizers(self):
        self.g_optim = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        return self.g_optim

    def validation_step(self, batch, batch_idx):
        #median pwd of entire labelled + unlabelled
        sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]
        #get median pwd of val only and see if it maeks a difference
        val_feat=batch[0].squeeze(0)
        trans_feat = batch[1].squeeze(0)
        
        

        
        if val_feat.shape[0]>10000:
            val_feat=val_feat[torch.randperm(10000),:]

        gen_input=torch.randn_like(val_feat) #sample noise
        x_hat=self.gen(gen_input) #generate x samples random
        #get rbf mmd2 joint

        val_mmd_loss=mix_rbf_mmd2(x_hat,val_feat,sigma_list=sigma_list)
        
        #if x_hat.shape[0]>50000, too big:
        
        if trans_feat.shape[0]>10000:
            trans_feat=trans_feat[torch.randperm(10000),:]



        gen_input = torch.randn_like(trans_feat)  # sample noise
        x_hat = self.gen(gen_input)  # generate x samples random
        trans_mmd_loss=mix_rbf_mmd2(x_hat,trans_feat,sigma_list=sigma_list)

        self.log("val_mmd", val_mmd_loss)
        self.log("trans_mmd", trans_mmd_loss)
        self.vmmd_losses.append(val_mmd_loss.detach().item())

        print('t mmd loss: {0}'.format(trans_mmd_loss))
        self.log("s_i",self.hparams.s_i)
        self.log("d_n",self.hparams.dn_log)
        return(self)



