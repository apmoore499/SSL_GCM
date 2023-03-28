
import pytorch_lightning as pl

from benchmarks_cgan import *


class Generator_X_from_X(pl.LightningModule):
    def __init__(self,
                 lr,
                 d_n,
                 s_i,
                 dn_log,
                 input_dim,
                 output_dim,
                 median_pwd_tx,
                 median_pwd_cx,
                 num_hidden_layer,
                 middle_layer_size,
                 label_batch_size=4,
                 unlabel_batch_size=256):
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

    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x


    def training_step(self, batch, batch_idx):
        
        #set_trace()
        unlabelled=batch

        sigma_list_target_x =  [self.hparams.median_pwd_tx * x for x in [0.125, 0.25, 0.5, 1, 2]]
        
        sigma_list_conditional_x = [self.hparams.median_pwd_cx * x for x in [0.125, 0.25, 0.5, 1, 2]]
    
        target_x,conditional_x=unlabelled

        target_x=target_x.reshape((-1,1))

        conditional_x=conditional_x.reshape((target_x.shape[0],-1))

        z=torch.randn_like(target_x)

        #cat input...
        gen_input=torch.cat((z,conditional_x),1).float()
        #prediction
        x_hat=self.gen(gen_input).float()
        
        

        loss=mix_rbf_mmd2_joint_regress(x_hat,
                                        target_x,
                                        conditional_x,
                                        conditional_x,
                                        sigma_list=sigma_list_target_x,
                                        sigma_list1=sigma_list_conditional_x)



        self.log('unlabelled_mmd_loss', loss)

        return(loss)



    def configure_optimizers(self):
        self.g_optim = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        return self.g_optim

    def validation_step(self, batch, batch_idx):
        

        val_feat_target=batch[0].squeeze(0).float().reshape((-1,1))
        val_feat_cond=batch[1].squeeze(0).float().reshape((val_feat_target.shape[0],-1))
        
        sigma_list_target_x =  [self.hparams.median_pwd_tx * x for x in [0.125, 0.25, 0.5, 1, 2]]
        sigma_list_conditional_x = [self.hparams.median_pwd_cx * x for x in [0.125, 0.25, 0.5, 1, 2]]
    

        z=torch.randn_like(val_feat_target)

        #cat input...
        gen_input=torch.cat((z,val_feat_cond),1).float()
        #prediction
        
        
        #print(self.gen)
        #print(gen_input)
        x_hat=self.gen(gen_input).float()

        val_mmd_loss=mix_rbf_mmd2_joint_regress(x_hat,
                                        val_feat_target,
                                        val_feat_cond,
                                        val_feat_cond,
                                        sigma_list=sigma_list_target_x,
                                        sigma_list1=sigma_list_conditional_x)
        
        trans_feat_target=batch[2].squeeze(0).float().reshape((-1,1))
        trans_feat_cond=batch[3].squeeze(0).float().reshape((trans_feat_target.shape[0],-1))

        z=torch.randn_like(trans_feat_target)
        
        #cat input...
        gen_input=torch.cat((z,trans_feat_cond),1).float()
        #prediction
        x_hat=self.gen(gen_input).float()
        
        trans_mmd_loss=mix_rbf_mmd2_joint_regress(x_hat,
                                        trans_feat_target,
                                        trans_feat_cond,
                                        trans_feat_cond,
                                        sigma_list=sigma_list_target_x,
                                        sigma_list1=sigma_list_conditional_x)
        
        self.log("val_mmd", val_mmd_loss)
        self.log("trans_mmd",trans_mmd_loss)
        
        self.vmmd_losses.append(val_mmd_loss.detach().item())

        print('val mmd loss: {0}'.format(val_mmd_loss))
        print('t mmd loss: {0}'.format(trans_mmd_loss))

        
        self.log("s_i",self.hparams.s_i)

        self.log("d_n",self.hparams.dn_log)
        

        return(self)