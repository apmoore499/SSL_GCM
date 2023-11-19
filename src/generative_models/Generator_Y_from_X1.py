
import pytorch_lightning as pl


import sys

import sys
import os

# Add the project directory to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


#sys.path.append('/media/krillman/240GB_DATA/codes2/SSL_GCM/src/generative_models')


from benchmarks_cgan import *
n_classes=2
class Generator_Y_from_X1(pl.LightningModule):
    def __init__(self,
                 lr,
                 d_n,
                 s_i,
                 dn_log,
                 input_dim,
                 output_dim,
                 num_hidden_layer,
                 middle_layer_size):
        super().__init__()

        self.save_hyperparameters()
        

        if self.hparams.num_hidden_layer==1:
            self.classifier=get_one_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==3:
            self.classifier=get_three_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==5:
            self.classifier=get_standard_net(input_dim=input_dim,
                                      output_dim=output_dim)

            
        self.vmmd_losses=[]
        self.celoss = torch.nn.BCEWithLogitsLoss()
        
        self.d_n=d_n
        self.s_i=s_i
        self.dn_log=dn_log

    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.classifier(z.to(self.device))
        return generated_x
    
    # Using custom or multiple metrics (default_hp_metric=False)
    #def on_train_start(self):
     #   self.logger.log_hyperparams(self.hparams, {"hp/metric_1_val_mmd": 0, 
      #                                             "hp/metric_2_trans_mmd": 0})




    def training_step(self, batch, batch_idx):
        
        #set_trace()

        x,y=batch
        #cat input...
        classifier_input=x
        #prediction
        y_hat=self.classifier(classifier_input)

        loss=self.celoss(y_hat,y.float())
        self.log('classifier_bce_loss', loss)
        return(loss)



    def configure_optimizers(self):
        self.g_optim = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr)
        return self.g_optim

    def validation_step(self, batch, batch_idx):
        #set_trace()
        val_feat=batch[0].squeeze(0)
        val_y=batch[1].squeeze(0)
        trans_feat=batch[2].squeeze(0)
        trans_y=batch[3].squeeze(0)
        #get val loss
        y_hat = self.classifier(val_feat)
        v_acc=get_accuracy_cgan(y_hat,val_y)
        v_celoss = self.celoss(y_hat, val_y.float())
        #get transduction loss
        y_hat = self.classifier(trans_feat)

        t_acc=get_accuracy_cgan(y_hat,trans_y)
        t_celoss= self.celoss(y_hat, trans_y.float())


        
        #log them
        self.log("val_acc", v_acc)
        self.log("t_acc",t_acc)

        self.log("v_celoss", v_celoss)
        self.log("t_celoss",t_celoss)

        self.log("s_i",self.hparams.s_i)


     
        self.log("d_n",self.hparams.dn_log)
        
        #print
        print('val acc: {0}'.format(v_acc))
        print('t_acc: {0}'.format(t_acc))



