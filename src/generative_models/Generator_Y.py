
import pytorch_lightning as pl

from benchmarks_cgan import *

class Generator_Y(pl.LightningModule):
    def __init__(self,
                 d_n,
                 s_i,
                 dn_log,
                 yvals):
        super().__init__()

        #self.save_hyperparameters()
        
        self.set_bp(yvals)
        self.d_n=d_n
        self.s_i=s_i
        
        out_d=get_dataset_folder(d_n)
        #self.save_model(out_d)
        

    def forward(self, n_samples):
        # in lightning, forward defines the prediction/inference actions
        
        seed=torch.ones(n_samples)*self.bp
        generated_y = torch.bernoulli(seed)
        generated_y=torch.nn.functional.one_hot(generated_y.long(),2)
        return generated_y
    
    # Using custom or multiple metrics (default_hp_metric=False)
    #def on_train_start(self):
     #   self.logger.log_hyperparams(self.hparams, {"hp/metric_1_val_mmd": 0, 
      #                                             "hp/metric_2_trans_mmd": 0})
    
        
    
    def set_bp(self,yvals):
        #sets the bernoulli parameter 
        
        #get the data..
        self.bp=yvals.mean()
        return(self)
    
    def save_model(self,data_dir):
        
        #geny.state_dict()
        output_name="data/{0}/saved_models/GEN_Y-d_n={1}-s_i={2}".format(data_dir,self.d_n,self.s_i)

        
        torch.save(self.state_dict,output_name)
        
        return(self)