
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
                 sel_device='gpu',
                 precision=32,

                 label_batch_size=4,
                 unlabel_batch_size=256,
                 feature_dim=2,
                 x_l=None,
                 y_l=None,
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
        
        
        self.n_lab=n_lab
        self.n_ulab=n_ulab
        
        
        if x_l is not None:
            self.register_buffer("x_l", x_l)

        if y_l is not None:
            self.register_buffer("y_l", y_l)
        
        self.noise_placeholder_val=torch.zeros((12000,2),device=torch.device('cuda'))
        self.noise_placeholder_train=torch.zeros((256,2),device=torch.device('cuda'))
        
        
        self.hparams.s_i=torch.tensor(float(self.hparams.s_i),device=torch.device('cuda'))
        self.hparams.dn_log=torch.tensor(float(self.hparams.dn_log),device=torch.device('cuda'))
        
        #self.automatic_optimization=False
        
        
        self.precision=int(precision)
        self.sel_device=sel_device


        self.conditional_on_y=True

    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x
        
        
        
        #self.target_x_labelled=target_x_labelled
        #self.target_y_labelled=target_y_labelled

    def set_precompiled(self,dop):
        
        sel_device=self.sel_device
                
        sel_dtype=torch.float32
        if self.precision==16:
            sel_dtype=torch.float16
            
        if 'gpu' in sel_device or 'cuda' in sel_device:
            sel_device=torch.device('cuda')
        else:
            sel_device=torch.device('cpu')
            
        
        self.sigma_list=torch.tensor([self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]],dtype=sel_dtype,device=sel_device)
            
        
        rbf_kern=dop['mix_rbf_mmd2']
        #X=torch.randn((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
        #dummy=rbf_kern(X,X)
        
        #self.rbf_kern=rbf_kern
        
        
        
        self.dop=dop
        
    def delete_compiled_modules(self):
        
        del self.rbf_kern
        
        return self



    def training_step(self, batch, batch_idx):
        unlabelled=batch#['loader_unlabelled']
        sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]
        
        #random sample the samll label
        
        lab_perm=torch.randperm(self.n_lab,device=torch.device('cuda'))[:4]
        

        x_l=self.x_l[lab_perm]
        y_l=self.y_l[lab_perm]

        probs = torch.ones(self.n_lab,device=torch.device('cuda'))/self.n_lab
        ul_p=torch.multinomial(probs,unlabelled[0].shape[0],replacement=True) #for batch, not all ulab
        
        
        

        
        
        y_u=self.y_l[ul_p]
        
        
        #sample noise
        x_l=x_l.reshape((-1,self.hparams['output_dim']))
        #z=torch.randn_like(x_l)
        
        self.noise_placeholder_train.normal_()
        
        #y=torch.nn.functional.one_hot(y)
        
        #cat input...
        gen_input=torch.cat((self.noise_placeholder_train[:x_l.shape[0]],y_l),1)#.float()
        
        #prediction
        x_hat=self.gen(gen_input)
        y_l=y_l.float()
        
        
        #lab_loss=mix_rbf_mmd2_joint(x_hat,x_l,y_l,y_l,sigma_list=sigma_list)
        
        
        lab_loss=self.dop['mix_rbf_mmd2_joint_1_feature_1_label'](x_hat,x_l,y_l,y_l,self.sigma_list)
        
        
        #dict_of_precompiled['mix_rbf_mmd2']=torch.compile(mix_rbf_mmd2_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
        #dict_of_precompiled['mix_rbf_mmd2_joint_1_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
        #dict_of_precompiled['mix_rbf_mmd2_joint_regress_2_feature']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')

        
        
        
        
        
        
        
        
        
        

        self.log('labelled_mmd_loss', lab_loss)
            #return(loss)
            
        

        #if optimizer_idx==1:
        x_u=unlabelled[0]
        
        self.noise_placeholder_train.normal_()
        
        
        x_u=x_u.reshape((-1,self.hparams['output_dim']))
        #z=torch.randn_like(x_u)
        #y=torch.nn.functional.onehot(y)
        gen_input=torch.cat((self.noise_placeholder_train[:x_u.shape[0]],y_u),1)#.float()
        #prediction
        x_hat=self.gen(gen_input)
        #ulab_loss=mix_rbf_mmd2(x_hat,x_u,sigma_list=sigma_list)
        
        
        ulab_loss=self.dop['mix_rbf_mmd2'](x_hat,x_u,self.sigma_list)
        
        
        
        
        
        
        
        
        # get batch size
        # cbatch_size = float(z.shape[0])
        # ratio_cbatch = cbatch_size / self.hparams.n_ulab
        # #loss*=ratio_cbatch
        
        #loss.backward()
        
        #opt_ulab.step()
        
        self.log('unlabelled_mmd_loss', ulab_loss)
        
        
        
        loss = lab_loss + ulab_loss
        
        
        return(loss)
        
        
        #return(loss)



    
    
    
    
    # def training_step(self, batch, batch_idx):
        
    #     labelled=batch['loader_labelled']
    #     unlabelled=batch['loader_unlabelled']

    #     sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]

        
    #     #opt_lab, opt_ulab=self.optimizers()
    #     #opt_lab.zero_grad()
    #     #opt_ulab.zero_grad()

    #     #if optimizer_idx==0: #labelled loader
    #     x,y=labelled
    #     #sample noise
    #     x=x.reshape((-1,self.hparams['output_dim']))
    #     z=torch.randn_like(x)
        
    #     #y=torch.nn.functional.one_hot(y)
        
    #     #cat input...
    #     gen_input=torch.cat((z,y),1).float()
    #     #prediction
    #     x_hat=self.gen(gen_input)
    #     y=y.float()
    #     lab_loss=mix_rbf_mmd2_joint(x_hat,x,y,y,sigma_list=sigma_list)
    #     # get batch size
    #     #cbatch_size = float(z.shape[0])
    #     #ratio_cbatch = cbatch_size / self.hparams.n_lab
    #     #loss*=ratio_cbatch
        
    #     #loss.backward()
        
    #     #opt_lab.step()
    #     self.log('labelled_mmd_loss', lab_loss)
    #         #return(loss)
            
        

    #     #if optimizer_idx==1:
    #     x,y=unlabelled
    #     x=x.reshape((-1,self.hparams['output_dim']))
    #     z=torch.randn_like(x)
    #     #y=torch.nn.functional.onehot(y)
    #     gen_input=torch.cat((z,y),1).float()
    #     #prediction
    #     x_hat=self.gen(gen_input)
    #     ulab_loss=mix_rbf_mmd2(x_hat,x,sigma_list=sigma_list)
    #     # get batch size
    #     # cbatch_size = float(z.shape[0])
    #     # ratio_cbatch = cbatch_size / self.hparams.n_ulab
    #     # #loss*=ratio_cbatch
        
    #     #loss.backward()
        
    #     #opt_ulab.step()
        
    #     self.log('unlabelled_mmd_loss', ulab_loss)
        
        
        
    #     loss = lab_loss + ulab_loss
        
        
    #     return(loss)
        
        
    #     #return(loss)



    def configure_optimizers(self):
        #try with just single optimiser. having two optimiser doesn't make sense.
        #self.opt_lab = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        #self.opt_ulab = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.Adam(self.gen.parameters(),lr=self.hparams.lr)
        return optimizer

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
        #noise=torch.randn_like(val_feat) #sample noise
        
        
        self.noise_placeholder_val.normal_()
        
        
        
        gen_input=torch.cat((self.noise_placeholder_val[:val_feat.shape[0]],val_y_oh),1).float() #concatentate noise with label info
        x_hat=self.gen(gen_input) #generate x samples random
        #get rbf mmd2 joint
        val_mmd_loss=mix_rbf_mmd2_joint(x_hat,val_feat,val_y_oh,val_y_oh,sigma_list=sigma_list)
        
        #joint mmd transduction
        
        #x,y=batch #entire batch 
        noise=torch.randn_like(trans_feat)  #sample noise
        
        
        self.noise_placeholder_val.normal_()
        
        
        gen_input=torch.cat((self.noise_placeholder_val[:trans_feat.shape[0]],trans_y_oh),1).float() #concatentate noise with label info
        x_hat=self.gen(gen_input) #generate x samples random
        #get rbf mmd2 joint
        #trans_mmd_loss=mix_rbf_mmd2_joint(x_hat,trans_feat,trans_y_oh,trans_y_oh,sigma_list=sigma_list)      



        trans_mmd_loss=mix_rbf_mmd2(x_hat,trans_feat,sigma_list=sigma_list) #unlabelled



        val_trans_mmd_loss=torch.mean(val_mmd_loss + trans_mmd_loss) #labelled + unlabelled
        

        self.log("val_mmd", val_mmd_loss)
        self.log("trans_mmd", trans_mmd_loss)
        self.log("val_trans_mmd",val_trans_mmd_loss)
        
        
        #self.vmmd_losses.append(val_mmd_loss)
        
        #get min one..
        
        #print(self.vmmd_losses)
        

        #self.log("hp_metric", min(self.vmmd_losses))
        #set_trace()
        
        print('-------------------------')
        #print('val mmd loss: {0}'.format(val_mmd_loss))
        #print('t mmd loss: {0}'.format(trans_mmd_loss))
        #print('trans_val_mmd_loss: {0}'.format(val_trans_mmd_loss))
        print('-------------------------')
        
        
        self.log("s_i",self.hparams.s_i)
        self.log("d_n",self.hparams.dn_log)
        return(self)

