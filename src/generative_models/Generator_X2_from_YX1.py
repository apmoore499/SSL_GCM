


# X1 -> X2
# Y -> X2
import pytorch_lightning as pl

from benchmarks_cgan import *
n_classes=2
class Generator_X2_from_YX1(pl.LightningModule):
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
                 n_lab,
                 n_ulab,
                 sel_device='gpu',
                 precision=32,
                 
                 label_batch_size=4,
                 unlabel_batch_size=256,
                 labmda_U=1.0, #lambda for unlabelled data...
                 dict_for_mmd=None):
        super().__init__()

        self.save_hyperparameters()

        #self.save_hyperparameters(ignore=)
        
        
        
        if self.hparams.num_hidden_layer==1:
            self.gen=get_one_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==3:
            self.gen=get_three_net(input_dim,output_dim,self.hparams.middle_layer_size)
            
        if self.hparams.num_hidden_layer==5:
            self.gen=get_standard_net(input_dim=input_dim,
                                      output_dim=output_dim)

            
        self.vmmd_losses=[]

        self.input_dim=input_dim
        self.output_dim=output_dim
        
        
        self.hparams.s_i=torch.tensor(float(self.hparams.s_i),device=torch.device('cuda'))
        self.hparams.dn_log=torch.tensor(float(self.hparams.dn_log),device=torch.device('cuda'))
        
        #self.automatic_optimization=False
        self.s_i=torch.tensor(float(self.hparams.s_i),device=torch.device('cuda'))
        self.dn_log=torch.tensor(float(self.hparams.dn_log),device=torch.device('cuda'))
        
        
        self.precision=int(precision)
        self.sel_device=sel_device
        
        
        self.conditional_on_y=True


    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        generated_x = self.gen(z)
        return generated_x
    
    # Using custom or multiple metrics (default_hp_metric=False)
    #def on_train_start(self):
     #   self.logger.log_hyperparams(self.hparams, {"hp/metric_1_val_mmd": 0, 
      #                                             "hp/metric_2_trans_mmd": 0})



    def set_precompiled(self,dop):
        
        sel_device=self.sel_device
                
        sel_dtype=torch.float32
        if self.precision==16:
            sel_dtype=torch.float16
            
        if 'gpu' in sel_device or 'cuda' in sel_device:
            sel_device=torch.device('cuda')
        else:
            sel_device=torch.device('cpu')
            
        
        self.sigma_list_target_x=torch.tensor([self.hparams.median_pwd_tx * x for x in [0.125, 0.25, 0.5, 1, 2]],dtype=sel_dtype,device=sel_device)
        self.sigma_list_cond_x=torch.tensor([self.hparams.median_pwd_cx * x for x in [0.125, 0.25, 0.5, 1, 2]],dtype=sel_dtype,device=sel_device)
            
        
        #rbf_kern=dop['mix_rbf_mmd2']
        #X=torch.randn((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
        #dummy=rbf_kern(X,X)
        
        #self.rbf_kern=rbf_kern
        
        
        
        self.dop=dop
        
    def delete_compiled_modules(self):
        
        del self.rbf_kern
        
        return self



    def training_step(self, batch, batch_idx):
        
        #labelled=batch['loader_labelled']
        #unlabelled=batch['loader_unlabelled']

        #sigma_list = [self.hparams.median_pwd * x for x in [0.125, 0.25, 0.5, 1, 2]]

        
        #opt_lab, opt_ulab=self.optimizers()
        #opt_lab.zero_grad()
        #opt_ulab.zero_grad()


    #def training_step(self, batch, batch_idx,optimizer_idx):
        
        #set_trace()

        labelled=batch['loader_labelled']
        unlabelled=batch['loader_unlabelled']

        sigma_list_target_x =  [self.hparams.median_pwd_tx * x for x in [0.125, 0.25, 0.5, 1, 2]]
        sigma_list_conditional_x = [self.hparams.median_pwd_cx * x for x in [0.125, 0.25, 0.5, 1, 2]]
    
       #if optimizer_idx==0: #labelled loader
        target_x,conditional_x,y=labelled
        
        y=y.float()
        #sample noise
        z=torch.randn_like(target_x)

        z=z.reshape((-1,self.hparams['output_dim']))
        target_x = target_x.reshape((-1, self.hparams['output_dim']))
        #conditional_x = conditional_x.reshape((-1, 1))
        #y=torch.nn.functional.one_hot(y)
        
        #cat input...
        gen_input=torch.cat((z,conditional_x,y),1).float()
        #prediction
        x_hat=self.gen(gen_input)


        if self.hparams['dict_for_mmd'] is not None:
            #derive new combined MMD loss
            vlist=self.hparams['dict_for_mmd']['vlist']

            input_features = vlist[:-1] #get input features
            label_var = vlist[-1] #get label variable


            feature_dict={}

            label_dict={}
            label_dict[label_var] = {}
            label_dict[label_var]['est']=y
            label_dict[label_var]['ground_truth'] = y

            for in_f in input_features:
                dloader_idx=self.hparams['dict_for_mmd'][in_f]['dataloader_idx']
                sigma_list=self.hparams['dict_for_mmd'][in_f]['sigma_list']
                feature_dict[in_f]={}
                feature_dict[in_f]['est']=conditional_x[:,dloader_idx]
                feature_dict[in_f]['ground_truth'] =conditional_x[:,dloader_idx]
                feature_dict[in_f]['sigma_list'] = sigma_list

            #self.hparams['dict_for_mmd'][in_f]=
            target_variable=self.hparams['dict_for_mmd']['target_variable']
            sigma_list = self.hparams['dict_for_mmd'][in_f]['sigma_list']
            feature_dict[target_variable]={}
            feature_dict[target_variable]['est'] = x_hat
            feature_dict[target_variable]['ground_truth'] = target_x
            feature_dict[target_variable]['sigma_list'] = sigma_list
                    #get labels

            lab_loss=MMD_multiple(feature_dict=feature_dict, label_dict=label_dict)


            #loss=mix_rbf_mmd2_joint_regress(x_hat,
            #                                target_x,
            #                                conditional_x,
            #                                conditional_x,
            #                                y,
            #                                y,
            #                                sigma_list=sigma_list_target_x,
            #                                sigma_list1=sigma_list_conditional_x)
            # get batch size for balancing contrib of label / unlabel
            cbatch_size = float(z.shape[0])
            #ratio_cbatch = cbatch_size / self.hparams.n_lab
            #loss*=ratio_cbatch

            self.log('labelled_mmd_loss', lab_loss)
            #return(loss)

        else:
            
            lab_loss = self.dop['mix_rbf_mmd2_joint_regress_2_feature_1_label'](x_hat,
                                            target_x,
                                            conditional_x,
                                            conditional_x,
                                            y,
                                            y,
                                            self.sigma_list_target_x,
                                            self.sigma_list_cond_x)
            
            # mix_rbf_mmd2_joint_1_feature_1_label
            # mix_rbf_mmd2_joint_regress_2_feature
            # mix_rbf_mmd2
            # lab_loss=mix_rbf_mmd2_joint_regress(x_hat,
            #                                 target_x,
            #                                 conditional_x,
            #                                 conditional_x,
            #                                 y,
            #                                 y,
            #                                 sigma_list=sigma_list_target_x,
            #                                 sigma_list1=sigma_list_conditional_x)
            # # get batch size for balancing contrib of label / unlabel
            cbatch_size = float(z.shape[0])
            #ratio_cbatch = cbatch_size / self.hparams.n_lab
            #loss *= ratio_cbatch
            #lab_loss = self.dop['mix_rbf_mmd2_joint_regress_2_feature'](x_hat,

            

            self.log('labelled_mmd_loss', lab_loss)
            #return (loss)
            
        #lab_loss.backward()
        
        #opt_lab.step()

        #if optimizer_idx==1:
            
        target_x,conditional_x,y=unlabelled
        
        z=torch.randn_like(target_x)
        z = z.reshape((-1, self.hparams['output_dim']))
        target_x = target_x.reshape((-1, self.hparams['output_dim']))
        #conditional_x = conditional_x.reshape((-1, ))

        #cat input...
        gen_input=torch.cat((z,conditional_x,y),1).float()
        #prediction
        x_hat=self.gen(gen_input).float()

        if self.hparams['dict_for_mmd'] is not None:
            # derive new combined MMD loss

            # derive new combined MMD loss
            vlist = self.hparams['dict_for_mmd']['vlist']

            input_features = vlist[:-1]  # get input features
            label_var = vlist[-1]  # get label variable

            feature_dict = {}

            label_dict = {}
            label_dict[label_var] = {}
            label_dict[label_var]['est'] = y
            label_dict[label_var]['ground_truth'] = y

            for in_f in input_features:
                dloader_idx = self.hparams['dict_for_mmd'][in_f]['dataloader_idx']
                sigma_list = self.hparams['dict_for_mmd'][in_f]['sigma_list']
                feature_dict[in_f] = {}
                feature_dict[in_f]['est'] = conditional_x[:, dloader_idx]
                feature_dict[in_f]['ground_truth'] = conditional_x[:, dloader_idx]
                feature_dict[in_f]['sigma_list'] = sigma_list

            # self.hparams['dict_for_mmd'][in_f]=
            target_variable = self.hparams['dict_for_mmd']['target_variable']
            sigma_list = self.hparams['dict_for_mmd'][in_f]['sigma_list']
            feature_dict[target_variable] = {}
            feature_dict[target_variable]['est'] = x_hat
            feature_dict[target_variable]['ground_truth'] = target_x
            feature_dict[target_variable]['sigma_list'] = sigma_list
            # get labels

            loss = MMD_multiple(feature_dict=feature_dict)  # , label_dict=label_dict)

            # get batch size for balancing contrib of label / unlabel
            cbatch_size = float(z.shape[0])
            #ratio_cbatch = cbatch_size / self.hparams.n_ulab
            #loss *= ratio_cbatch
            ulab_loss *=self.hparams.labmda_U

            self.log('unlabelled_mmd_loss', ulab_loss)
            #return (loss)
        else:


            
            ulab_loss = self.dop['mix_rbf_mmd2_joint_regress_2_feature'](x_hat,
                                            target_x,
                                            conditional_x,
                                            conditional_x,
                                            self.sigma_list_target_x,
                                            self.sigma_list_cond_x)



            # ulab_loss=mix_rbf_mmd2_joint_regress(x_hat,
            #                                 target_x,
            #                                 conditional_x,
            #                                 conditional_x,
            #                                 sigma_list=sigma_list_target_x,
            #                                 sigma_list1=sigma_list_conditional_x)

            # get batch size for balancing contrib of label / unlabel
            ulab_loss *=self.hparams.labmda_U

            self.log('unlabelled_mmd_loss', ulab_loss)
            #return(loss)
        
                    
        #loss.backward()
        
        #opt_ulab.step()
        
        
        
        
        
        
        
        
        
        total_loss = lab_loss + ulab_loss
        
        return total_loss



    def configure_optimizers(self):
        #try with just single optimiser. having two optimiser doesn't make sense.
        #self.opt_lab = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        #self.opt_ulab = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        
        optimizer = torch.optim.Adam(self.gen.parameters(),lr=self.hparams.lr)
        
        return optimizer

    def validation_step(self, batch, batch_idx):
        
        sigma_list_target_x =  [self.hparams.median_pwd_tx * x for x in [0.125, 0.25, 0.5, 1, 2]]
        sigma_list_conditional_x = [self.hparams.median_pwd_cx * x for x in [0.125, 0.25, 0.5, 1, 2]]
    
        # MMD_L (X2,Y,X1) <- Labelled


        # MMD_UL (X2,X1) <- Unlabelled

        # MMD_X2 (X2)

        # Total Loss = MMD_L + MMD_UL + MMD_X2


        # 50 Labelled

        # 950 Unlabelled <- model does not see labels for these

        randperm_val=torch.randperm(10000)
        randperm_trans=torch.randperm(10000)

        # if val_feat.shape[0]>10000:
        #     val_feat=val_feat[torch.randperm(10000),:]

        #
        
        val_feat_target=batch[0].squeeze(0).float()
        val_feat_cond=batch[1].squeeze(0).float()
        val_y=batch[2].squeeze(0)
        

        trans_feat_target=batch[3].squeeze(0).float()
        trans_feat_cond=batch[4].squeeze(0).float()
        trans_y=batch[5].squeeze(0)

        #set_trace()
        # if val_feat_target.shape[0]>10000:
        #     #take over a sample?
        #     val_feat_target=val_feat_target[randperm_val]
        #     val_feat_cond=val_feat_cond[randperm_val]
        #     val_y=val_y[randperm_val]
        
        # if trans_feat_target.shape[0]>10000:
        #     trans_feat_target=trans_feat_target[randperm_trans]
        #     trans_feat_cond=trans_feat_cond[randperm_trans]
        #     trans_y=trans_y[randperm_trans]
        
        val_y_oh=val_y.float()#torch.nn.functional.one_hot(val_y)
        trans_y_oh=trans_y.float()#torch.nn.functional.one_hot(trans_y)

        #joint mmd on validation data
        #x,y=batch #entire batch 
        noise=torch.randn_like(val_feat_target) #sample noise


        noise=noise.reshape((-1,self.hparams['output_dim']))
        val_feat_target=val_feat_target.reshape((-1,self.hparams['output_dim']))

        gen_input=torch.cat((noise,val_feat_cond,val_y_oh),1).float() #concatentate noise with label info
        x_hat=self.gen(gen_input) #generate x samples random
        #get rbf mmd2 joint
        
        
        
        # vfc_clone=val_feat_cond
        # vft_clone=val_feat_target

        # val_mmd_loss = self.dop['mix_rbf_mmd2_joint_regress_2_feature_1_label'](x_hat,
        #                             vft_clone,
        #                             vfc_clone,
        #                             vfc_clone,
        #                             val_y_oh,
        #                             val_y_oh,
        #                         self.sigma_list_target_x,
        #                         self.sigma_list_cond_x).detach()
            
        
        
        
        
        
        
        
        
        
        
        
        
        val_mmd_loss=mix_rbf_mmd2_joint_regress(x_hat,
                                                val_feat_target,
                                                val_feat_cond,
                                                val_feat_cond,
                                                val_y_oh,
                                                val_y_oh,
                                                sigma_list=self.sigma_list_target_x,
                                                sigma_list1=self.sigma_list_cond_x)
        
        #joint mmd transduction
        
        #x,y=batch #entire batch 
        noise=torch.randn_like(trans_feat_target) #sample noise
        noise=noise.reshape((-1,self.hparams['output_dim']))
        #trans_feat_cond=trans_feat_cond.reshape((-1,1))
        trans_feat_target = trans_feat_target.reshape((-1, self.hparams['output_dim']))

        gen_input=torch.cat((noise,trans_feat_cond,trans_y_oh),1).float() #concatentate noise with label info
        x_hat=self.gen(gen_input) #generate x samples random
        #get rbf mmd2 joint
        # trans_mmd_loss=mix_rbf_mmd2_joint_regress(x_hat,
        #                                           trans_feat_target,
        #                                           trans_feat_cond,
        #                                           trans_feat_cond,
        #                                           trans_y_oh,
        #                                           trans_y_oh,
        #                                           sigma_list=sigma_list_target_x,
        #                                           sigma_list1=sigma_list_conditional_x)
        
        
        tft_clone=trans_feat_target
        
        trans_mmd_loss =self.dop['mix_rbf_mmd2'](x_hat,tft_clone,self.sigma_list_target_x).detach()
            
        
        
        
        
        
        #trans_mmd_loss =mix_rbf_mmd2(x_hat,trans_feat_target,sigma_list=sigma_list_target_x)


        
        #self.vmmd_losses.append(val_mmd_loss.detach().item())
        
        
        
        val_trans_mmd_loss=val_mmd_loss + trans_mmd_loss
        
        self.log("val_trans_mmd",val_trans_mmd_loss)
        
        #self.log("val_mmd", val_mmd_loss)
        #self.log("trans_mmd",trans_mmd_loss)
        
        
        
        #self.log("hp_metric", min(self.vmmd_losses))
        print('val_ mmd: {0}'.format(val_mmd_loss))
        print('tranas_ mmd: {0}'.format(trans_mmd_loss))
        print('val_trans_mmd: {0}'.format(val_trans_mmd_loss))
        print('val_trans_mmd: {0}'.format(val_trans_mmd_loss))
        #print('t mmd loss: {0}'.format(trans_mmd_loss))

        self.log("s_i",self.s_i)

            
        self.log("d_n",self.dn_log)
        return(self)
        
