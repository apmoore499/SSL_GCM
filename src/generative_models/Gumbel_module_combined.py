import torch



# X1 -> X2
# Y -> X2
import pytorch_lightning as pl

#torch.set_float32_matmul_precision('medium')


from benchmarks_cgan import *
n_classes=2

from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.callbacks import ModelCheckpoint

def return_chkpt_min_mmd_gumbel(model_name,dspec_save_folder,monitor='labelled_bce_and_all_feat_mmd'):
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=os.path.join(f'{dspec_save_folder}',"saved_models/"),
        #filename=model_name+ "-{d_n:.0f}-{s_i:.0f}-{epoch:02d}-{labelled_bce_and_all_feat_mmd:.4f}",
        filename=model_name+ "-{s_i:.0f}-{epoch:02d}-{labelled_bce_and_all_feat_mmd:.4f}",
        save_top_k=1,
        mode="min",
    )
    return(checkpoint_callback)

def return_estop_min_mmd_gumbel(monitor='labelled_bce_and_all_feat_mmd',patience=50):
    early_stop_callback = EarlyStopping(monitor=monitor,
                                        min_delta=0.00,
                                        patience=patience,
                                        verbose=False,
                                        mode="min",
                                        check_finite=True)
    return(early_stop_callback)












torch.set_float32_matmul_precision('high')




class GumbelModuleCombined(pl.LightningModule):
    def __init__(self,  dsc_generators,
                        labelled_key,
                        conditional_keys,
                        unlabelled_keys,
                        median_pwd_dict,
                        median_pwd_dict_lab,
                        median_pwd_dict_val,
                        feature_idx_subdict,
                        labelled_median_pwd,
                        total_median_pwd,
                        causes_of_y_idx_dl,
                        all_feat_names,
                        dsc,
                        d_n,
                        s_i,
                        dn_log,
                        n_labelled,
                        n_unlabelled,
                        
                        lab_bsize,
                        tot_bsize,
                        val_loss_criterion='labelled_bce_and_all_feat_mmd',
                        lr=1e-3,

                        templist=[]):
        super().__init__()
        
        
        self.dsc_generators=torch.nn.ModuleDict(dsc_generators)
        self.val_loss_criterion=val_loss_criterion
        
        self.median_pwd_dict=median_pwd_dict
        
        self.median_pwd_dict_lab=median_pwd_dict_lab
        self.median_pwd_dict_val=median_pwd_dict_val
        
        
        self.feature_idx_subdict=feature_idx_subdict
        
        self.causes_of_y_idx_dl=causes_of_y_idx_dl

        self.labelled_key=labelled_key
        self.conditional_keys=conditional_keys
        self.unlabelled_keys=unlabelled_keys
        
        
        self.all_feat_names=all_feat_names
        
        
        self.dsc=dsc

        #self.d_n=d_n
        #self.s_i=s_i
        #self.dn_log=dn_log
        

        self.total_median_pwd=total_median_pwd
        self.labelled_median_pwd=labelled_median_pwd
        
        self.n_labelled=n_labelled
        
        
        self.n_unlabelled=n_unlabelled
        
        
        #self.save_hyperparameters('lr','d_n','s_i','dn_log')
        
        self.save_hyperparameters()#'lr','d_n','s_i','dn_log')
        

        
        #self.model = Transformer(vocab_size=vocab_size)
        
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False


        self.cb = torch.nn.BCEWithLogitsLoss() #for label bce loss...
        
        
        #self.temp=templist[0]
        self.tempstep=templist[1]-templist[0]
        self.ntemp=len(templist)
        
        self.mintemp=templist[-1]
        
        
        self.templist=list(templist)
        
        self.log('mintemp',self.mintemp)
        


        self.register_buffer("temp", torch.tensor(templist[0]).float())
        
        
        self.val_outs = []


        self.reset_list_of_train_loss()
        
        
        self.hparams.dn_log=torch.tensor(float(self.hparams.dn_log),device=torch.device('cuda'))
        self.hparams.s_i=torch.tensor(float(self.hparams.s_i),device=torch.device('cuda'))

        self.sigma_list_total=torch.tensor([self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]],device=torch.device('cuda'))
        self.sigma_list_labelled=torch.tensor([self.labelled_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]],device=torch.device('cuda'))




        self.sigma_list_cs=torch.tensor([self.median_pwd_dict['cause_spouse'] * i for i in [0.125, 0.25, 0.5, 1, 2]],device=torch.device('cuda'))

        self.sigma_list_e=torch.tensor([self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]],device=torch.device('cuda'))





    


            




    #     self.setup_dnsi()

    # def setup_dnsi(self):
        
    #     self.si_tens=torch.tensor(float(self.hparams.s_i),device=torch.device('cuda'))
    #     self.sn_tens=torch.tensor(float(self.hparams.d_n),device=torch.device('cuda'))
        
    #     return(self)
    
        self.noise_placeholder_val=torch.zeros((12000,dsc.feature_dim),device=torch.device('cuda'))
        #self.noise_placeholder_train=torch.zeros((256,dsc.feature_dim),device=torch.device('cuda'))
        self.noise_placeholder_train_lab=torch.zeros((lab_bsize,dsc.feature_dim*len(self.conditional_keys)),device=torch.device('cuda'))
        self.noise_placeholder_train_ulab=torch.zeros((tot_bsize,dsc.feature_dim*len(self.conditional_keys)),device=torch.device('cuda'))
    
    
    
    
    
    
    
    
    def reset_list_of_train_loss(self):
        self.train_labouts=torch.zeros(200,device=torch.device('cuda'))
        self.train_ulabouts=torch.zeros(200,device=torch.device('cuda'))
        
        return(self)
        
        
        
    def set_dsc_generators(self,dsc_generators):
        
        self.dsc_generators=torch.nn.ModuleDict(dsc_generators)
        
        return(self)



    def setup_compiled_mmd_losses(self,dict_of_precompiled):
        
        
        dop=dict_of_precompiled
        
#         templates['mix_rbf_kernel']=torch.compile(mix_rbf_kernel_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# templates['mix_rbf_mmd2']=torch.compile(mix_rbf_kernel_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# templates['mix_rbf_mmd2_joint_1_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# templates['mix_rbf_mmd2_joint_regress_2_feature']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')


        #instance_mix_rbf=dop['mix_rbf']
        instance_mix_rbf_mmd2=dop['mix_rbf_mmd2']
        instance_mix_rbf_mmd2_joint_1_feature_1_label=dop['mix_rbf_mmd2_joint_1_feature_1_label']
        instance_mix_rbf_mmd2_joint_regress_2_feature=dop['mix_rbf_mmd2_joint_regress_2_feature']
        instance_mix_rbf_mmd2_joint_regress_2_feature=dop['mix_rbf_mmd2_joint_regress_2_feature']
        
        instance_mix_rbf_mmd2_joint_regress_2_feature_1_label=dop['mix_rbf_mmd2_joint_regress_2_feature_1_label']
        
        #mix_rbf_mmd2_joint_regress_2_feature
        
        
        
        
        X=torch.randn((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
        Y=torch.ones((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
        Y[:,1]=0.0
        
        self.dict_of_mix_rbf_mmd2_joint_regress={}
        
        for ev in self.conditional_keys: #effect of Y
            
            self.dict_of_mix_rbf_mmd2_joint_regress[ev]={}
            
            sigma_list_ev=torch.tensor([self.median_pwd_dict[ev] * i for i in [0.125,0.25,0.5,1,2]],dtype=torch.float16,device=torch.device('cuda')) #effect....
            
            #cy=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label(sigma_list_effect=sigma_list_ev).to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
            
            
            cy=instance_mix_rbf_mmd2_joint_1_feature_1_label
            
                             
                             
                            #  options={'shape_padding':True,
                            #                                                                                                                                                                                              'triton.cudagraphs':True,
                            #                                                                                                                                                                                             'trace.enabled':False,
                            #                                                                                                                                                                                             'epilogue_fusion':True})
            #dummy=cy(X,X,Y,Y)
            mrbf=instance_mix_rbf_mmd2#torch.compile(mix_rbf_mmd2_class(sigma_list=sigma_list_ev).to(torch.float16).cuda(),dynamic=False,fullgraph=True,mode='reduce-overhead')
            
            
            self.dict_of_mix_rbf_mmd2_joint_regress[ev]['conditional_y']=cy
            self.dict_of_mix_rbf_mmd2_joint_regress[ev]['mix_rbf']=mrbf
            self.dict_of_mix_rbf_mmd2_joint_regress[ev]['sigma_list_ev']=sigma_list_ev
            
            
            
            
            for cv in self.unlabelled_keys: #cause of Y
                
                
                self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]={}
                
                
               # sigma_list_cv=[self.median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]] #cause.....
                sigma_list_cv=torch.tensor([self.median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]],dtype=torch.float16,device=torch.device('cuda')) #effect....
                
                
                #compiled_loss=torch.compile(mix_rbf_mmd2_joint_regress_2_feature(sigma_list_effect=sigma_list_ev,sigma_list_cause=sigma_list_cv).to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
                
                #dummy=compiled_loss(X,X,X,X)#,Y,Y)
                
                self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]['mix_rbf_mmd2_joint_regress_2_feature']=instance_mix_rbf_mmd2_joint_regress_2_feature
                self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]['sigma_list_cv']=sigma_list_cv
                self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]['sigma_list_ev']=sigma_list_ev
                
               
        
        self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']={}
        
        
        
        self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2']=dop['mix_rbf_mmd2']
        self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_1_feature_1_label']=dop['mix_rbf_mmd2_joint_1_feature_1_label']
        self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_regress_2_feature']=dop['mix_rbf_mmd2_joint_regress_2_feature']
        self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_regress_2_feature_1_label']=dop['mix_rbf_mmd2_joint_regress_2_feature_1_label']
                 

                #i#nstance_mix_rbf_mmd2_joint_regress_2_feature_1_label
                
                

        return(self)
    #templates
    
    
    
    
#     def setup_compiled_mmd_losses(self,dict_of_precompiled):
        
        
#         dop=dict_of_precompiled
        
# #         templates['mix_rbf_kernel']=torch.compile(mix_rbf_kernel_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# # templates['mix_rbf_mmd2']=torch.compile(mix_rbf_kernel_class().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# # templates['mix_rbf_mmd2_joint_1_feature_1_label']=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
# # templates['mix_rbf_mmd2_joint_regress_2_feature']=torch.compile(mix_rbf_mmd2_joint_regress_2_feature().to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')


#         instance_mix_rbf=dop['mix_rbf']
#         instance_mix_rbf_mmd2=dop['mix_rbf_mmd2']
#         instance_mix_rbf_mmd2_joint_1_feature_1_label=dop['mix_rbf_mmd2_joint_1_feature_1_label']
#         instance_mix_rbf_mmd2_joint_regress_2_feature=dop['mix_rbf_mmd2_joint_regress_2_feature']
        
        
                
        
        
        
        
        
#         X=torch.randn((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
#         Y=torch.ones((4,2),dtype=torch.float16,device=torch.device('cuda'))
        
#         Y[:,1]=0.0
        
#         self.dict_of_mix_rbf_mmd2_joint_regress={}
        
#         for ev in self.conditional_keys: #effect of Y
            
#             self.dict_of_mix_rbf_mmd2_joint_regress[ev]={}
            
#             sigma_list_ev=torch.tensor([self.median_pwd_dict[ev] * i for i in [0.125,0.25,0.5,1,2]],dtype=torch.float16,device=torch.device('cuda')) #effect....
            
#             cy=torch.compile(mix_rbf_mmd2_joint_1_feature_1_label(sigma_list_effect=sigma_list_ev).to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
                             
                             
#                             #  options={'shape_padding':True,
#                             #                                                                                                                                                                                              'triton.cudagraphs':True,
#                             #                                                                                                                                                                                             'trace.enabled':False,
#                             #                                                                                                                                                                                             'epilogue_fusion':True})
#             #dummy=cy(X,X,Y,Y)
#             mrbf=torch.compile(mix_rbf_mmd2_class(sigma_list=sigma_list_ev).to(torch.float16).cuda(),dynamic=False,fullgraph=True,mode='reduce-overhead')
            
            
#             self.dict_of_mix_rbf_mmd2_joint_regress[ev]['conditional_y']=cy
#             self.dict_of_mix_rbf_mmd2_joint_regress[ev]['mix_rbf_feature_only']=mrbf
            
            
#             for cv in self.unlabelled_keys: #cause of Y
                
                
#                # sigma_list_cv=[self.median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]] #cause.....
#                 sigma_list_cv=torch.tensor([self.median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]],dtype=torch.float16,device=torch.device('cuda')) #effect....
                
                
#                 compiled_loss=torch.compile(mix_rbf_mmd2_joint_regress_2_feature(sigma_list_effect=sigma_list_ev,sigma_list_cause=sigma_list_cv).to(torch.float16).cuda(),fullgraph=True,mode='reduce-overhead')
                
#                 #dummy=compiled_loss(X,X,X,X)#,Y,Y)
                
#                 self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]=compiled_loss
                

#         return(self)
#     #templates
    
    def delete_compiled_modules(self):
        
        
        del self.dict_of_mix_rbf_mmd2_joint_regress
        
        
        # for ev in self.conditional_keys: #effect of Y
        #     del self.dict_of_mix_rbf_mmd2_joint_regress[ev]['conditional_y']
        #     del self.dict_of_mix_rbf_mmd2_joint_regress[ev]['mix_rbf_feature_only']
        #     del self.dict_of_mix_rbf_mmd2_joint_regress[ev]['sigma_list_ev']
            
        #     for cv in self.unlabelled_keys: #cause of Y
            
        #         del self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]
                
                
        
        return self


    def training_step(self, batch, batch_idx):
        
        
        labelled_loss=torch.tensor(0.0)
        unlabelled_loss=torch.tensor(0.0)
        
        combined_labelled_optimiser, combined_unlabelled_optimiser = self.optimizers()
        
        #---------------------------------------------
        
        
        # LABELLED STEP
        
        
        
        if batch['labelled'] is not None:
            
            
            
            cur_batch_features, cur_batch_label = batch['labelled']
            # split into features and label for this instance
            # instantiate our ancestor dict that will be used for current batch
            current_ancestor_dict = {}
            # variables where Y is not ancestor
            # append them to current_ancestor_dict
            for k in self.unlabelled_keys:
                # retrieve potential multidm
                associated_names = self.dsc.label_names_alphan[k]
                # retrieve index of feature
                current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
                # put into ancestor dict
                current_ancestor_dict[k] = cur_batch_features[:, current_feature_idx].view(
                    (-1, len(current_feature_idx)))  # maintain orig shape
            # generate label first, and gumbel it
            input_for_y = cur_batch_features[:, self.causes_of_y_idx_dl]
            y_generated = self.dsc_generators[self.labelled_key].forward(input_for_y)
            #hard=False
            y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=False,tau=self.temp)  # I think use softmax label here?
            # now put into ancestor dictionary
            current_ancestor_dict[self.labelled_key] = torch.nn.functional.one_hot(y_gumbel_softmax.argmax(1),num_classes=2)
            # now loop thru rest
            # each k in this instance is for each feature in causal graph
            
            prev_idx=0
                #prev_idx+=dsc.feature_dim
            next_idx=self.dsc.feature_dim
            for k in self.conditional_keys:
                # get our generator please
                k_has_label = self.dsc_generators[k].conditional_on_label
                extra_causes = self.dsc_generators[k].conditional_feature_names
                # get the idx of causes...
                if k_has_label:
                    cur_feature_inputs = extra_causes +[self.labelled_key]
                else:
                    cur_feature_inputs = extra_causes
                # perform list comprehension to extract variables
                cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
                cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
                # then concatenate the variables
                generator_input = torch.cat(cur_feature_inputs_lc, 1)
                # then add some noise
                
                #current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device=generator_input.device)
                self.noise_placeholder_train_lab.normal_()
                
                # then concatenate the noise
                generator_input_w_noise = torch.cat((self.noise_placeholder_train_lab[:generator_input.shape[0],prev_idx:next_idx], generator_input), 1)
                # then predict
                predicted_value = self.dsc_generators[k].forward(generator_input_w_noise)
                # and put this one into ancestor dict
                current_ancestor_dict[k] = predicted_value
                # and pause here for my lord
                # calculate joint mmd between observable X and generated X
                # pull out features from current_ancestor_dict
                
                
                prev_idx+=self.dsc.feature_dim
                next_idx+=self.dsc.feature_dim
                
            cur_ancestors = [current_ancestor_dict[f] for f in self.all_feat_names]
            cur_ancestors = tuple(cur_ancestors)
            batch_estimate_features = torch.cat(cur_ancestors, 1)
            batch_estimate_label = current_ancestor_dict[self.labelled_key]
            cur_batch_label_one_hot = torch.nn.functional.one_hot(cur_batch_label.long(), 2)
            # pull out features in current batch as given by data loader
            # try modify median pwd
            #cur_median_pwd = get_median_pwd(cur_batch_features)
            #sigma_list = [self.labelled_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
            labelled_loss_array=[]


            #get our BCE loss

            #cb = torch.nn.BCEWithLogitsLoss()
            #cbl_oh = cur_batch_label_one_hot.float()
            #loss_lab_bce = cb(y_gumbel_softmax, cbl_oh)

            #labelled_loss = mix_rbf_mmd2_joint(batch_estimate_features,
            #                                   cur_batch_features,
            #                                   y_gumbel_softmax,
            #                                   cur_batch_label_one_hot,
            #                                   sigma_list=sigma_list)

            #labelled_loss_array.append(labelled_loss)

            # get loss on individual feature variables...
            feature_y_losses = []
            # for c in self.unlabelled_keys:
            #     associated_names = self.dsc.label_names_alphan[c]
            #     # retrieve index of feature
            #     current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
            #     # put into ancestor dict
            #     current_batch_ground_truth = cur_batch_features[:, current_feature_idx].view((-1, self.dsc.feature_dim))
            #     # get sigma list for current variable
            #     sigma_list = [self.median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
            #     # calculate mmd loss

            for c in self.conditional_keys:
                associated_names = self.dsc.label_names_alphan[c]
                # retrieve index of feature
                current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
                # put into ancestor dict
                current_batch_ground_truth = cur_batch_features[:, current_feature_idx].view((-1, self.dsc.feature_dim))
                # get sigma list for current variable
                sigma_list_ev = torch.tensor([self.median_pwd_dict_lab[c] * i for i in [0.125, 0.25, 0.5, 1, 2]],device=self.device)
                # calculate mmd loss


                #sigma_list_ev=self.dict_of_mix_rbf_mmd2_joint_regress[c]['sigma_list_ev']
                
                
                
                
                
                mloss=self.dict_of_mix_rbf_mmd2_joint_regress[c]['conditional_y'](current_ancestor_dict[c],current_batch_ground_truth,y_gumbel_softmax,cur_batch_label_one_hot,sigma_list_ev)

                # mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c],
                #                         current_batch_ground_truth,
                #                         y_gumbel_softmax,
                #                         cur_batch_label_one_hot,
                #                         sigma_list=sigma_list)
                
                feature_y_losses.append(mloss)

            for m in feature_y_losses:
                labelled_loss_array.append(m)

            labelled_loss=torch.hstack(labelled_loss_array).mean()
            #n_loss_terms=len(labelled_loss_array)
            #scale by batch size and whether labelled or unlabelled
            #labelled_loss=#total_labelled_loss*cur_batch_features.shape[0]/self.n_labelled/n_loss_terms

            #labelled_loss=total_labelled_loss

            combined_labelled_optimiser.zero_grad()
            self.manual_backward(labelled_loss)
            combined_labelled_optimiser.step()

            # ...log the running loss
            #writer.add_scalar('train_labelled_mmd_loss',labelled_loss, t_iter)

            
        #---------------------------------------------
        
        
        
        #---------------------------------------------
        
        # UNLABELLED STEP
        
        
        cur_batch = batch['unlabelled'][0]
        
        # code go here...
        # instantiate our ancestor dict that will be used for current batch
        current_ancestor_dict = {}
        # variables where Y is not ancestor
        # append them to current_ancestor_dict
        for k in self.unlabelled_keys:
            associated_names = self.dsc.label_names_alphan[k]
            # retrieve index of feature
            current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
            # put into ancestor dict
            current_ancestor_dict[k] = cur_batch[:, current_feature_idx].view((-1, self.dsc.feature_dim))  # maintain orig shape
        # generate label first, and gumbel it
        input_for_y = cur_batch[:, self.causes_of_y_idx_dl]
        y_generated = self.dsc_generators[self.labelled_key].forward(input_for_y)

        y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated,hard=True,tau=self.temp)
        # anneal our way out of the self.temperature
        # now put into ancestor dictionary
        current_ancestor_dict[self.labelled_key] = y_gumbel_softmax
        # now loop thru rest
        # each k in this instance is for each feature in causal graph
        
        self.noise_placeholder_train_ulab.normal_()
        
        prev_idx=0
            #prev_idx+=dsc.feature_dim
        next_idx=self.dsc.feature_dim
        for k in self.conditional_keys:
            # get our generator please
            k_has_label = self.dsc_generators[k].conditional_on_label
            extra_causes = self.dsc_generators[k].conditional_feature_names
            # get the idx of causes...
            if k_has_label:
                cur_feature_inputs = extra_causes + [self.labelled_key]
            else:
                cur_feature_inputs = extra_causes
            # perform list comprehension to extract variables
            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
            # then concatenate the variables
            generator_input = torch.cat(cur_feature_inputs_lc, 1)
            # then add some noise
            #current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device=generator_input.device)
            

            # then concatenate the noise
            generator_input_w_noise = torch.cat((self.noise_placeholder_train_ulab[:generator_input.shape[0],prev_idx:next_idx], generator_input), 1)
            # then predict
            predicted_value = self.dsc_generators[k].forward(generator_input_w_noise)
            # and put this one into ancestor dict
            current_ancestor_dict[k] = predicted_value
            # and pause here for my lord
            # calculate joint mmd between observable X and generated X
            # pull out features from current_ancestor_dict
            
            prev_idx+=self.dsc.feature_dim
            next_idx+=self.dsc.feature_dim
            
            
        cur_ancestors = [current_ancestor_dict[f] for f in self.all_feat_names]
        cur_ancestors = tuple(cur_ancestors)
        batch_estimate = torch.cat(cur_ancestors, 1)
        # pull out features in current batch as given by data loader
        #sigma_list = [self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]

        unlabelled_loss_array=[]

        individual_feat_mmd_losses=[]
        for c in self.conditional_keys:
            associated_names = self.dsc.label_names_alphan[c]
            # retrieve index of feature
            current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
            # put into ancestor dict
            current_batch_ground_truth = cur_batch[:, current_feature_idx].reshape((-1, self.dsc.feature_dim))
            #get sigma list for current variable
            sigma_list=[self.median_pwd_dict[c] * i for i in [0.125,0.25,0.5,1,2]]
            #calculate mmd loss
            #mloss = mix_rbf_mmd2(current_ancestor_dict[c], current_batch_ground_truth, sigma_list)
            sigma_list_ev=self.dict_of_mix_rbf_mmd2_joint_regress[c]['sigma_list_ev']
            
            mloss=self.dict_of_mix_rbf_mmd2_joint_regress[c]['mix_rbf'](current_ancestor_dict[c],current_batch_ground_truth,sigma_list_ev) #mix_rbf_mmd2
            
            
            
                
            #mloss=self.dict_of_mix_rbf_mmd2_joint_regress[c]['mix_rbf_feature_only'](current_ancestor_dict[c],current_batch_ground_truth,y_gumbel_softmax,cur_batch_label_one_hot,sigma_list_ev)
            
            
            individual_feat_mmd_losses.append(mloss)

        #now add into total loss
        for m in individual_feat_mmd_losses:
            unlabelled_loss_array.append(m)

        joint_feat_mmd_losses=[]
        #get loss on joint feature variables...
        
        
        
        
        
        for ev in self.conditional_keys: #effect of Y
            for cv in self.unlabelled_keys: #cause of Y
                #get effect v
                associated_names = self.dsc.label_names_alphan[ev]
                # retrieve index of feature
                current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
                # put into ancestor dict
                current_batch_ground_truth = cur_batch[:, current_feature_idx].view((-1, self.dsc.feature_dim))

                #sigma_list_ev=[self.median_pwd_dict[ev] * i for i in [0.125,0.25,0.5,1,2]] #effect....
                estimate_ev=current_ancestor_dict[ev]
                true_ev=current_batch_ground_truth

                #sigma_list_cv=[self.median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]] #cause.....
                estimate_cv=current_ancestor_dict[cv]
                true_cv=current_ancestor_dict[cv]
                
                
                sigma_list_ev=self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]['sigma_list_ev']
                sigma_list_cv=self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]['sigma_list_cv']
                                
                
                #mix_rbf
                #mix_rbf_mmd2_joint_regress_2_feature
                
                joint_mmd_loss=self.dict_of_mix_rbf_mmd2_joint_regress[ev][cv]['mix_rbf_mmd2_joint_regress_2_feature'](estimate_ev,true_ev,estimate_cv,true_cv,sigma_list_ev,sigma_list_cv)

                # joint_mmd_loss =mix_rbf_mmd2_joint_regress( estimate_ev,
                #                                             true_ev,
                #                                             estimate_cv,
                #                                             true_cv,
                #                                             sigma_list=sigma_list_ev,
                #                                             sigma_list1=sigma_list_cv)

                joint_feat_mmd_losses.append(joint_mmd_loss)


        for m in joint_feat_mmd_losses:
            unlabelled_loss_array.append(m)

        #take mean of unlabelled_loss
        unlabelled_loss_total=torch.hstack(unlabelled_loss_array).mean()
        #n_loss_terms=len(unlabelled_loss_array)

        #scale loss by batch size and number of unlabelled cases
        unlabelled_loss = unlabelled_loss_total# * cur_batch.shape[0] / (self.n_unlabelled)/n_loss_terms

        combined_unlabelled_optimiser.zero_grad()
        #unlabelled_loss.backward()
        self.manual_backward(unlabelled_loss)
        combined_unlabelled_optimiser.step()

        # ...log the running loss
        #writer.add_scalar('train_unlabelled_mmd_loss',
        #                unlabelled_loss, t_iter)

        return(dict(labelled_loss=labelled_loss,unlabelled_loss=unlabelled_loss))
        
        #---------------------------------------------
        
        
        
        
        
        

        
        
    def on_train_batch_end(self,outputs,batch,batch_idx):    
        
        
        labelled_loss =outputs['labelled_loss']#[o['labelled_loss'] for o in outputs].sum()
        unlabelled_loss=outputs['unlabelled_loss']#[o['unlabelled_loss'] for o in outputs].sum()
        
        
        if labelled_loss!=0.0:
            self.train_labouts[batch_idx]=labelled_loss
        
        self.train_ulabouts[batch_idx]=unlabelled_loss
        
        
        
        
        
    def on_train_epoch_end(self) -> None:
        
        
        labelled_loss_train=torch.sum(torch.tensor(self.train_labouts))
        unlabelled_loss_train=torch.sum(torch.tensor(self.train_ulabouts))
        
        
        
        
        print('popping templist here')
        #set_trace()
        
        
        
        self.temp=torch.tensor(self.templist.pop(0)).float()

        
        
        self.reset_list_of_train_loss()
        self.log_dict({"train_labelled_mmd_loss": labelled_loss_train, "train_unlabelled_mmd_loss": unlabelled_loss_train,"temp":self.temp},prog_bar=False)
        
        return 
        
    
    
    def validation_step(self,batch,batch_idx):
        
                #self.val_feat.unsqueeze(0),self.val_lab.unsqueeze(0),self.lab_ulab_feat.unsqueeze(0)
            
        
        #labelled_val_losses=[]
        # code go here...
        
        
        cur_batch_features = batch[0].squeeze(0)
        
        #cur_batch_label = batch['labelled'][1]#.squeeze(0)
        cur_batch_label_one_hot= batch[1].squeeze(0)
        
        unlabelled_batch=batch[2].squeeze(0)
        
        
        
        #self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2']=dop['mix_rbf_mmd2']
        #self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_1_feature_1_label']=dop['mix_rbf_mmd2_joint_1_feature_1_label']
        #self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_regress_2_feature']=dop['mix_rbf_mmd2_joint_regress_2_feature']
        
        
        
        
        #self.sigma_list_total
        #self.sigma_list_labelled
        
        #cur_batch_features, cur_batch_label = labelled_batch
        # split into features and label for this instance
        # instantiate our ancestor dict that will be used for current batch
        current_ancestor_dict = {}
        # variables where Y is not ancestor
        # append them to current_ancestor_dict
        for k in self.unlabelled_keys:
            # retrieve potential multidm
            associated_names=self.dsc.label_names_alphan[k]
            # retrieve index of feature
            current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
            # put into ancestor dict
            current_ancestor_dict[k] = cur_batch_features[:, current_feature_idx].view((-1, len(current_feature_idx)))  # maintain orig shape
        
        
        features_copy={}
            
        for k in self.conditional_keys:
            associated_names=self.dsc.label_names_alphan[k]
            # retrieve index of feature
            current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
            # put into ancestor dict
            #current_ancestor_dict[k] = cur_batch_features[:, current_feature_idx].view((-1, len(current_feature_idx)))  # maintain orig shape
            
            features_copy[k]=cur_batch_features[:, current_feature_idx].view((-1, len(current_feature_idx)))  # maintain orig shape
            
        # generate label first, and gumbel it
        input_for_y = cur_batch_features[:, self.causes_of_y_idx_dl]
        y_generated = self.dsc_generators[self.labelled_key].forward(input_for_y)
        # y_generated consists of un-normalised probabilities
        # pass thru gumbel to generate labels

        y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated,hard=True,tau=self.temp)
        current_ancestor_dict[self.labelled_key] = y_gumbel_softmax
        
        cur_ancestors_cause_spouse = [current_ancestor_dict[f] for f in self.unlabelled_keys]
        cur_ancestors_cause_spouse = tuple(cur_ancestors_cause_spouse)
        cur_ancestors_cause_spouse = torch.cat(cur_ancestors_cause_spouse, 1)

        # now loop thru rest
        # each k in this instance is for each feature in causal graph
        for k in self.conditional_keys:
            # get our generator please
            k_has_label = self.dsc_generators[k].conditional_on_label
            extra_causes = self.dsc_generators[k].conditional_feature_names
            # get the idx of causes
            if k_has_label:
                cur_feature_inputs = extra_causes +[self.labelled_key]
            else:
                cur_feature_inputs = extra_causes
            # perform list comprehension to extract variables
            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
            # then concatenate the variables
            generator_input = torch.cat(cur_feature_inputs_lc, 1)
            # then add some noise
            self.noise_placeholder_val.normal_()
            
            #current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device=generator_input.device)


            # then concatenate the noise
            generator_input_w_noise = torch.cat((self.noise_placeholder_val[:generator_input.shape[0]], generator_input), 1)
            # then predict
            predicted_value = self.dsc_generators[k].forward(generator_input_w_noise)
            # and put this one into ancestor dict
            current_ancestor_dict[k] = predicted_value

        #get ancestors of current variable
        cur_ancestors = [current_ancestor_dict[f] for f in self.all_feat_names]
        cur_ancestors = tuple(cur_ancestors)
        batch_estimate_features = torch.cat(cur_ancestors, 1)
        batch_estimate_label=current_ancestor_dict[self.labelled_key]
        
        
        cur_ancestors_label = current_ancestor_dict[self.labelled_key] #for f in self.conditional_keys]











        
        cur_ancestors_effect_p = [current_ancestor_dict[f] for f in self.conditional_keys]
        cur_ancestors_effect_p = tuple(cur_ancestors_effect_p)
        cur_ancestors_effect_p = torch.cat(cur_ancestors_effect_p, 1)
        
        
        cur_batch_features_effect = [features_copy[f] for f in self.conditional_keys]
        cur_batch_features_effect = tuple(cur_batch_features_effect)
        cur_batch_features_effect = torch.cat(cur_batch_features_effect, 1)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        #mrb_joint=self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_1_feature_1_label']
        
        mrb_joint_lab=self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_regress_2_feature_1_label']#=dop['mix_rbf_mmd2_joint_regress_2_feature_1_label']
        mrb_two_feat=self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_regress_2_feature']#=dop['mix_rbf_mmd2_joint_regress_2_feature_1_label']
        
        #self.sigma_list_cs=torch.tensor([self.median_pwd_dict['cause_spouse'] * i for i in [0.125, 0.25, 0.5, 1, 2]],device=torch.device('cuda'))

        #self.sigma_list_e=torch.tensor([self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]],device=torch.device('cuda'))

 
        
        

        
        
        
        
        
        
        
        
        
        
        
        #self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2_joint_regress_2_feature']
        
        #sigma_list_total = 
        
        
        #cur_batch_label_one_hot=torch.nn.functional.one_hot(cur_batch_label.long(),2)
        # pull out features in current batch as given by data loader

        #sigma_list = [self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
        
        if batch_estimate_features.shape[0]>11000:
            
            rp=torch.randperm(batch_estimate_features.shape[0])[:11000]
            
            loss_val_lab=mrb_joint_lab(
            cur_ancestors_effect_p[rp],
            cur_batch_features_effect[rp],
            cur_ancestors_cause_spouse[rp],
            cur_ancestors_cause_spouse[rp],
            batch_estimate_label[rp],
            cur_ancestors_label[rp],
            self.sigma_list_e,self.sigma_list_cs).clone().detach()
            #mix_rbf_mmd2
            # loss_val_lab = mrb_joint(batch_estimate_features[rp],
            #                                 cur_batch_features[rp],
            #                                 y_gumbel_softmax[rp],
            #                                 cur_batch_label_one_hot[rp], self.sigma_list_total).detach()

        else:
            
            loss_val_lab=mrb_joint_lab(
            cur_ancestors_effect_p,
            cur_batch_features_effect,
            cur_ancestors_cause_spouse,
            cur_ancestors_cause_spouse,
            batch_estimate_label,
            cur_ancestors_label,
            self.sigma_list_e,
            self.sigma_list_cs).clone().detach()

            # loss_val_lab = mrb_joint(batch_estimate_features,
            #                                 cur_batch_features,
            #                                 y_gumbel_softmax,
            #                                 cur_batch_label_one_hot, self.sigma_list_total).detach()
            
            
            #loss_val_lab = mrb_joint_lab()

        #labelled_val_losses.append(loss_val_lab)

        
        #loss_lab_bce=torch.tensor([-1],device=cur_batch_label_one_hot.device)
        
        from IPython.core.debugger import set_trace
        
        ##set_trace()
        loss_lab_bce=self.cb(y_generated,cur_batch_label_one_hot)



    
        #unlabelled_batch=batch['unlabelled'][0]#.squeeze(0)
        



        #unlabelled_val_losses=[]
        cur_batch = unlabelled_batch
        current_ancestor_dict = {}
        # variables where Y is not ancestor
        # append them to current_ancestor_dict
        for k in self.unlabelled_keys:
            associated_names = self.dsc.label_names_alphan[k]
            # retrieve index of feature
            current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
            # put into ancestor dict
            current_ancestor_dict[k] = cur_batch[:, current_feature_idx].view(
                (-1, self.dsc.feature_dim))  # maintain orig shape
        
        
        features_copy={}
            
        for k in self.conditional_keys:
            associated_names=self.dsc.label_names_alphan[k]
            # retrieve index of feature
            current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
            # put into ancestor dict
            #current_ancestor_dict[k] = cur_batch_features[:, current_feature_idx].view((-1, len(current_feature_idx)))  # maintain orig shape
            
            features_copy[k]=cur_batch[:, current_feature_idx].view((-1, len(current_feature_idx)))  # maintain orig shape
            
        
        
        
        
        
        
        
        
        
        
        # generate label first, and gumbel it
        input_for_y = cur_batch[:, self.causes_of_y_idx_dl]
        y_generated = self.dsc_generators[self.labelled_key].forward(input_for_y)
        y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=True,tau=self.temp)#,tau=self.temp)
        # now put into ancestor dictionary
        current_ancestor_dict[self.labelled_key] = y_gumbel_softmax
        
        
        

        # now loop thru rest
        # each k in this instance is for each feature in causal graph
        for k in self.conditional_keys:
            # get our generator please
            k_has_label = self.dsc_generators[k].conditional_on_label
            extra_causes = self.dsc_generators[k].conditional_feature_names
            # get the idx of causes...
            if k_has_label:
                cur_feature_inputs =  extra_causes + [self.labelled_key]
            else:
                cur_feature_inputs = extra_causes
            # perform list comprehension to extract variables
            cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
            cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
            # then concatenate the variables
            generator_input = torch.cat(cur_feature_inputs_lc, 1)
            # then add some noise
            #current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device=generator_input.device)
            self.noise_placeholder_val.normal_()

            # then concatenate the noise
            generator_input_w_noise = torch.cat((self.noise_placeholder_val[:generator_input.shape[0]], generator_input), 1)

            # then predict
            predicted_value = self.dsc_generators[k].forward(generator_input_w_noise)
            # and put this one into ancestor dict
            current_ancestor_dict[k] = predicted_value
            # and pause here for my lord
            # calculate joint mmd between observable X and generated X
            # pull out features from current_ancestor_dict
        cur_ancestors = [current_ancestor_dict[f] for f in self.all_feat_names]
        cur_ancestors = tuple(cur_ancestors)

        batch_estimate = torch.cat(cur_ancestors,1)
        # pull out features in current batch as given by data loader
        #sigma_list = [self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
        #mrb_single=self.dict_of_mix_rbf_mmd2_joint_regress['basic_functions']['mix_rbf_mmd2']
        
        
        
        
        
        



        cur_ancestors_cause_spouse = [current_ancestor_dict[f] for f in self.unlabelled_keys]
        cur_ancestors_cause_spouse = tuple(cur_ancestors_cause_spouse)
        cur_ancestors_cause_spouse = torch.cat(cur_ancestors_cause_spouse, 1)

        
        cur_ancestors_effect_p = [current_ancestor_dict[f] for f in self.conditional_keys]
        cur_ancestors_effect_p = tuple(cur_ancestors_effect_p)
        cur_ancestors_effect_p = torch.cat(cur_ancestors_effect_p, 1)
        
        
        cur_batch_features_effect = [features_copy[f] for f in self.conditional_keys]
        cur_batch_features_effect = tuple(cur_batch_features_effect)
        cur_batch_features_effect = torch.cat(cur_batch_features_effect, 1)
        
        
        
        
        
        
        idx_along=torch.arange(batch_estimate.shape[0]).split(1000)
        
        
        
        vul=[]
        
        
        for idx in idx_along:
            loss_val_ulab=mrb_two_feat(cur_ancestors_effect_p[idx],
                                        cur_batch_features_effect[idx],
                                        
                                        
                                        cur_ancestors_cause_spouse[idx],
                                        cur_ancestors_cause_spouse[idx],
                                        
                                        self.sigma_list_e,self.sigma_list_cs).detach()
            
            vul.append(loss_val_ulab.clone())
            
        loss_val_ulab=torch.hstack(vul).flatten().mean()
        # if batch_estimate.shape[0]>11000:
            
        #     rp=torch.randperm(batch_estimate.shape[0])[:11000]
            
            
        #     loss_val_ulab=mrb_two_feat(cur_ancestors_effect_p[rp],
        #                                cur_batch_features_effect[rp],
                                       
                                       
        #                                cur_ancestors_cause_spouse[rp],
        #                                cur_ancestors_cause_spouse[rp],
                                       
        #                                self.sigma_list_e,self.sigma_list_cs).detach()
            
            
            
        
        #     #loss_val_ulab = mrb_single(batch_estimate[rp], cur_batch[rp], self.sigma_list_total).detach()
            
        # else:
            
            
            
            
        #     loss_val_ulab=mrb_two_feat(cur_ancestors_effect_p,
        #                                cur_batch_features_effect,
        #                                cur_ancestors_cause_spouse,
        #                                cur_ancestors_cause_spouse,
        #                                self.sigma_list_e,self.sigma_list_cs).detach()
            
            
            
            
            
            
            
            
            
            
            
            #loss_val_ulab = mrb_single(batch_estimate, cur_batch, self.sigma_list_total).detach()
        
        
        self.log('loss_lab_bce',loss_lab_bce)
        self.log('validation_bce_loss',loss_lab_bce)
        self.log('loss_val_lab',loss_val_lab)
        self.log('loss_val_ulab',loss_val_ulab)
        
        self.log('labelled_bce_and_labelled_mmd',loss_lab_bce+loss_val_lab)
        #self.log('labelled_bce_and_all_feat_mmd',loss_lab_bce+loss_val_ulab)
        self.log('labelled_bce_and_all_feat_mmd',loss_lab_bce+loss_val_ulab+loss_val_lab)
        
        
        
        #self.log('labelled_bce_and_labelled_mmd',loss_lab_bce)
        #self.log('labelled_bce_and_all_feat_mmd',torch.tensor(0.0))
        
        #-----------------------------------------
        
        # for saving fn later on ...
        
        self.log_sidn() #doesn't do it...
                

        
    def log_sidn(self):
        self.log("s_i",self.hparams.s_i)
        self.log("d_n",self.hparams.dn_log)
        
        
    
    # def validation_step(self,batch,batch_idx):
        
        
        
                
                
            
        
    #     labelled_val_losses=[]
    #     # code go here...
        
        
    #     cur_batch_features = batch['labelled'][0]#.squeeze(0)
        
    #     cur_batch_label = batch['labelled'][1]#.squeeze(0)
        
        
    #     #cur_batch_features, cur_batch_label = labelled_batch
    #     # split into features and label for this instance
    #     # instantiate our ancestor dict that will be used for current batch
    #     current_ancestor_dict = {}
    #     # variables where Y is not ancestor
    #     # append them to current_ancestor_dict
    #     for k in self.unlabelled_keys:
    #         # retrieve potential multidm
    #         associated_names=self.dsc.label_names_alphan[k]
    #         # retrieve index of feature
    #         current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
    #         # put into ancestor dict
    #         current_ancestor_dict[k] = cur_batch_features[:, current_feature_idx].reshape((-1, len(current_feature_idx)))  # maintain orig shape
    #     # generate label first, and gumbel it
    #     input_for_y = cur_batch_features[:, self.causes_of_y_idx_dl]
    #     y_generated = self.dsc_generators[self.labelled_key].forward(input_for_y)
    #     # y_generated consists of un-normalised probabilities
    #     # pass thru gumbel to generate labels

    #     y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated,hard=True,tau=self.temp)
    #     current_ancestor_dict[self.labelled_key] = y_gumbel_softmax

    #     # now loop thru rest
    #     # each k in this instance is for each feature in causal graph
    #     for k in self.conditional_keys:
    #         # get our generator please
    #         k_has_label = self.dsc_generators[k].conditional_on_label
    #         extra_causes = self.dsc_generators[k].conditional_feature_names
    #         # get the idx of causes
    #         if k_has_label:
    #             cur_feature_inputs = extra_causes +[self.labelled_key]
    #         else:
    #             cur_feature_inputs = extra_causes
    #         # perform list comprehension to extract variables
    #         cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
    #         cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
    #         # then concatenate the variables
    #         generator_input = torch.cat(cur_feature_inputs_lc, 1)
    #         # then add some noise
    #         current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device=generator_input.device)

    #         # if has_gpu:
    #         #     current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device=generator_input.device)
    #         # else:
    #         #     current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim))

    #         #current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim))
    #         # then concatenate the noise
    #         generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
    #         # then predict
    #         predicted_value = self.dsc_generators[k].forward(generator_input_w_noise)
    #         # and put this one into ancestor dict
    #         current_ancestor_dict[k] = predicted_value

    #     #get ancestors of current variable
    #     cur_ancestors = [current_ancestor_dict[f] for f in self.all_feat_names]
    #     cur_ancestors = tuple(cur_ancestors)
    #     batch_estimate_features = torch.cat(cur_ancestors, 1)
    #     batch_estimate_label=current_ancestor_dict[self.labelled_key]
        
        
        
    #     cur_batch_label_one_hot=torch.nn.functional.one_hot(cur_batch_label.long(),2)
    #     # pull out features in current batch as given by data loader

    #     sigma_list = [self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
        
    #     if batch_estimate_features.shape[0]>11000:
            
    #         rp=torch.randperm(batch_estimate_features.shape[0])[:11000]
            
    #         loss_val_lab = mix_rbf_mmd2_joint(batch_estimate_features[rp],
    #                                         cur_batch_features[rp],
    #                                         y_gumbel_softmax[rp],
    #                                         cur_batch_label_one_hot[rp], sigma_list=sigma_list)

    #     else:

    #         loss_val_lab = mix_rbf_mmd2_joint(batch_estimate_features,
    #                                         cur_batch_features,
    #                                         y_gumbel_softmax,
    #                                         cur_batch_label_one_hot, sigma_list=sigma_list)

    #     labelled_val_losses.append(loss_val_lab)

    #     # # get loss on individual feature variables...
    #     # feature_y_losses = []
    #     # for c in self.unlabelled_keys:
    #     #     associated_names = self.dsc.label_names_alphan[c]
    #     #     # retrieve index of feature
    #     #     current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
    #     #     # put into ancestor dict
    #     #     current_batch_ground_truth = cur_batch_features[:, current_feature_idx].reshape(
    #     #         (-1, self.dsc.feature_dim))
    #     #     # get sigma list for current variable
    #     #     sigma_list = [self.median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
    #     #     # calculate mmd loss
            
    #     #     if current_ancestor_dict[c].shape[0]>11000:
                
    #     #         rp=torch.randperm(current_ancestor_dict[c].shape[0])[:11000]
                
    #     #         mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c][rp],
    #     #                                 current_batch_ground_truth[rp],
    #     #                                 y_gumbel_softmax[rp],
    #     #                                 cur_batch_label_one_hot[rp],
    #     #                                 sigma_list=sigma_list)
                
    #     #     else:
            
    #     #         mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c],
    #     #                                 current_batch_ground_truth,
    #     #                                 y_gumbel_softmax,
    #     #                                 cur_batch_label_one_hot,
    #     #                                 sigma_list=sigma_list)
    #     #     feature_y_losses.append(mloss)

    #     # for c in self.conditional_keys:
    #     #     associated_names = self.dsc.label_names_alphan[c]
    #     #     # retrieve index of feature
    #     #     current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
    #     #     # put into ancestor dict
    #     #     current_batch_ground_truth = cur_batch_features[:, current_feature_idx].reshape(
    #     #         (-1, self.dsc.feature_dim))
    #     #     # get sigma list for current variable
    #     #     sigma_list = [self.median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
    #     #     # calculate mmd loss
    #     #     if current_ancestor_dict[c].shape[0]>11000:
                
    #     #         rp=torch.randperm(current_ancestor_dict[c].shape[0])[:11000]
                
    #     #         mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c][rp],
    #     #                                 current_batch_ground_truth[rp],
    #     #                                 y_gumbel_softmax[rp],
    #     #                                 cur_batch_label_one_hot[rp],
    #     #                                 sigma_list=sigma_list)
    #     #     else:
    #     #         mloss = mix_rbf_mmd2_joint(current_ancestor_dict[c],
    #     #                                 current_batch_ground_truth,
    #     #                                 y_gumbel_softmax,
    #     #                                 cur_batch_label_one_hot,
    #     #                                 sigma_list=sigma_list)
    #     #     feature_y_losses.append(mloss)

    #     # for m in feature_y_losses:
    #     #     labelled_val_losses.append(m)

    #     #GET OUR TRAINING DATA, LABELLED  FOR Y
        
    #     # outputs=dict(feature_y_losses=feature_y_losses,
    #     #              labelled_val_losses=labelled_val_losses,
    #     #              unlabelled_val_losses=unlabelled_val_losses,
    #     #              cur_batch_label_one_hot=cur_batch_label_one_hot,
    #     #              y_generated=y_generated,
    #     #              loss_val_ulab=loss_val_ulab,
    #     #              loss_val_lab=loss_val_lab)
        
            
    #     loss_lab_bce=self.cb(y_generated,cur_batch_label_one_hot.half())
    #     #loss_val_lab=loss_val_lab
        
    
    
    #     unlabelled_batch=batch['unlabelled'][0]#.squeeze(0)
        
    #     #unlabelled_batch = batch['unlabelled'][0]
        
        
    #     # mmd loss on unlabelled features in training set, not using label
    #     # features drawn from labelled + unlabelled data
    



    #     unlabelled_val_losses=[]
    #     cur_batch = unlabelled_batch
    #     current_ancestor_dict = {}
    #     # variables where Y is not ancestor
    #     # append them to current_ancestor_dict
    #     for k in self.unlabelled_keys:
    #         associated_names = self.dsc.label_names_alphan[k]
    #         # retrieve index of feature
    #         current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
    #         # put into ancestor dict
    #         current_ancestor_dict[k] = cur_batch[:, current_feature_idx].reshape(
    #             (-1, self.dsc.feature_dim))  # maintain orig shape
    #     # generate label first, and gumbel it
    #     input_for_y = cur_batch[:, self.causes_of_y_idx_dl]
    #     y_generated = self.dsc_generators[self.labelled_key].forward(input_for_y)
    #     y_gumbel_softmax = torch.nn.functional.gumbel_softmax(y_generated, hard=True,tau=self.temp)#,tau=self.temp)
    #     # now put into ancestor dictionary
    #     current_ancestor_dict[self.labelled_key] = y_gumbel_softmax

    #     # now loop thru rest
    #     # each k in this instance is for each feature in causal graph
    #     for k in self.conditional_keys:
    #         # get our generator please
    #         k_has_label = self.dsc_generators[k].conditional_on_label
    #         extra_causes = self.dsc_generators[k].conditional_feature_names
    #         # get the idx of causes...
    #         if k_has_label:
    #             cur_feature_inputs =  extra_causes + [self.labelled_key]
    #         else:
    #             cur_feature_inputs = extra_causes
    #         # perform list comprehension to extract variables
    #         cur_feature_inputs_lc = [current_ancestor_dict[f] for f in cur_feature_inputs]
    #         cur_feature_inputs_lc = tuple(cur_feature_inputs_lc)
    #         # then concatenate the variables
    #         generator_input = torch.cat(cur_feature_inputs_lc, 1)
    #         # then add some noise
    #         current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device=generator_input.device)

    #         # if has_gpu:
    #         #     current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim),device='cuda:0')
    #         # else:
    #         #     current_noise = torch.randn((generator_input.shape[0], self.dsc.feature_dim))

            
    #         # then concatenate the noise
    #         generator_input_w_noise = torch.cat((current_noise, generator_input), 1)
    #         # then predict
    #         predicted_value = self.dsc_generators[k].forward(generator_input_w_noise)
    #         # and put this one into ancestor dict
    #         current_ancestor_dict[k] = predicted_value
    #         # and pause here for my lord
    #         # calculate joint mmd between observable X and generated X
    #         # pull out features from current_ancestor_dict
    #     cur_ancestors = [current_ancestor_dict[f] for f in self.all_feat_names]
    #     cur_ancestors = tuple(cur_ancestors)
    #     #batch_estimate_features = torch.cat(cur_ancestors, 1)
    #     #batch_estimate_label = current_ancestor_dict[self.labelled_key]
    #     #cur_batch_label_one_hot = torch.nn.functional.one_hot(cur_batch_label.long(), 2)
    #     # pull out features in current batch as given by data loader
    #     # try modify median pwd
    #     ###cur_median_pwd = get_median_pwd(cur_batch_features)
    #     # sigma_list = [total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
    #     batch_estimate = torch.cat(cur_ancestors,1)
    #     # pull out features in current batch as given by data loader
    #     sigma_list = [self.total_median_pwd * i for i in [0.125, 0.25, 0.5, 1, 2]]
        
    #     if batch_estimate.shape[0]>11000:
    #         rp=torch.randperm(batch_estimate.shape[0])[:11000]
        
    #         loss_val_ulab = mix_rbf_mmd2(batch_estimate[rp], cur_batch[rp], sigma_list)
            
    #     else:
    #         loss_val_ulab = mix_rbf_mmd2(batch_estimate, cur_batch, sigma_list)
            

    #     # ...log the running loss
    #     #writer.add_scalar('validation_unlabelled_mmd_loss',loss_val_ulab, t_iter)

    #     # unlabelled_val_losses.append(loss_val_ulab)

    #     # # get loss on individual feature variables...
    #     # individual_feat_mmd_losses = []
    #     # for c in self.conditional_keys:
    #     #     associated_names = self.dsc.label_names_alphan[c]
    #     #     # retrieve index of feature
    #     #     current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
    #     #     # put into ancestor dict
    #     #     current_batch_ground_truth = cur_batch[:, current_feature_idx].reshape((-1, self.dsc.feature_dim))
    #     #     # get sigma list for current variable
    #     #     sigma_list = [self.median_pwd_dict[c] * i for i in [0.125, 0.25, 0.5, 1, 2]]
    #     #     # calculate mmd loss
    #     #     if current_ancestor_dict[c].shape[0]>11000:
    #     #         rp=torch.randperm(current_ancestor_dict[c].shape[0])[:11000]
    #     #         mloss = mix_rbf_mmd2(current_ancestor_dict[c][rp], current_batch_ground_truth[rp], sigma_list)
                
    #     #     else:
            
            
    #     #         mloss = mix_rbf_mmd2(current_ancestor_dict[c], current_batch_ground_truth, sigma_list)
                
    #     #     individual_feat_mmd_losses.append(mloss)


    #     # for m in individual_feat_mmd_losses:
    #     #     unlabelled_val_losses.append(m)

    #     # joint_feat_mmd_losses=[]

    #     # #get loss on joint feature variables...
    #     # for ev in self.conditional_keys: #effect of Y
    #     #     for cv in self.unlabelled_keys: #cause of Y
    #     #         #get effect v
    #     #         associated_names = self.dsc.label_names_alphan[ev]
    #     #         # retrieve index of feature
    #     #         current_feature_idx = [self.feature_idx_subdict[k] for k in associated_names]
    #     #         # put into ancestor dict
    #     #         current_batch_ground_truth = cur_batch[:, current_feature_idx].reshape((-1, self.dsc.feature_dim))

    #     #         sigma_list_ev=[self.median_pwd_dict[ev] * i for i in [0.125,0.25,0.5,1,2]]
    #     #         estimate_ev=current_ancestor_dict[ev]
    #     #         true_ev=current_batch_ground_truth

    #     #         #get cause c
    #     #         sigma_list_cv=[self.median_pwd_dict[cv] * i for i in [0.125,0.25,0.5,1,2]]
    #     #         estimate_cv=current_ancestor_dict[cv]
    #     #         true_cv=current_ancestor_dict[cv]
                
                
    #     #         if estimate_ev.shape[0]>11000:
                    
    #     #             rp=torch.randperm(estimate_ev.shape[0])[:11000]
                    
    #     #             joint_mmd_loss =mix_rbf_mmd2_joint_regress( estimate_ev[rp],
    #     #                                                         true_ev[rp],
    #     #                                                         estimate_cv[rp],
    #     #                                                         true_cv[rp],
    #     #                                                         sigma_list=sigma_list_ev,
    #     #                                                         sigma_list1=sigma_list_cv)
    #     #         else:
                    
                

    #     #             joint_mmd_loss =mix_rbf_mmd2_joint_regress( estimate_ev,
    #     #                                                         true_ev,
    #     #                                                         estimate_cv,
    #     #                                                         true_cv,
    #     #                                                         sigma_list=sigma_list_ev,
    #     #                                                         sigma_list1=sigma_list_cv)

    #     #         joint_feat_mmd_losses.append(joint_mmd_loss)

    #     # for m in joint_feat_mmd_losses:
    #     #     unlabelled_val_losses.append(m)

    #     # joint mmd loss on labelled data, including label y
    #     #for batch_idx, d in enumerate(val_dloader_labelled):
        
        
        
    #     self.log('loss_lab_bce',loss_lab_bce)
    #     self.log('validation_bce_loss',loss_lab_bce)
    #     self.log('loss_val_lab',loss_val_lab)
    #     self.log('loss_val_ulab',loss_val_ulab)
    #     self.log('labelled_bce_and_labelled_mmd',loss_lab_bce+loss_val_lab)
    #     self.log('labelled_bce_and_all_feat_mmd',loss_lab_bce+loss_val_ulab)
        
    #     #return(outputs)
        
    #     #-----------------------------------------
        
    #     # for saving fn later on ...
        
    #     self.log("s_i",self.hparams.s_i)
    #     self.log("d_n",self.hparams.dn_log)
        
        
    #     #self.val_outs=[] #reset them...
        
        
        
        
        
        
        
    #     #return(dict(loss_lab_bce=loss_lab_bce,loss_val_lab=loss_val_lab,loss_val_ulab=loss_val_ulab))
    
        
    # def on_validation_batch_end(self,outputs,batch,batch_idx,dataloader_idx):
    #     #https://github.com/Lightning-AI/pytorch-lightning/discussions/11659
        
        
        
    #     self.val_outs.append(outputs)
        
    # def on_validation_epoch_end(self) -> None:
        
    #     #see wha thappens
        
    #     #print('pause here')
    #     #self.val_outs[0].update(self.val_outs[1])
        
    #     #outputs=self.val_outs[0]##(self.val_outs[1])
    #     #loss_lab_bce=self.val_outs[0]['loss_lab_bce']
    #     #loss_val_lab=self.val_outs[0]['loss_val_lab']
    #     #loss_val_ulab=self.val_outs[1]['loss_val_ulab']
        
    #     #labelled_bce_and_labelled_mmd=loss_lab_bce+loss_val_lab
    #     #labelled_bce_and_all_feat_mmd=loss_lab_bce+loss_val_ulab
        
        
    #     self.log('loss_lab_bce',self.val_outs[0]['loss_lab_bce'])
    #     self.log('validation_bce_loss',self.val_outs[0]['loss_lab_bce'])
    #     self.log('loss_val_lab',self.val_outs[0]['loss_val_lab'])
    #     self.log('loss_val_ulab',self.val_outs[1]['loss_val_ulab'])
    #     self.log('labelled_bce_and_labelled_mmd',self.val_outs[0]['loss_lab_bce']+self.val_outs[0]['loss_val_lab'])
    #     self.log('labelled_bce_and_all_feat_mmd',self.val_outs[0]['loss_lab_bce']+self.val_outs[1]['loss_val_ulab'])
        
    #     #return(outputs)
        
    #     #-----------------------------------------
        
    #     # for saving fn later on ...
        
    #     self.log("s_i",self.hparams.s_i)
    #     self.log("d_n",self.hparams.dn_log)
        
        
    #     self.val_outs=[] #reset them...
        
    #     return 
        
    
    
    def configure_optimizers(self):

        all_parameters_labelled = [] # parameters of all generators including generator for Y
        for k in [self.labelled_key] + self.conditional_keys:
            all_parameters_labelled = all_parameters_labelled + list(self.dsc_generators[k].parameters())
        combined_labelled_optimiser = torch.optim.Adam(all_parameters_labelled, lr=self.hparams.lr)
        
        all_parameters_unlabelled = []
        for k in self.conditional_keys:
            all_parameters_unlabelled = all_parameters_unlabelled + list(self.dsc_generators[k].parameters())
        combined_unlabelled_optimiser = torch.optim.Adam(all_parameters_unlabelled, lr=self.hparams.lr)  #reset unlabelled opt

        return combined_labelled_optimiser, combined_unlabelled_optimiser
        
        
    
    
    
import pandas as pd


class GumbelDataModule(pl.LightningDataModule):
    
    
    def __init__(self,  merge_dat,
                        lab_name,
                        lab_bsize: int = 4,
                        tot_bsize: int = 32,
                        nworkers: int = 4,
                        using_resampled_lab=True):
        super().__init__()



        self.save_hyperparameters('lab_bsize','tot_bsize','nworkers')
        

        validation=merge_dat[merge_dat.type=='validation']
        labelled=merge_dat[merge_dat.type=='labelled']
        unlabelled=merge_dat[merge_dat.type=='unlabelled']
        
        
        all_c=[c for c in merge_dat.columns]
        
        feature_names=[c for c in all_c if (c!=lab_name and c!='type')]
        
        val_y, val_x = validation[[lab_name]],validation[feature_names]
        lab_y, lab_x = labelled[[lab_name]],labelled[feature_names]
        ulab_y, ulab_x = unlabelled[[lab_name]],unlabelled[feature_names]
        
        
        
        ##set_trace()
        self.dd={}
        self.dd['val_y']=val_y
        self.dd['val_x']=val_x
        self.dd['lab_y']=lab_y
        self.dd['lab_x']=lab_x
        self.dd['ulab_y']=ulab_y #NOT used
        self.dd['ulab_x']=ulab_x
        
        
        ##set_trace()
        
        for k in self.dd.keys():
            self.dd[k]=torch.from_numpy(self.dd[k].astype(float).values)
        
        #self.all_validation_features=all_validation_features
        
        
        
        self.lab_ulab_feat=torch.cat((self.dd['lab_x'],self.dd['ulab_x']),dim=0)
        
        
        
        
        
        
        
        
        
        
        
        #self.all_validation_label=val_y
        self.using_resampled_lab=False
        #self.all_unlabelled_and_labelled_features=all_unlabelled_and_labelled_features
        
        #self.all_labelled_features=all_labelled_features
        
        #self.all_labelled_label=all_labelled_label


    def setup(self, stage, precision=16):
        
        print('setting up data loader!!!')
        print(f' stage: {stage}')
        print(f'precision: {precision}')
        #validation data
        #set_trace()
        
        #self.lab_ulab_feat=torch.from_numpy(self.all_unlabelled_and_labelled_features.values)
        
        
        
        self.n_lab_ulab_val=self.lab_ulab_feat.shape[0]
        #self.val_lab=self.dd['val_y']
        #self.val_feat=torch.from_numpy(self.all_validation_features.values)
        
        
        
        self.n_lab_val=self.dd['val_y'].shape[0]

        
        # train data
        
        #self.train_set_X_unlabel=torch.from_numpy(self.all_unlabelled_and_labelled_features.values)
        
        
        #self.train_set_X_unlabel_only=torch.from_numpy(self.all_unlabelled_and_labelled_features.values)
        
        
        #s#elf.train_set_Y_label=torch.from_numpy(self.all_validation_label.values)
        ##self.train_set_X_label=torch.from_numpy(self.all_validation_features.values)
        
        
        #n_unlabelled = X_train_ulab.shape[0]
        #n_labelled = X_train_lab.shape[0]
        
        
        
        #d#ummy_label_weights = torch.ones(n_labelled)
        #resampled_i = torch.multinomial(dummy_label_weights, num_samples=n_unlabelled, replacement=True)

        
        
        if int(precision)==16:
            self.train_set_X_unlabel=self.dd['ulab_x'].half().cuda().contiguous()
            self.train_set_Y_label=self.dd['lab_y'].flatten().half().cuda().contiguous()
            self.train_set_X_label=self.dd['lab_x'].half().cuda().contiguous()
            self.val_feat=self.dd['val_x'].half().cuda().contiguous()
            self.val_lab=torch.nn.functional.one_hot(self.dd['val_y'].flatten().long(),num_classes=2).half().cuda().contiguous()
            self.lab_ulab_feat=self.lab_ulab_feat.half().cuda().contiguous()
            
        elif int(precision)==32:
            self.train_set_X_unlabel=self.dd['ulab_x'].float().cuda().contiguous()
            self.train_set_Y_label=self.dd['lab_y'].flatten().float().cuda().contiguous()
            self.train_set_X_label=self.dd['lab_x'].float().cuda().contiguous()
            
            self.val_feat=self.dd['val_x'].float().cuda().contiguous()
            #self.val_lab=self.val_lab.half().contiguous()
            
            self.val_lab=torch.nn.functional.one_hot(self.dd['val_y'].flatten().long(),2).float().cuda().contiguous()
            
            
            
            self.lab_ulab_feat=self.lab_ulab_feat.float().cuda().contiguous()
        
        
        self.train_set_unlabel=torch.utils.data.TensorDataset(self.train_set_X_unlabel)    
        self.train_set_label=torch.utils.data.TensorDataset(self.train_set_X_label,self.train_set_Y_label)
        
        
                
        
        self.val_set_label=torch.utils.data.TensorDataset(self.val_feat,self.val_lab)
        self.val_set_unlabel=torch.utils.data.TensorDataset(self.lab_ulab_feat)
        
        #unsqueeze into a single loader
        
        self.val_set_entire=torch.utils.data.TensorDataset(self.val_feat.unsqueeze(0),self.val_lab.unsqueeze(0),self.lab_ulab_feat.unsqueeze(0))
        

        
        #val_dloader_unlabelled = DataLoader(self.val_set_unlabel,batch_size=self.n_lab_ulab_val,num_workers=self.hparams.nworkers,pin_memory=True,persistent_workers=True)
        #val_dloader_labelled = DataLoader(self.val_set_label,batch_size=self.n_lab_val,num_workers=self.hparams.nworkers,pin_memory=True,persistent_workers=True)

        
        #iterables=dict(labelled=val_dloader_labelled,
        #             unlabelled=val_dloader_unlabelled)
        
        #s#elf.combined_sequential_val=DataLoader(self.val_set_entire, batch_size=1,num_workers=self.hparams.nworkers,persistent_workers=True,shuffle=True,pin_memory=True)
        self.combined_sequential_val=DataLoader(self.val_set_entire, batch_size=1)#,num_workers=self.hparams.nworkers,persistent_workers=True,shuffle=True,pin_memory=True)
        
        
        return self

    def train_dataloader(self):
                
        train_dloader_unlabelled= DataLoader(self.train_set_unlabel,batch_size=self.hparams.tot_bsize,shuffle=True,drop_last=False)#,pin_memory=True,persistent_workers=True,prefetch_factor=4)
        train_dloader_labelled= DataLoader(self.train_set_label,batch_size=self.hparams.lab_bsize,shuffle=True,drop_last=False)
        

        iterables=dict(labelled=train_dloader_labelled,
                        unlabelled=train_dloader_unlabelled)
        #'max_size_cycle'
        #combined_sequential=CombinedLoader(iterables, 'max_size') #have to install nightly to get support for this, ie pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
        combined_sequential=CombinedLoader(iterables, 'max_size_cycle') #have to install nightly to get support for this, ie pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
        
        return(combined_sequential)


        #return(dict(labelled=train_dloader_labelled,
        #            unlabelled=train_dloader_unlabelled))

    def val_dataloader(self):

        return(self.combined_sequential_val)
        
    
    


        
        
        # val_dloader_entire = DataLoader(self.val_set_entire,batch_size=1,shuffle=False,num_workers=self.hparams.nworkers,pin_memory=True,persistent_workers=True)
        
        # return val_dloader_entire
        