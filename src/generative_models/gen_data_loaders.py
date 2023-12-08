from benchmarks_cgan import *


import pytorch_lightning as pl
from typing import Optional
has_gpu=torch.cuda.is_available()
NUM_WORKERS=4




#from pl_bolts.datamodules.async_dataloader import  AsynchronousLoader


class SSLDataModule_Unlabel_X(pl.LightningDataModule):
    def __init__(self,
                 orig_data_df,
                 target_x,
                 batch_size: int = 128):
        super().__init__()
        self.orig_data_df = orig_data_df
        self.batch_size = batch_size
        self.target_x=target_x

    def setup(self, stage: Optional[str] = None):
        odf=self.orig_data_df
        odc=[c for c in odf.columns]
        feature_cols=self.target_x
        print(feature_cols)

        target_x = odf[odf.type.isin(['labelled','unlabelled'])][feature_cols].values.astype(np.float32)
        target_x = torch.Tensor(target_x)
        self.data_train=torch.utils.data.TensorDataset(target_x)

        all_x=target_x.unsqueeze(0)

        #just validation set only
        validation_x=odf[odf.type.isin(['validation'])][feature_cols].values.astype(np.float32)
        validation_x=torch.Tensor(validation_x).unsqueeze(0)



        self.data_validation=torch.utils.data.TensorDataset(validation_x,all_x)


        return(self)



    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)


    def val_dataloader(self):
        #return AsynchronousLoader(DataLoader(self.data_validation, batch_size=1),device=torch.device('cuda'))
        return DataLoader(self.data_validation, batch_size=self.batch_size,num_workers=NUM_WORKERS,persistent_workers=True,pin_memory=True)

# class SSLDataModule_X_from_Y(pl.LightningDataModule):
#     def __init__(self,
#                  orig_data_df,
#                  tvar_name,
#                  cvar_name,
#                  cvar_type,
#                  labelled_batch_size: int = 4,
#                  unlabelled_batch_size: int = 128,

#                 ):
#         super().__init__()
#         self.orig_data_df=orig_data_df
#         self.tvar_name=tvar_name
#         self.cvar_name=cvar_name
#         self.cvar_type=cvar_type
#         self.labelled_batch_size=labelled_batch_size
#         self.unlabelled_batch_size = unlabelled_batch_size

#     def setup(self, stage: Optional[str] = None):

#         orig_data_df=self.orig_data_df

#         #get all unique variables from this one...

#         orig_dcols=[c for c in orig_data_df.columns] #get all columns
#         orig_vcols=[c for c in orig_dcols if c!='type'] #drop 'type' column



#         orig_fcols=[c for c in orig_vcols if 'Y' not in c] #drop label=='Y' column

#         #now split on '_' and take LHS, this will give you the variable name, ie
#         #ie, X2_0 -> X2
#         orig_feat_vars=[o.split('_')[0] for o in orig_fcols]
#         unique_vars=list(set(orig_feat_vars))
#         #derive the conditional X from this one...
#         #conditional_X= [v for v in unique_vars if v!=self.target_feature_label]
#         #now match this back to the colummns in the dataframe..
#         #conditional_cols=[vcol for vcol in orig_dcols if vcol.startswith(tuple(conditional_X))]
#         #get target_cols also
#         #target_cols=[tcol for tcol in orig_dcols if tcol.startswith(self.tvar_name)]

#         #get label col
#         #label_col=[ycol for ycol in orig_dcols if 'Y' in ycol]

#         #label

#         #ok we have now three useful piece of info:

#         #1. target feature columns
#         #2. conditional feature columns
#         #3. label column(s) <- should just be 1



#         # now we are formatting data loaders so that we can: 
#         # ----------#
#         # Training Labelled (features)
#         # ----------#

#         target_x_labelled=orig_data_df[orig_data_df.type=='labelled'][self.tvar_name]
#         #conditional_x_labelled=orig_data_df[orig_data_df.type=='labelled'][conditional_cols]
#         y_labelled=orig_data_df[orig_data_df.type=='labelled'][self.cvar_name]



#         y_labelled_oh=torch.nn.functional.one_hot(torch.Tensor(y_labelled.values).long()).squeeze(1)



#         target_x_labelled=torch.tensor(target_x_labelled.values.astype(np.float32))
        
#         target_y_labelled=torch.tensor(y_labelled_oh)
        
        



#         #self.data_train_labelled=torch.utils.data.TensorDataset(torch.tensor(target_x_labelled.values.astype(np.float32)),torch.tensor(y_labelled_oh))


#         # ----------#
#         # Training Unlabelled (features)
#         # ----------#

#         target_x_unlabelled=orig_data_df[orig_data_df.type.isin(['labelled','unlabelled'])][self.tvar_name]
#         #conditional_x_unlabelled=orig_data_df[orig_data_df.type.isin(['labelled','unlabelled'])][conditional_cols]


#         #now resample,

#         n_labelled=target_x_labelled.shape[0] #nrow of labelled x
#         #set up for multinomial rfesample
#         n_unlabelled=target_x_unlabelled.shape[0]
#         y_resampled_labels=y_labelled.sample(n_unlabelled,replace=True, random_state=1)
#         y_resampled_oh=torch.nn.functional.one_hot(torch.Tensor(y_resampled_labels.values.astype(np.float32)).long()).squeeze(1)


#         target_x_unlabelled=torch.tensor(target_x_unlabelled.values.astype(np.float32))
#         target_y_unlabelled=torch.tensor(y_resampled_oh)
        
        
#         self.x_l=target_x_labelled
#         self.y_l=target_y_labelled



#         self.data_train_labelled=torch.utils.data.TensorDataset(target_x_labelled,target_y_labelled)
#         self.data_train_unlabelled=torch.utils.data.TensorDataset(target_x_unlabelled,target_y_unlabelled)









#         # get validation set


#         valset=orig_data_df[orig_data_df.type=='validation']

#         #pull out y

#         val_target_x=orig_data_df[orig_data_df.type=='validation'][self.tvar_name]

#         #val_conditional_x=orig_data_df[orig_data_df.type=='validation'][conditional_cols]

#         y_val =orig_data_df[orig_data_df.type=='validation'][self.cvar_name]



#         #self.validation_set=torch.utils.data.TensorDataset(target_x_labelled,
#         #                                                     conditional_x_unlabelled,
#          #                                                    y_resampled_labels)



#         # get transduction set
#         valset=orig_data_df[orig_data_df.type=='unlabelled']

#         #pull out y
#         target_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][self.tvar_name]

#         #conditional_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][conditional_cols]
#         y_trans =orig_data_df[orig_data_df.type=='unlabelled'][self.cvar_name]

#         #self.validation_set=torch.utils.data.TensorDataset(target_x_labelled,
#         #                                                     conditional_x_unlabelled,
#         #                                                     y_resampled_labels)


#         # Validation Sets

#         # -------------#
#         # Validation 
#         # -------------#

#         #set_trace()
#         vfeat_target=torch.tensor(val_target_x.values.astype(np.float32)).unsqueeze(0)
#         #vfeat_cond=torch.tensor(val_conditional_x.values).unsqueeze(0)

#         yval_oh=torch.nn.functional.one_hot(torch.Tensor(y_val.values).long()).squeeze(1)
#         vlab=yval_oh.unsqueeze(0)


#         tfeat_target=torch.tensor(target_x_trans.values.astype(np.float32)).unsqueeze(0)
#         #tfeat_cond=torch.tensor(conditional_x_trans.values).unsqueeze(0)
#         ytrans_oh=torch.nn.functional.one_hot(torch.Tensor(y_trans.values).long()).squeeze(1)
#         tlab=ytrans_oh.unsqueeze(0)

#         #set_trace()

#         self.data_validation=torch.utils.data.TensorDataset(vfeat_target,
#                                                             vlab,
#                                                             tfeat_target,
#                                                             tlab)

#         return(self)

#     def train_dataloader(self):
#         labelled_loader=torch.utils.data.DataLoader(self.data_train_labelled,batch_size=self.labelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)
#         unlabelled_loader=torch.utils.data.DataLoader(self.data_train_unlabelled,batch_size=self.unlabelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)


#         loaders = {"loader_labelled":labelled_loader,
#                    "loader_unlabelled":unlabelled_loader}
#         return loaders
        

#         # #labelled_loader=torch.utils.data.DataLoader(self.data_train_labelled,batch_size=self.labelled_batch_size,num_workers=1,persistent_workers=True,shuffle=True,pin_memory=True)
#         # #unlabelled_loader=torch.utils.data.DataLoader(self.data_train_unlabelled,batch_size=self.unlabelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)


#         # # loaders = {"loader_labelled":labelled_loader,
#         # #            "loader_unlabelled":unlabelled_loader}
        
#         # return unlabelled_loader



#     def val_dataloader(self):
#         return DataLoader(self.data_validation, batch_size=1,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)









class SSLDataModule_X_from_Y(pl.LightningDataModule):
    def __init__(self,
                 orig_data_df,
                 tvar_name,
                 cvar_name,
                 cvar_type,
                 labelled_batch_size: int = 4,
                 unlabelled_batch_size: int = 128,

                ):
        super().__init__()
        self.orig_data_df=orig_data_df
        self.tvar_name=tvar_name
        self.cvar_name=cvar_name
        self.cvar_type=cvar_type
        self.labelled_batch_size=labelled_batch_size
        self.unlabelled_batch_size = unlabelled_batch_size

    def setup(self, stage: Optional[str] = None):

        orig_data_df=self.orig_data_df

        #get all unique variables from this one...

        orig_dcols=[c for c in orig_data_df.columns] #get all columns
        orig_vcols=[c for c in orig_dcols if c!='type'] #drop 'type' column



        orig_fcols=[c for c in orig_vcols if 'Y' not in c] #drop label=='Y' column

        #now split on '_' and take LHS, this will give you the variable name, ie
        #ie, X2_0 -> X2
        orig_feat_vars=[o.split('_')[0] for o in orig_fcols]
        unique_vars=list(set(orig_feat_vars))
        #derive the conditional X from this one...
        #conditional_X= [v for v in unique_vars if v!=self.target_feature_label]
        #now match this back to the colummns in the dataframe..
        #conditional_cols=[vcol for vcol in orig_dcols if vcol.startswith(tuple(conditional_X))]
        #get target_cols also
        #target_cols=[tcol for tcol in orig_dcols if tcol.startswith(self.tvar_name)]

        #get label col
        #label_col=[ycol for ycol in orig_dcols if 'Y' in ycol]

        #label

        #ok we have now three useful piece of info:

        #1. target feature columns
        #2. conditional feature columns
        #3. label column(s) <- should just be 1



        # now we are formatting data loaders so that we can: 
        # ----------#
        # Training Labelled (features)
        # ----------#

        target_x_labelled=orig_data_df[orig_data_df.type=='labelled'][self.tvar_name]
        #conditional_x_labelled=orig_data_df[orig_data_df.type=='labelled'][conditional_cols]
        y_labelled=orig_data_df[orig_data_df.type=='labelled'][self.cvar_name]



        y_labelled_oh=torch.nn.functional.one_hot(torch.Tensor(y_labelled.values).long()).squeeze(1)



        target_x_labelled=torch.tensor(target_x_labelled.values.astype(np.float32))
        target_y_labelled=torch.tensor(y_labelled_oh)
        
        



        #self.data_train_labelled=torch.utils.data.TensorDataset(torch.tensor(target_x_labelled.values.astype(np.float32)),torch.tensor(y_labelled_oh))


        # ----------#
        # Training Unlabelled (features)
        # ----------#

        target_x_unlabelled=orig_data_df[orig_data_df.type.isin(['labelled','unlabelled'])][self.tvar_name]
        #conditional_x_unlabelled=orig_data_df[orig_data_df.type.isin(['labelled','unlabelled'])][conditional_cols]


        #now resample,

        n_labelled=target_x_labelled.shape[0] #nrow of labelled x
        #set up for multinomial rfesample
        n_unlabelled=target_x_unlabelled.shape[0]
        y_resampled_labels=y_labelled.sample(n_unlabelled,replace=True, random_state=1)
        y_resampled_oh=torch.nn.functional.one_hot(torch.Tensor(y_resampled_labels.values.astype(np.float32)).long()).squeeze(1)


        target_x_unlabelled=torch.tensor(target_x_unlabelled.values.astype(np.float32))
        target_y_unlabelled=torch.tensor(y_resampled_oh)
        
        
        self.x_l=target_x_labelled
        self.y_l=target_y_labelled



        self.data_train_labelled=torch.utils.data.TensorDataset(target_x_labelled,target_y_labelled)
        self.data_train_unlabelled=torch.utils.data.TensorDataset(target_x_unlabelled)









        # get validation set


        valset=orig_data_df[orig_data_df.type=='validation']

        #pull out y

        val_target_x=orig_data_df[orig_data_df.type=='validation'][self.tvar_name]

        #val_conditional_x=orig_data_df[orig_data_df.type=='validation'][conditional_cols]

        y_val =orig_data_df[orig_data_df.type=='validation'][self.cvar_name]



        #self.validation_set=torch.utils.data.TensorDataset(target_x_labelled,
        #                                                     conditional_x_unlabelled,
         #                                                    y_resampled_labels)



        # get transduction set
        valset=orig_data_df[orig_data_df.type=='unlabelled']

        #pull out y
        target_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][self.tvar_name]

        #conditional_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][conditional_cols]
        y_trans =orig_data_df[orig_data_df.type=='unlabelled'][self.cvar_name]

        #self.validation_set=torch.utils.data.TensorDataset(target_x_labelled,
        #                                                     conditional_x_unlabelled,
        #                                                     y_resampled_labels)


        # Validation Sets

        # -------------#
        # Validation 
        # -------------#

        #set_trace()
        vfeat_target=torch.tensor(val_target_x.values.astype(np.float32)).unsqueeze(0)
        #vfeat_cond=torch.tensor(val_conditional_x.values).unsqueeze(0)

        yval_oh=torch.nn.functional.one_hot(torch.Tensor(y_val.values).long()).squeeze(1)
        vlab=yval_oh.unsqueeze(0)


        tfeat_target=torch.tensor(target_x_trans.values.astype(np.float32)).unsqueeze(0)
        #tfeat_cond=torch.tensor(conditional_x_trans.values).unsqueeze(0)
        ytrans_oh=torch.nn.functional.one_hot(torch.Tensor(y_trans.values).long()).squeeze(1)
        tlab=ytrans_oh.unsqueeze(0)

        #set_trace()

        self.data_validation=torch.utils.data.TensorDataset(vfeat_target,
                                                            vlab,
                                                            tfeat_target,
                                                            tlab)

        return(self)

    def train_dataloader(self):
        #labelled_loader=torch.utils.data.DataLoader(self.data_train_labelled,batch_size=self.labelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)
        unlabelled_loader=torch.utils.data.DataLoader(self.data_train_unlabelled,batch_size=self.unlabelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)

        return unlabelled_loader


    # def train_dataloader(self):
    #     labelled_loader=torch.utils.data.DataLoader(self.data_train_labelled,batch_size=self.labelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)
    #     unlabelled_loader=torch.utils.data.DataLoader(self.data_train_unlabelled,batch_size=self.unlabelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)



    #     loaders = {"loader_labelled":labelled_loader,
    #                "loader_unlabelled":unlabelled_loader}
    #     return loaders


    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=1,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True,pin_memory=True)













        return loaders








class SSLDataModule_Y_from_X(pl.LightningDataModule):
    def __init__(self,
                 orig_data_df,
                 tvar_name,
                 cvar_name,
                 batch_size: int = 128,
                **kwargs):
        super().__init__()

        self.orig_data_df = orig_data_df
        self.tvar_name=tvar_name
        self.cvar_name=cvar_name
        self.batch_size = batch_size



    def setup(self, stage: Optional[str] = None):

        odf=self.orig_data_df


        #subset to labelled cases..
        odf_labelled=odf[odf.type=='labelled']
        x_lab, y_lab = odf_labelled[self.cvar_name], odf_labelled[self.tvar_name]
        #convert to numpy
        x_lab, y_lab = x_lab.values.astype(np.float32), y_lab.values
        #convert to torch tensor
        x_lab, y_lab = torch.Tensor(x_lab), torch.Tensor(y_lab)
        #rename for consistency with earlier syntax
        X_train_lab=x_lab

        y_lab=torch.nn.functional.one_hot(y_lab.long()).squeeze(1)

        y_train_lab=y_lab

        #subset to unlabelled cases..
        odf_unlabelled=odf[odf.type=='unlabelled']
        x_unlab, y_unlab = odf_unlabelled[self.cvar_name], odf_unlabelled[self.tvar_name]
        #convert to numpy
        x_unlab, y_unlab = x_unlab.values.astype(np.float32), y_unlab.values
        #convert to torch tensor
        x_unlab, y_unlab = torch.Tensor(x_unlab), torch.Tensor(y_unlab)



        y_unlab=torch.nn.functional.one_hot(y_unlab.long()).squeeze(1)

        #rename for consistency with earlier syntax
        X_train_ulab=x_unlab
        y_train_ulab=y_unlab






        # ----------#
        # Training Labelled (features)
        # ----------#



        #X_train_lab = orig_data['label_features']
        #y_train_lab = orig_data['label_y']

        # ----------#
        # Training Unlabelled (features)
        # ----------#
        #X_train_ulab = orig_data['unlabel_features']
        #y_train_ulab = orig_data['unlabel_y']


        # cat them togeth
        self.data_train=torch.utils.data.TensorDataset(X_train_lab,y_train_lab)


        # Validation Sets

        # -------------#
        # Validation 
        # -------------#

        #subset to validation cases..
        #odf_validation=odf[odf.type=='validation']
        #use labelled cases instead
        odf_validation = odf[odf.type == 'labelled']
        x_validation, y_validation = odf_validation[self.cvar_name], odf_validation[self.tvar_name]
        #convert to numpy
        x_validation, y_validation = x_validation.values.astype(np.float32), y_validation.values
        #convert to torch tensor
        x_validation, y_validation = torch.Tensor(x_validation), torch.Tensor(y_validation)
        #rename for consistency with earlier syntax

        y_validation=torch.nn.functional.one_hot(y_validation.long()).squeeze(1)

        X_val=x_validation
        y_val=y_validation

        vfeat=X_val.unsqueeze(0)
        vlab=y_val.unsqueeze(0)
        tfeat=X_train_ulab.unsqueeze(0)
        tlab=y_train_ulab.unsqueeze(0)

        self.data_validation=torch.utils.data.TensorDataset(vfeat,vlab,tfeat,tlab)
        self.nval=vfeat.shape[0]

        return(self)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.nval)

class SSLDataModule_X2_from_Y_and_X1(pl.LightningDataModule):
    def __init__(self,
                 orig_data_df,
                 tvar_names,
                 cvar_names,
                 label_var_name,
                 labelled_batch_size: int = 4,
                 unlabelled_batch_size: int = 128,
                 use_bernoulli=False,
                 causes_of_y=[],

                 ):
        super().__init__()
        self.orig_data_df = orig_data_df
        self.labelled_batch_size = labelled_batch_size
        self.unlabelled_batch_size = unlabelled_batch_size
        self.tvar_names=tvar_names
        self.cvar_names=cvar_names
        self.label_var_name=label_var_name
        self.causes_of_y=causes_of_y
        self.use_bernoulli=use_bernoulli


    def setup(self, stage: Optional[str] = None):

        orig_data_df=self.orig_data_df

        # ----------#
        # Training Labelled
        # ----------#
        rs_lab=orig_data_df[orig_data_df.type=='labelled']
        target_x_labelled=rs_lab[self.tvar_names]
        conditional_x_labelled=rs_lab[self.cvar_names]
        y_labelled=rs_lab[self.label_var_name]

        y_labelled_oh=torch.nn.functional.one_hot(torch.Tensor(y_labelled.values).long()).squeeze(1)

        self.data_train_labelled=torch.utils.data.TensorDataset(torch.tensor(target_x_labelled.values.astype(np.float32)),
                                                           torch.tensor(conditional_x_labelled.values.astype(np.float32)),
                                                           torch.tensor(y_labelled_oh))

        # ----------#
        # Training Unlabelled (features)
        # ----------#
        rs_ulab = orig_data_df[orig_data_df.type.isin(['labelled', 'unlabelled'])]
        target_x_unlabelled=rs_ulab[self.tvar_names]
        conditional_x_unlabelled=rs_ulab[self.cvar_names]


        # conditional_x_unlabelled=orig_data_df[orig_data_df.type.isin(['labelled','unlabelled'])][conditional_cols]

        # now resample,

        # set up for multinomial rfesample
        n_unlabelled = target_x_unlabelled.shape[0]
        y_resampled_labels = y_labelled.sample(n_unlabelled, replace=True, random_state=1)
        y_resampled_oh = torch.nn.functional.one_hot(torch.Tensor(y_resampled_labels.values).long()).squeeze(1)

        self.data_train_unlabelled = torch.utils.data.TensorDataset(torch.tensor(target_x_unlabelled.values.astype(np.float32)),
                                                                    torch.tensor(conditional_x_unlabelled.values.astype(np.float32)),
                                                                    torch.tensor(y_resampled_oh))


        #pull out y

        val_target_x=orig_data_df[orig_data_df.type=='validation'][self.tvar_names]
        val_conditional_x=orig_data_df[orig_data_df.type=='validation'][self.cvar_names]
        y_val =orig_data_df[orig_data_df.type=='validation'][self.label_var_name]

        # get transduction set

        valset=orig_data_df[orig_data_df.type=='unlabelled']

        #pull out y

        target_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][self.tvar_names]

        conditional_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][self.cvar_names]

        y_trans =orig_data_df[orig_data_df.type=='unlabelled'][self.label_var_name]




        vfeat_target=torch.tensor(val_target_x.values.astype(np.float32)).unsqueeze(0)
        vfeat_cond=torch.tensor(val_conditional_x.values.astype(np.float32)).unsqueeze(0)

        yval_oh=torch.nn.functional.one_hot(torch.Tensor(y_val.values).long()).squeeze(1)
        vlab=yval_oh.unsqueeze(0)


        tfeat_target=torch.tensor(target_x_trans.values.astype(np.float32)).unsqueeze(0)
        tfeat_cond=torch.tensor(conditional_x_trans.values.astype(np.float32)).unsqueeze(0)
        ytrans_oh=torch.nn.functional.one_hot(torch.Tensor(y_trans.values).long()).squeeze(1)
        tlab=ytrans_oh.unsqueeze(0)



        self.data_validation=torch.utils.data.TensorDataset(vfeat_target,
                                                            vfeat_cond,
                                                            vlab,
                                                            tfeat_target,
                                                            tfeat_cond,
                                                            tlab)

        return(self)

    def train_dataloader(self):
        labelled_loader=torch.utils.data.DataLoader(self.data_train_labelled,batch_size=self.labelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True)
        unlabelled_loader=torch.utils.data.DataLoader(self.data_train_unlabelled,batch_size=self.unlabelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True)


        loaders = {"loader_labelled":labelled_loader,
                   "loader_unlabelled":unlabelled_loader}

        return loaders


    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=1)

class SSLDataModule_X_from_X(pl.LightningDataModule):
    def __init__(self,
                 orig_data_df,
                 tvar_names,
                 cvar_names,
                 cvar_types,
                 labelled_batch_size: int = 4,
                 unlabelled_batch_size: int = 128,
                 ):
        super().__init__()
        self.orig_data_df = orig_data_df
        self.labelled_batch_size = labelled_batch_size
        self.unlabelled_batch_size = unlabelled_batch_size
        self.tvar_names = tvar_names
        self.cvar_names = cvar_names
        self.cvar_types = cvar_types

    def setup(self, stage: Optional[str] = None):

        orig_data_df=self.orig_data_df

        # ----------#
        # Training Unlabelled (features)
        # ----------#

        target_x_unlabelled=orig_data_df[orig_data_df.type.isin(['labelled','unlabelled'])][self.tvar_names]
        conditional_x_unlabelled=orig_data_df[orig_data_df.type.isin(['labelled','unlabelled'])][self.cvar_names]

        self.entire_feature_dataset=torch.utils.data.TensorDataset(torch.tensor(target_x_unlabelled.values.astype(np.float32)),
                                                             torch.tensor(conditional_x_unlabelled.values.astype(np.float32)))

        # get validation set
        val_target_x=orig_data_df[orig_data_df.type=='validation'][self.tvar_names]
        val_conditional_x=orig_data_df[orig_data_df.type=='validation'][self.cvar_names]

        # get transduction set
        target_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][self.tvar_names]
        conditional_x_trans=orig_data_df[orig_data_df.type=='unlabelled'][self.cvar_names]

        vfeat_target=torch.tensor(val_target_x.values.astype(np.float32)).unsqueeze(0)
        vfeat_cond=torch.tensor(val_conditional_x.values.astype(np.float32)).unsqueeze(0)

        tfeat_target=torch.tensor(target_x_trans.values.astype(np.float32)).unsqueeze(0)
        tfeat_cond=torch.tensor(conditional_x_trans.values.astype(np.float32)).unsqueeze(0)

        self.data_validation=torch.utils.data.TensorDataset(vfeat_target,
                                                            vfeat_cond,
                                                            tfeat_target,
                                                            tfeat_cond)

        return(self)

    def train_dataloader(self):
        unlabelled_loader=torch.utils.data.DataLoader(self.entire_feature_dataset,batch_size=self.unlabelled_batch_size,num_workers=NUM_WORKERS,persistent_workers=True,shuffle=True)

        return unlabelled_loader


    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=1)

class X_dataset_resample_Y(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, orig_data_df,
                 tvar_names,
                 cvar_names,
                 label_var_name,
                 use_bernoulli=False,
                 causes_of_y=[]):

        self.orig_data_df = orig_data_df
        self.tvar_names = tvar_names
        self.cvar_names = cvar_names
        self.label_var_name = label_var_name
        self.use_bernoulli=use_bernoulli



        # set target X

        all_x = orig_data_df[orig_data_df.type.isin(['labelled', 'unlabelled'])]  # use both for the moment

        target_x = all_x[self.tvar_names].values.flatten().astype(np.float32)

        # set conditional X
        cond_x = all_x[self.cvar_names].values.astype(np.float32)

        # set labels
        labels = orig_data_df[orig_data_df.type.isin(['labelled'])][
            self.label_var_name].values  # use both for the moment

        # target_x=torch.tensor(target_x)
        # target_x=target_x.reshape((-1,1))

        # tx_shape=target_x.shape #getting shape of x

        # cond_x=torch.tensor(cond_x)
        # cond_x=cond_x.reshape((tx_shape[0],-1))


        #get bernnoulli parameters (if they exist...)

        if 'y_given_x_bp' in orig_data_df.columns:

            y_given_x_bp=orig_data_df[['y_given_x_bp']].values.flatten()

            self.y_given_x_bp=y_given_x_bp
            self.use_bernoulli=use_bernoulli

        labels = labels.flatten()

        self.target_x = target_x
        self.cond_x = cond_x
        self.labels = labels

    def __len__(self):
        return (len(self.target_x))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t_x = self.target_x[idx]
        cond_x = self.cond_x[idx]


        if self.use_bernoulli:
            cur_bp=torch.tensor(self.y_given_x_bp[idx])
            est_label=torch.bernoulli(cur_bp).long()
            label=torch.nn.functional.one_hot(est_label, 2) #convert to one hot
        else:
            label = np.int(np.random.choice(self.labels))
            label=torch.nn.functional.one_hot(torch.tensor(label), 2) #convert to one hot

        sample=(t_x,cond_x,label)

        return sample





