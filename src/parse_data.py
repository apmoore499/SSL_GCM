#pd.DataFrame(s_data,dtype='float').drop(columns=['plcg','pjnk','P38','PKC']).to_csv('s_data_plcg_amean.csv',index=False)


from cdt.data import load_dataset #external library
import numpy as np
import pandas as pd
import igraph
import os
import pickle
from igraph import Graph


############################
#
#       PARSE DATA - parse real data into graph and dataset for plotting / algorithms
#
############################


from numpy.random import default_rng 
rng = default_rng()
import torch
from IPython.core.debugger import set_trace


class ds:
    def __init__(self,adj_mat,labels):
        self.adj_mat=adj_mat
        self.labels=labels
        if type(adj_mat)!='list':
            adj_mat=adj_mat.tolist()
        self.dag=igraph.Graph.Adjacency(adj_mat)
        self.all_vi=[i.index for i in self.dag.vs]
        self.set_vertex_dict()


    def set_vertex_dict(self):
        vertex_dict={}
        vertex_dict['sv_i']={var_i:self.get_source_vi(var_i) for var_i in self.all_vi}
        vertex_dict['tv_i']={var_i:self.get_target_vi(var_i) for var_i in self.all_vi}
        vertex_dict['sv_lab']={var_i:vertex_dict['sv_i'][var_i] for var_i in self.all_vi}
        vertex_dict['tv_lab']={var_i:vertex_dict['tv_i'][var_i] for var_i in self.all_vi}
        self.vertex_dict=vertex_dict
        return(self)
    
    def get_source_vi(self,var_i):
        source_edges = self.dag.es.select(_target=var_i)
        source_vertices = [s_e.source_vertex for s_e in source_edges]
        sv_i = [v.index for v in source_vertices]  # get index of them
        return(sv_i)
    
    def get_target_vi(self,var_i):
        target_edges = self.dag.es.select(_source=var_i)
        target_vertices = [t_e.source_vertex for t_e in target_edges]
        tv_i = [v.index for v in target_vertices]  # get index of them
        return(tv_i)


def parse_graph(input_fn):
    #read in all lines
    with open(input_fn,'r') as f:
        glines=np.array([l.replace('\n','') for l in f.readlines()])

    #Graph Nodes:
    gn_idx=np.where(glines=='Graph Nodes:')[0][0]
    node_names=[e for e in glines[gn_idx+1].split(';')]
    #Graph Edges:
    e_idx=np.where(glines=='Graph Edges:')[0][0]
    edges=[e for e in glines[e_idx+1:]]
    #remove any empty
    edges=[e for e in edges if len(e)>0]
    #split in edge from --> to
    edges_ft=[(e.split('-->')[0].split(' ')[1],e.split('-->')[1].replace(' ','')) for e in edges]
    #now replace node name by idx
    ed_dict={i:k for k,i in enumerate(node_names)}
    #now convert edges_ft to numerical idx
    edges_ft_i=[(ed_dict[f],ed_dict[t]) for f,t in edges_ft]
    #new igraph object
    gd = Graph(directed=True)
    #add vertices
    n_vertices=len(node_names)
    gd.add_vertices(n_vertices)
    gd.add_edges(edges_ft_i)
    out_dag=gd.get_adjacency()
    out_dag=np.array(out_dag.data)
    return node_names,out_dag

def dag_to_df(dag):
    ncol=dag.shape[1]
    dag=dag[-ncol:,:] #the algo will append to file rather than overwrite, so ned to select last ncol elem for square matrix
    dag_from_to=pd.DataFrame(np.where(dag==1)).transpose()
    dag_from_to.columns=['source','target']
    dag_from_to.head()
    dag_vars=np.unique(dag_from_to.values.flatten())
    all_vars=[v for v in range(dag.shape[1])]
    #dag_vars.intersect
    keep_vars=np.intersect1d(dag_vars, all_vars, assume_unique=False)
    #drop column not in DAG
    dag_df=pd.DataFrame(dag)
    dag_df=dag_df.iloc[keep_vars,keep_vars] #the dag
    return(dag_df)

def create_data_class(dag_df,values,label_var,n_labelled=50,n_validation=100,s_i=0,variable_types=None):
    spam_dat=ds(dag_df.values,[c for c in values.columns]) 
    spam_dat.merge_dat=values #set data
    pdat=val_split_real(spam_dat.merge_dat,label_var,n_labelled,n_validation)
    spam_dat.merge_dat=pdat
    spam_dat.class_varname=label_var
    spam_dat.feature_varnames=[c for c in spam_dat.merge_dat.columns if c !=label_var and c!='type']
    spam_dat=subset_tensors_real(spam_dat)
    if variable_types:
        print('setting variable types')
        spam_dat.variable_types=variable_types
    return(spam_dat)


def val_split_real(in_df,label_var,n_labelled,n_validation,enforce_strict=True):

    # split into class1 and class0

    merge_dat_c0=in_df[in_df[label_var]==0]
    merge_dat_c1=in_df[in_df[label_var]==1]

    if enforce_strict: #strict balance of labels
        n_c0=merge_dat_c0.shape[0]
        n_c1=merge_dat_c1.shape[0]
        if n_c0>n_c1:
            #resample c0 
            labelled_c0_idx=rng.choice([i for i in merge_dat_c0.index],n_c1,replace=False).ravel()
            merge_dat_c0=merge_dat_c0.loc[labelled_c0_idx]

        elif n_c1>n_c0:
            #resample
            labelled_c1_idx=rng.choice([i for i in merge_dat_c1.index],n_c0,replace=False).ravel()
            merge_dat_c1=merge_dat_c1.loc[labelled_c1_idx]


        in_df=pd.concat([merge_dat_c0,merge_dat_c1],axis=0,ignore_index=True)

    merge_dat_c0=in_df[in_df[label_var]==0]
    merge_dat_c1=in_df[in_df[label_var]==1]
    
    n_samples=in_df.shape[0]

    list_to_sample=[l for l in range(n_samples)]

    #first get labelled
    c0_idx_remaining=merge_dat_c0.index.to_list()
    c1_idx_remaining=merge_dat_c1.index.to_list()

    ratio_of_c0=merge_dat_c0.shape[0]/(merge_dat_c1.shape[0]+merge_dat_c0.shape[0])

    n_labelled_c0=int(ratio_of_c0*n_labelled)
    n_labelled_c1=n_labelled-n_labelled_c0


    labelled_c0_idx=rng.choice(c0_idx_remaining,n_labelled_c0,replace=False).ravel() #labelled cases #rewrite..
    labelled_c1_idx=rng.choice(c1_idx_remaining,n_labelled_c1,replace=False).ravel() #labelled cases #rewrite..

    c0_idx_remaining=list(set(c0_idx_remaining).difference(set(labelled_c0_idx)))
    c1_idx_remaining=list(set(c1_idx_remaining).difference(set(labelled_c1_idx)))

    #second get validation

    n_val_c0=int(ratio_of_c0*n_validation)
    n_val_c1=n_validation-n_val_c0


    val_c0_idx=rng.choice(c0_idx_remaining,n_val_c0,replace=False).ravel() #labelled cases #rewrite..
    val_c1_idx=rng.choice(c1_idx_remaining,n_val_c1,replace=False).ravel() #labelled cases #rewrite..


    #third derive unlabelled idx as only 3 sets
    unlabel_c0_idx=np.array(list(set(c0_idx_remaining).difference(set(val_c0_idx))))
    unlabel_c1_idx=np.array(list(set(c1_idx_remaining).difference(set(val_c1_idx))))



    all_togeth=np.concatenate([labelled_c0_idx,
    labelled_c1_idx,
    val_c0_idx,
    val_c1_idx,
    unlabel_c0_idx,
    unlabel_c1_idx],axis=0)

    assert(len(set(all_togeth).difference(set(list_to_sample)))==0)


    #get crosstabs....

    n_unlabelled=n_samples-n_labelled-n_validation

    print('labelled pc c0 / c1: {:.3f}\t{:.3f}'.format(len(labelled_c0_idx)/n_labelled,len(labelled_c1_idx)/n_labelled))
    print('val pc c0 / c1: {:.3f}\t{:.3f}'.format(len(val_c0_idx)/n_validation,len(val_c1_idx)/n_validation))
    print('unlabel pc c0 / c1: {:.3f}\t{:.3f}'.format(len(unlabel_c0_idx)/n_unlabelled,len(unlabel_c1_idx)/n_unlabelled))

    #store indices in class
    labelled_i=np.concatenate([labelled_c0_idx,labelled_c1_idx],axis=0)
    validation_i=np.concatenate([val_c0_idx,val_c1_idx],axis=0)
    unlabelled_i=np.concatenate([unlabel_c0_idx,unlabel_c1_idx],axis=0)

    #set data types in pandas df
    in_df['type']='type_not_specified'
    in_df.iloc[labelled_i, in_df.columns.get_loc('type')] = 'labelled'
    in_df.iloc[validation_i, in_df.columns.get_loc('type')] = 'validation'
    in_df.iloc[unlabelled_i, in_df.columns.get_loc('type')] = 'unlabelled'

    crosstabs=pd.crosstab(in_df.type, in_df[label_var])
    print('crosstabs')
    print(crosstabs)

    out_df=in_df

    return(out_df)   

def subset_tensors_real(in_dset):
    all_vars=[c for c in in_dset.merge_dat.columns]
    #in_dset.feature_varnames=[v for v in all_vars if 'X' in v]
    #in_dset.class_varname=[v for v in all_vars if 'Y' in v]
    #labelled
    in_dset.train_label_dataset=in_dset.merge_dat[in_dset.merge_dat['type']=='labelled']
    in_dset.train_unlabel_dataset=in_dset.merge_dat[in_dset.merge_dat['type']=='unlabelled']
    in_dset.val_dataset=in_dset.merge_dat[in_dset.merge_dat['type']=='validation']

    #-----------------------------------
    #split into features and class label
    #-----------------------------------

    #-----------
    # labelled
    #-----------
    in_dset.label_features=torch.Tensor(in_dset.train_label_dataset[in_dset.feature_varnames].values).float()
    in_dset.label_y=torch.Tensor(in_dset.train_label_dataset[in_dset.class_varname].values)

    #-----------
    # unlabelled
    #-----------

    in_dset.unlabel_features=torch.Tensor(in_dset.train_unlabel_dataset[in_dset.feature_varnames].values).float()
    in_dset.unlabel_y=torch.Tensor(in_dset.train_unlabel_dataset[in_dset.class_varname].values)        

    #-----------
    # validation
    #-----------

    in_dset.val_features=torch.Tensor(in_dset.val_dataset[in_dset.feature_varnames].values).float()
    in_dset.val_y=torch.Tensor(in_dset.val_dataset[in_dset.class_varname].values)

    #--------------------
    #convert y to one-hot and squeeze
    #--------------------

    in_dset.label_y=torch.nn.functional.one_hot(in_dset.label_y.type(torch.LongTensor),2).squeeze(1)
    in_dset.unlabel_y=torch.nn.functional.one_hot(in_dset.unlabel_y.type(torch.LongTensor),2).squeeze(1)
    in_dset.val_y=torch.nn.functional.one_hot(in_dset.val_y.type(torch.LongTensor),2).squeeze(1)

    return(in_dset)

def output_dataset(ds_name,data_class,s_i=0):
    current_dn='real_{0}'.format(ds_name)
    cd_dir='dataset_{0}'.format(current_dn)
    os.makedirs('dataset_{0}'.format(current_dn),exist_ok=True)
    data_types=['label_features',
                    'unlabel_features',
                    'val_features',
                    'label_y',
                    'unlabel_y',
                    'val_y']
    #saving all data types ie label/unlabel etc
    for dt in data_types:
        out_tens=getattr(data_class, dt)
        out_fn='{0}/d_n_{1}_s_i_{2}_{3}.pt'.format(cd_dir,current_dn,s_i,dt)
        torch.save(out_tens,out_fn)
    #also save the whole class so can recover hyperparameter etc if need
    print('dataset tensors saved for {0}'.format(ds_name))
    #save it
    with open('{0}/d_n_{1}_s_i_{2}_dataset_class.pickle'.format(cd_dir,current_dn,s_i), 'wb') as file:
        pickle.dump(data_class, file) 
    print('dataset saved to pickle for {0}'.format(ds_name))
    #and make empty folders

    folders_to_create=['saved_models','synthetic_data']

    for f in folders_to_create:
        os.makedirs('{0}/{1}'.format(cd_dir,f),exist_ok=True)
        print('successfully created empty folder for: {0}'.format(f))


def plotting_dag_real(ds,dn):
    print('graph structrure for dataset: {0}'.format(dn))
    retg=igraph.plot(ds.dag, 
    bbox=(0, 0, 1000, 200),
    vertex_label=ds.labels,
    vertex_label_size=5,
    vertex_size=10,
    vertex_color='white',
    edge_arrow_size=0.3,
    vertex_label_dist=5,
    vertex_shape='hidden',
    vertex_label_angle=0.2,             
    layout=ds.dag.layout(layout='circle'))
    
    return(retg)        



if __name__=='__main__':
    os.chdir('feature_selection')

    print('pausing here')

    mek_dict={'dag_fn':'sachs/sachs_mek_log_dag.txt',
                'values_fn':'sachs/sachs_mek_log.csv',
                'label_var':'mek',
                'ds_name':'sachs_mek_log'}

    pka_dict={'dag_fn':'sachs/sachs_pka_log_dag.txt',
                'values_fn':'sachs/sachs_pka_log.csv',
                'label_var':'pka',
                'ds_name':'sachs_pka_log'}

    raf_dict={'dag_fn':'sachs/sachs_raf_log_dag.txt',
                'values_fn':'sachs/sachs_raf_log.csv',
                'label_var':'raf',
                'ds_name':'sachs_raf_log'}

    bcancer_dict={'dag_fn':'bcancer_diagnosis_dag.txt',
                'values_fn':'bcancer_diagnosis.csv',
                'label_var':'diagnosis',
                'ds_name':'bcancer_diagnosis'}


    list_of_dicts=[mek_dict,pka_dict,raf_dict,bcancer_dict]

    list_of_dicts=[]

    N_SAMPLES=100

    for c_dict in list_of_dicts:

        #get dag df
        names,tdag=parse_graph(c_dict['dag_fn'])
        dag_df=dag_to_df(tdag)

        #get values
        values=pd.read_csv(c_dict['values_fn'])

        # need to reorder columns of values dataframe so that
        # matches the adjacency matrix
        values = values[names]

        #set label_var
        label_var=c_dict['label_var']

        #set dataset name
        ds_name=c_dict['ds_name']

        for s_i in range(N_SAMPLES):
            rng = default_rng(s_i)
            data_class=create_data_class(dag_df,values,label_var,n_labelled=10,s_i=s_i)
            print('pausing here')
            output_dataset(ds_name,data_class,s_i)





