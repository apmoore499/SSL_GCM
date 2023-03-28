
from cdt.data import load_dataset
import numpy as np
import pandas as pd
# import networkx as nx
import igraph
import os
import pickle
from igraph import Graph
from numpy.random import default_rng
import torch
import argparse
import cdt
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import copy
from IPython.core.debugger import set_trace
from cdt.data import load_dataset
import torch
import pytorch_lightning as pl


RANDOM_SEED = 101
rng = default_rng(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
pl.seed_everything(RANDOM_SEED)


def parse_graph(input_fn):
    # read in all lines
    with open(input_fn, 'r') as f:
        glines = np.array([l.replace('\n', '') for l in f.readlines()])

    # Graph Nodes:
    gn_idx = np.where(glines == 'Graph Nodes:')[0][0]
    node_names = [e for e in glines[gn_idx + 1].split(';')]

    # Graph Edges:
    e_idx = np.where(glines == 'Graph Edges:')[0][0]
    edges = [e for e in glines[e_idx + 1:]]

    # remove any empty
    edges = [e for e in edges if len(e) > 0]
    # split in edge from --> to
    edges_ft = [(e.split('-->')[0].split(' ')[1], e.split('-->')[1].replace(' ', '')) for e in edges]
    # now replace node name by idx
    ed_dict = {i: k for k, i in enumerate(node_names)}
    # now convert edges_ft to numerical idx
    edges_ft_i = [(ed_dict[f], ed_dict[t]) for f, t in edges_ft]
    # edges_pd=pd.DataFrame(edges_ft)
    # edges_pd.columns=['from','to']
    # edges_pd.replace({'from': ed_dict,'to':ed_dict},inplace=True)
    # new igraph object
    gd = Graph(directed=True)
    # add vertices
    n_vertices = len(node_names)
    gd.add_vertices(n_vertices)
    gd.add_edges(edges_ft_i)
    out_dag = gd.get_adjacency()

    out_dag = np.array(out_dag.data)

    # G=nx.from_pandas_edgelist(edges_pd,'from','to',create_using=nx.DiGraph())
    # out_dag=np.array(nx.adjacency_matrix(G).todense())
    return node_names, out_dag

def dag_to_df(dag):
    ncol = dag.shape[1]
    dag = dag[-ncol:,
          :]  # the algo will append to file rather than overwrite, so ned to select last ncol elem for square matrix
    dag_from_to = pd.DataFrame(np.where(dag == 1)).transpose()
    dag_from_to.columns = ['source', 'target']
    dag_from_to.head()
    dag_vars = np.unique(dag_from_to.values.flatten())
    all_vars = [v for v in range(dag.shape[1])]
    # dag_vars.intersect
    keep_vars = np.intersect1d(dag_vars, all_vars, assume_unique=False)
    # drop column not in DAG
    dag_df = pd.DataFrame(dag)
    dag_df = dag_df.iloc[keep_vars, keep_vars]  # the dag
    return (dag_df)

def create_data_class(dag_df, values, label_var, n_labelled=20, n_validation=20, n_test=20, n_unlabelled=None,variable_types=None,
                      enforce_strict=True):
    print(values.head())

    if enforce_strict:
        print('enforcing strict balancing of labels: p(Y=0)=p(Y=1)=0.5')
        # we need to exactly balance the labels to have proportion 0.5 each
        # so we need to resample according to min of 0 or 1 label
        idx_c0 = values[values[label_var] == 0].index.tolist()
        idx_c1 = values[values[label_var] == 1].index.tolist()
        n_c0 = len(idx_c0)
        n_c1 = len(idx_c1)
        if n_c0 == n_c1:
            dummy = 1
            selected_i = idx_c0 + idx_c1
        elif n_c0 > n_c1:
            print('label 0 > label 1')
            # we have more 0 label, resample these of len==n_c1
            c0_sel = np.random.choice(idx_c0, n_c1, replace=False).tolist()
            selected_i = c0_sel + idx_c1
        elif n_c1 > n_c0:
            print('label 1 > label 0')
            # we have more 1 label, resample these of len==n_c0
            c1_sel = np.random.choice(idx_c1, n_c0, replace=False).tolist()
            selected_i = idx_c0 + c1_sel
        # set_trace()
        values = values.iloc[selected_i].reset_index(drop=True)

    spam_dat = ds(dag_df.values.tolist(), [c for c in values.columns])
    spam_dat.merge_dat = values  # set data
    # set_trace()
    pdat = val_split_real(spam_dat.merge_dat, label_var, n_labelled, n_validation, n_test,n_unlabelled)
    spam_dat.merge_dat = pdat
    spam_dat.class_varname = label_var
    spam_dat.feature_varnames = [c for c in spam_dat.merge_dat.columns if c != label_var and c != 'type']
    spam_dat = subset_tensors_real(spam_dat)
    if variable_types:
        print('setting variable types')
        spam_dat.variable_types = variable_types
    return (spam_dat)

def val_split_real(in_df, label_var, n_labelled, n_validation, n_test,n_unlabelled=None):
    
    # split into class1 and class0

    merge_dat_c0 = in_df[in_df[label_var] == 0]
    merge_dat_c1 = in_df[in_df[label_var] == 1]

    n_samples = in_df.shape[0]

    list_to_sample = [l for l in range(n_samples)]

    # first get labelled
    c0_idx_remaining = merge_dat_c0.index.to_list()
    c1_idx_remaining = merge_dat_c1.index.to_list()

    ratio_of_c0 = merge_dat_c0.shape[0] / (merge_dat_c1.shape[0] + merge_dat_c0.shape[0])

    n_labelled_c0 = int(ratio_of_c0 * n_labelled)
    n_labelled_c1 = n_labelled - n_labelled_c0

    labelled_c0_idx = rng.choice(c0_idx_remaining, n_labelled_c0, replace=False).ravel()  # labelled cases #rewrite..
    labelled_c1_idx = rng.choice(c1_idx_remaining, n_labelled_c1, replace=False).ravel()  # labelled cases #rewrite..

    c0_idx_remaining = list(set(c0_idx_remaining).difference(set(labelled_c0_idx)))
    c1_idx_remaining = list(set(c1_idx_remaining).difference(set(labelled_c1_idx)))

    # second get validation

    n_val_c0 = int(ratio_of_c0 * n_validation)
    n_val_c1 = n_validation - n_val_c0

    val_c0_idx = rng.choice(c0_idx_remaining, n_val_c0, replace=False).ravel()  # labelled cases #rewrite..
    val_c1_idx = rng.choice(c1_idx_remaining, n_val_c1, replace=False).ravel()  # labelled cases #rewrite..

    c0_idx_remaining = list(set(c0_idx_remaining).difference(set(val_c0_idx)))
    c1_idx_remaining = list(set(c1_idx_remaining).difference(set(val_c1_idx)))

    # third get unlabelled

    if n_unlabelled==None:
        n_unlabelled = in_df.shape[0] - n_test - n_labelled - n_validation
    else:
        n_unlabelled=n_unlabelled
    n_ulab_c0 = int(ratio_of_c0 * n_unlabelled)
    n_ulab_c1 = n_unlabelled - n_ulab_c0

    ulab_c0_idx = rng.choice(c0_idx_remaining, n_ulab_c0, replace=False).ravel()  # labelled cases #rewrite..
    ulab_c1_idx = rng.choice(c1_idx_remaining, n_ulab_c1, replace=False).ravel()  # labelled cases #rewrite..

    # fourth get test
    # derive test idx as only 4 sets

    test_c0_idx = np.array(list(set(c0_idx_remaining).difference(set(ulab_c0_idx))))
    test_c1_idx = np.array(list(set(c1_idx_remaining).difference(set(ulab_c1_idx))))

    all_togeth = np.concatenate([labelled_c0_idx,
                                 labelled_c1_idx,
                                 val_c0_idx,
                                 val_c1_idx,
                                 ulab_c0_idx,
                                 ulab_c1_idx,
                                 test_c0_idx,
                                 test_c1_idx], axis=0)

    assert (len(set(all_togeth).difference(set(list_to_sample))) == 0)

    # get crosstabs....

    n_unlabelled = n_samples - n_labelled - n_validation

    print('labelled pc c0 / c1: {:.3f}\t{:.3f}'.format(len(labelled_c0_idx) / n_labelled,
                                                       len(labelled_c1_idx) / n_labelled))
    print('val pc c0 / c1: {:.3f}\t{:.3f}'.format(len(val_c0_idx) / n_validation, len(val_c1_idx) / n_validation))
    print('unlabel pc c0 / c1: {:.3f}\t{:.3f}'.format(len(ulab_c0_idx) / n_unlabelled, len(ulab_c1_idx) / n_unlabelled))
    print('test pc c0 / c1: {:.3f}\t{:.3f}'.format(len(test_c0_idx) / n_test, len(test_c1_idx) / n_test))

    # store indices in class
    labelled_i = np.concatenate([labelled_c0_idx, labelled_c1_idx], axis=0)
    validation_i = np.concatenate([val_c0_idx, val_c1_idx], axis=0)
    unlabelled_i = np.concatenate([ulab_c0_idx, ulab_c1_idx], axis=0)
    test_i = np.concatenate([test_c0_idx, test_c1_idx], axis=0)

    # set data types in pandas df
    in_df['type'] = 'type_not_specified'
    in_df.iloc[labelled_i, in_df.columns.get_loc('type')] = 'labelled'
    in_df.iloc[validation_i, in_df.columns.get_loc('type')] = 'validation'
    in_df.iloc[unlabelled_i, in_df.columns.get_loc('type')] = 'unlabelled'
    in_df.iloc[test_i, in_df.columns.get_loc('type')] = 'test'

    crosstabs = pd.crosstab(in_df.type, in_df[label_var])
    print('crosstabs')
    print(crosstabs)

    out_df = in_df

    return (out_df)

def subset_tensors_real(in_dset):
    all_vars = [c for c in in_dset.merge_dat.columns]
    # in_dset.feature_varnames=[v for v in all_vars if 'X' in v]
    # in_dset.class_varname=[v for v in all_vars if 'Y' in v]
    # labelled
    in_dset.train_label_dataset = in_dset.merge_dat[in_dset.merge_dat['type'] == 'labelled']
    in_dset.train_unlabel_dataset = in_dset.merge_dat[in_dset.merge_dat['type'] == 'unlabelled']
    in_dset.val_dataset = in_dset.merge_dat[in_dset.merge_dat['type'] == 'validation']
    in_dset.test_dataset = in_dset.merge_dat[in_dset.merge_dat['type'] == 'test']

    # -----------------------------------
    # split into features and class label
    # -----------------------------------

    # -----------
    # labelled
    # -----------
    in_dset.label_features = torch.Tensor(in_dset.train_label_dataset[in_dset.feature_varnames].values).float()
    in_dset.label_y = torch.Tensor(in_dset.train_label_dataset[in_dset.class_varname].values)

    # -----------
    # unlabelled
    # -----------

    in_dset.unlabel_features = torch.Tensor(in_dset.train_unlabel_dataset[in_dset.feature_varnames].values).float()
    in_dset.unlabel_y = torch.Tensor(in_dset.train_unlabel_dataset[in_dset.class_varname].values)

    # -----------
    # validation
    # -----------

    in_dset.val_features = torch.Tensor(in_dset.val_dataset[in_dset.feature_varnames].values).float()
    in_dset.val_y = torch.Tensor(in_dset.val_dataset[in_dset.class_varname].values)

    # -----------
    # test
    # -----------

    in_dset.test_features = torch.Tensor(in_dset.test_dataset[in_dset.feature_varnames].values).float()
    in_dset.test_y = torch.Tensor(in_dset.test_dataset[in_dset.class_varname].values)

    # --------------------
    # convert y to one-hot and squeeze
    # --------------------

    in_dset.label_y = torch.nn.functional.one_hot(in_dset.label_y.type(torch.LongTensor), 2).squeeze(1)
    in_dset.unlabel_y = torch.nn.functional.one_hot(in_dset.unlabel_y.type(torch.LongTensor), 2).squeeze(1)
    in_dset.val_y = torch.nn.functional.one_hot(in_dset.val_y.type(torch.LongTensor), 2).squeeze(1)
    in_dset.test_y = torch.nn.functional.one_hot(in_dset.test_y.type(torch.LongTensor), 2).squeeze(1)

    return (in_dset)

def output_dataset(ds_name, data_class, s_i):
    current_dn = 'real_{0}'.format(ds_name)
    cd_dir = 'dataset_{0}'.format(current_dn)

    os.makedirs('dataset_{0}'.format(current_dn), exist_ok=True)
    data_types = ['label_features',
                  'unlabel_features',
                  'val_features',
                  'test_features',
                  'label_y',
                  'unlabel_y',
                  'val_y',
                  'test_y']
    # saving all data types ie label/unlabel etc

    os.makedirs(f'./data/{cd_dir}',exist_ok=True) #make directory if not exist

    extra_dirs=['saved_models','synthetic_data','synthetic_data_gumbel','synthetic_data_marginal']

    for d in extra_dirs:
        os.makedirs(f'./data/{cd_dir}/{d}/', exist_ok=True)  # make directory if not exist
    for dt in data_types:
        out_tens = getattr(data_class, dt)
        out_fn = './data/{0}/d_n_{1}_s_i_{2}_{3}.pt'.format(cd_dir, current_dn, s_i, dt)
        torch.save(out_tens, out_fn)
    # also save the whole class so can recover hyperparameter etc if need
    print('dataset tensors saved for {0}'.format(ds_name))
    # save it
    with open('./data/{0}/d_n_{1}_s_i_{2}_dataset_class.pickle'.format(cd_dir, current_dn, s_i), 'wb') as file:
        pickle.dump(data_class, file)
    print('dataset saved to pickle for {0}'.format(ds_name))

def plotting_dag_real(ds, dn):
    print('graph structrure for dataset: {0}'.format(dn))
    retg = igraph.plot(ds.dag,
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

    return (retg)

def reduce_list(in_list):
    return (sum(in_list, []))

def plot_mb(v_i):
    current_node = all_nodes[v_i]
    mb_graph = s_graph.subgraph(merged_mb_dict[current_node])
    dag_status = nx.is_directed_acyclic_graph(mb_graph)
    # set_trace()

    odeg = mb_graph.out_degree(current_node)
    ideg = mb_graph.in_degree(current_node)
    plt.figure()
    plt.title(f'Node={current_node}   DAG={dag_status}  In Deg={ideg}  Out Deg={odeg}')
    pos = nx.circular_layout(mb_graph)
    nx.draw(mb_graph, with_labels=True, pos=pos)
    # plt.show()

def plot_mb_name(current_node):
    mb_graph = s_graph.subgraph(merged_mb_dict[current_node])
    plt.figure()
    plt.title(current_node)
    pos = nx.circular_layout(mb_graph)
    nx.draw(mb_graph, with_labels=True, pos=pos)
    # plt.show()

def get_mb_attributes(s_graph):
    # for each of the vertices

    v_n = [s for s in s_graph.nodes()]

    # get markov blanket
    mb = [s_graph.subgraph(merged_mb_dict[v]) for v in v_n]

    # dag status
    dag_status = [nx.is_directed_acyclic_graph(m) for m in mb]

    # in degree

    in_degree = [m.in_degree(v) for m, v in zip(mb, v_n)]

    # out degree

    out_degree = [m.out_degree(v) for m, v in zip(mb, v_n)]

    summary_df = pd.DataFrame([v_n, dag_status, in_degree, out_degree]).transpose()

    summary_df.columns = ['vertex', 'is_dag', 'in_degree', 'out_degree']

    return (summary_df)



class ds:
    def __init__(self, adj_mat, labels):
        self.adj_mat = adj_mat
        self.labels = labels
        self.dag = igraph.Graph.Adjacency(self.adj_mat)
        self.all_vi = [i.index for i in self.dag.vs]
        self.set_vertex_dict()
        # self.variable_types=variable_types
        # self.feature_dim=feature_dim

    def set_vertex_dict(self):
        vertex_dict = {}
        vertex_dict['sv_i'] = {var_i: self.get_source_vi(var_i) for var_i in self.all_vi}
        vertex_dict['tv_i'] = {var_i: self.get_target_vi(var_i) for var_i in self.all_vi}
        vertex_dict['sv_lab'] = {var_i: vertex_dict['sv_i'][var_i] for var_i in self.all_vi}
        vertex_dict['tv_lab'] = {var_i: vertex_dict['tv_i'][var_i] for var_i in self.all_vi}
        self.vertex_dict = vertex_dict
        return (self)

    def get_source_vi(self, var_i):
        source_edges = self.dag.es.select(_target=var_i)
        source_vertices = [s_e.source_vertex for s_e in source_edges]
        sv_i = [v.index for v in source_vertices]  # get index of them
        return (sv_i)

    def get_target_vi(self, var_i):
        target_edges = self.dag.es.select(_source=var_i)
        target_vertices = [t_e.source_vertex for t_e in target_edges]
        tv_i = [v.index for v in target_vertices]  # get index of them
        return (tv_i)



#-----------------------------------
#     SACHS DATASET
#-----------------------------------


# load it
s_data, s_graph = load_dataset('sachs')

# get our positions
pos = nx.circular_layout(s_graph)

pdict = {
    'PIP2': (-4, -4),
    'PKC': (3, 5),
    'plcg': (-4, 0),
    'PIP3': (-2, -2),
    'pjnk': (0, 2),
    'P38': (3, 2),
    'PKA': (3, -1),
    'praf': (6, 2),
    'pmek': (6, -1),
    'p44/42': (6, -4),
    'pakts473': (3, 7)
}

# drawing the graph if we want
# nx.draw(s_graph, with_labels=True, pos=pdict)


sachs_nconvert_dict = {
    'PIP2': 'pip2',
    'PKC': 'pkc',
    'plcg': 'plc',
    'PIP3': 'pip3',
    'pjnk': 'jnk',
    'P38': 'p38',
    'PKA': 'pka',
    'praf': 'raf',
    'pmek': 'mek',
    'p44/42': 'erk',
    'pakts473': 'akt'
}

# relabel graph nodes
nx.relabel.relabel_nodes(s_graph, sachs_nconvert_dict, copy=False)  # modify in place
# relabel data frame
s_data.rename(columns=sachs_nconvert_dict, inplace=True)
all_nodes = [n for n in s_graph.nodes]


mb_dict = {} # getting markov blanket
for n in s_graph.nodes:
    mb_dict[n] = {}
    # get parents
    mb_dict[n]['parent'] = [n for n in s_graph.predecessors(n)]
    # get children
    mb_dict[n]['children'] = [n for n in s_graph.successors(n)]
    # spouses
    mb_dict[n]['spouses'] = reduce_list([[s for s in s_graph.predecessors(c)] for c in mb_dict[n]['children']])

merged_mb_dict = {}
# new names of dag var: ['akt','erk',  'mek', 'raf', 'pka']
# should obtain names for these only
for k in mb_dict.keys():
    merged_mb_dict[k] = list(set(mb_dict[k]['parent'] + mb_dict[k]['children'] + mb_dict[k]['spouses'] + [k]))

mb_a = get_mb_attributes(s_graph)
selected_var = mb_a[(mb_a.is_dag == True) & (mb_a.out_degree > 0)]

dag_var = selected_var.vertex.values

# just use these ones for now:
mb_out_names = dag_var

for n in mb_out_names:
    mb_graph = s_graph.subgraph(merged_mb_dict[n])
    mb_data = s_data[merged_mb_dict[n]]

for n in mb_out_names:
    print('writing for node: {0}'.format(n))

    mb_data_fn = 'feature_selection/sachs/sachs_{0}_log.csv'.format(n)
    out_dag_fn = 'feature_selection/sachs/sachs_{0}_log_dag.txt'.format(n)
    fig_out_fn = 'feature_selection/sachs/sachs_{0}_log.png'.format(n)

    mb_graph = s_graph.subgraph(merged_mb_dict[n])
    mb_data = copy.deepcopy(s_data[merged_mb_dict[n]])
    log_names = [c for c in mb_data.columns if n != c]
    log_vars_dict = {n: n}  # target variable is unchanged
    for l in log_names:
        mb_data[l + '_log'] = np.log([v for v in mb_data[l]])
        mb_data.drop(columns=[l], inplace=True)
        log_vars_dict[l] = l + '_log'

    mb_data[n] = (mb_data[n] > np.median(mb_data[n])).astype(int)

    # relabel graph nodes

    mb_graph = nx.relabel.relabel_nodes(mb_graph, log_vars_dict, copy=True)  # modify in place

    # relabel data frame

    mb_data.rename(columns=log_vars_dict, inplace=True)

    # now mb_data is formatted
    # convert graph to txt file
    mb_nodes = [n for n in mb_graph.nodes]
    # form string of node names
    gn_str = 'Graph Nodes:'
    nodes_str = ';'.join(mb_nodes)
    sep_str = ''
    ge_str = 'Graph Edges:'
    gedges = ['{0}. {1} --> {2}'.format(k, e[0], e[1]) for (k, e) in enumerate(mb_graph.edges)]
    total_graph_txt = [gn_str, nodes_str, sep_str, ge_str] + gedges
    # writing data frame out
    mb_data.to_csv(mb_data_fn, index=False)
    print('data frame saved to: {0}'.format(mb_data_fn))
    # writing output of dag lines to a .txt file w graph info
    with open(out_dag_fn, 'w') as f:
        f.write('\n'.join(total_graph_txt))
    print('dag saved to: {0}'.format(out_dag_fn))

    # do a png of the plot

    plot_mb_name(n)
    plt.savefig(fig_out_fn, format="PNG")
    print('image of dag saved to: {0}'.format(fig_out_fn))



for s_i in range(100):
    list_of_sachs_variables=[]#['raf','mek']
    for sv in list_of_sachs_variables:
        dag_fn=f'feature_selection/sachs/sachs_{sv}_log_dag.txt'
        csv_fn=f'feature_selection/sachs/sachs_{sv}_log.csv'
        names,tdag=parse_graph(dag_fn)
        dag_df=dag_to_df(tdag)
        values=pd.read_csv(csv_fn,sep=',')
        keepcols=[n for n in names]
        values=values[keepcols]
        idx_v=[i for i in values.index]
        values.reset_index(inplace=True,drop=True)
        dcols=[c for c in values.columns]
        dcols=[c.replace('/','_') for c in dcols]
        values.columns=dcols
        label_var=sv
        data_class=create_data_class(dag_df,values,label_var,n_labelled=20,n_validation=20,n_test=20)
        ds_name=f'sachs_{sv}_log' #directory for saving dataset
        output_dataset(ds_name,data_class,s_i)


#-----------------------------------
#     BREAST CANCER WISCONSIN DATASET
#-----------------------------------

RANDOM_SEED = 102
rng = default_rng(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
pl.seed_everything(RANDOM_SEED)



for s_i in range(100):
    list_of_sachs_variables=[]#['diagnosis']
    for sv in list_of_sachs_variables:
        dag_fn=f'feature_selection/bcancer_wisconsin/bcancer_diagnosis_dag.txt'
        csv_fn=f'feature_selection/bcancer_wisconsin/bcancer_wisconsin_cleaned.csv'
        names,tdag=parse_graph(dag_fn)
        dag_df=dag_to_df(tdag)
        values=pd.read_csv(csv_fn,sep=',')
        #balaance the labels
        #find min number
        min_class=min(values[values.diagnosis==0].shape[0],values[values.diagnosis==1].shape[0])
        #resample according to class label
        c0=values[values.diagnosis == 0].sample(min_class,replace=False)
        c1=values[values.diagnosis == 1].sample(min_class,replace=False)
        #make new df by joining both
        values=pd.concat([c0,c1],axis=0,ignore_index=True)
        keepcols=[n for n in names]
        values=values[keepcols]
        idx_v=[i for i in values.index]
        values.reset_index(inplace=True,drop=True)
        dcols=[c for c in values.columns]
        dcols=[c.replace('/','_') for c in dcols]
        values.columns=dcols
        label_var=sv
        data_class=create_data_class(dag_df,values,label_var,n_labelled=20,n_validation=20,n_test=20)
        ds_name=f'bcancer_{sv}' #directory for saving dataset
        output_dataset(ds_name,data_class,s_i)



##########
#
# BREAST CANCER WISCONSIN (Z-SCORE-TRANSOFRM)
#
######
from scipy.stats import zscore
for s_i in range(100):
    list_of_sachs_variables=['diagnosis']
    for sv in list_of_sachs_variables:
        dag_fn=f'feature_selection/bcancer_wisconsin/bcancer_diagnosis_dag.txt'
        csv_fn=f'feature_selection/bcancer_wisconsin/bcancer_wisconsin_cleaned.csv'
        names,tdag=parse_graph(dag_fn)
        dag_df=dag_to_df(tdag)
        values=pd.read_csv(csv_fn,sep=',')

        feature_columns=[c for c in values.columns if c!='type' and c!='diagnosis']

        for f in feature_columns:
            values[f]=zscore(values[f])
        #balaance the labels
        #find min number
        min_class=min(values[values.diagnosis==0].shape[0],values[values.diagnosis==1].shape[0])
        #resample according to class label
        c0=values[values.diagnosis == 0].sample(min_class,replace=False)
        c1=values[values.diagnosis == 1].sample(min_class,replace=False)
        #make new df by joining both
        values=pd.concat([c0,c1],axis=0,ignore_index=True)
        keepcols=[n for n in names]
        values=values[keepcols]
        idx_v=[i for i in values.index]
        values.reset_index(inplace=True,drop=True)
        dcols=[c for c in values.columns]
        dcols=[c.replace('/','_') for c in dcols]
        values.columns=dcols
        label_var=sv
        data_class=create_data_class(dag_df,values,label_var,n_labelled=10,n_validation=10,n_test=2)
        ds_name=f'bcancer_{sv}_zscore' #directory for saving dataset
        output_dataset(ds_name,data_class,s_i)













