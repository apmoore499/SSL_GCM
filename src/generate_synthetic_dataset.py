import igraph
import pandas as pd
import sklearn.datasets
import torch
import os
import pickle
import copy
import igraph as ig
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import glob
from PIL import Image
plotting_dset = True
NSAMP_RATIO=2
import shutil



import pandas as pd
import numpy as np
import plotly.express as px


def return_d1(n_to_generate,x1x2,scale):
    # now, let's say these rows each are element of polynomial kernel, of degree 2
    # run x1,x2 thru each and plot if greater/lesser than, as per what is written directly above ^^

    condition = 'not_met'

    while condition == 'not_met':
        # generate the hyperparameters for our polynomial kernel
        rnd_init = np.random.random(size=(100, 5))
        kernel_params = pd.DataFrame(rnd_init, columns=['X1^2', 'X2^2', 'X1', 'X2', 'X1X2'])


        # find polynomial kernel terms for our data
        pt = get_polynomial_d2_terms(x1x2)
        pt = pd.DataFrame(pt)  # convert to dataframe

        # index of row for testing our polynomial kernel
        testrow = 0
        # df.loc[testrow,]

        test_k = pt * kernel_params.loc[testrow,]  # multiply by kernel elements
        test_k['sum'] = test_k.sum(axis=1)  # sum results. note this is equivalent to W^(T)*X
        #test_k['class'] = (test_k['sum'] > 1).astype(int)

        #now transform by weighted sigmoid function: #26_03_2022 AM 10:13AM
        sw = get_sigmoid_weighted(np.mean(test_k['sum']), scale=scale)
        new_class = np.random.binomial(1, sw(test_k['sum']))
        test_k['class'] = new_class

        tm = 0.5  # if mean==0.5, classes are balanced exactly

        dm = 0.1  # acceptable deviation in the mean ie we need mean +- deviation_mean/2


        mean_lower = tm - dm / 2
        mean_upper = tm + dm / 2

        mu_s = test_k['class'].mean()  # sample mu

        if mean_lower < mu_s and mu_s <= mean_upper:
            condition = 'met'
            # accept this one
            # which means that we return it
            ret_dict = {}
            ret_dict = {'kernel_params': kernel_params,
                        'kernel_calc': test_k,
                        'sample_mu': mu_s,
                        'x1x2y': pd.concat([x1x2, test_k['class']], axis=1)}

    return (ret_dict)



# get terms for polynomial kernel of degree=2
def get_polynomial_d2_terms(in_df):
    retval = {}
    retval['X1^2'] = in_df.X1 ** 2
    retval['X2^2'] = in_df.X2 ** 2
    retval['X1'] = in_df.X1
    retval['X2'] = in_df.X2
    retval['X1X2'] = in_df.X1 * in_df.X2
    return (retval)


newfolders = ['saved_models',
              'synthetic_data',
              'synthetic_data_gumbel',
              'synthetic_data_marginal',
              'synthetic_data_gumbel_tc']




def rmat_2d(theta):
    retval = np.array([[np.cos(np.radians(theta)), np.sin(np.radians(theta))],
                       [-np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    return (retval)

# returns a function
def get_sigmoid_weighted(mu, scale):
    retval = lambda x: 1 / (1 + np.exp(-(x - mu) / scale))
    return (retval)


def return_mb_dict(dag):

    mb_dict = {}  # getting markov blanket
    for n in dag.nodes:
        mb_dict[n] = {}
        # get parents
        mb_dict[n]['parent'] = [n for n in dag.predecessors(n)]
        # get children
        mb_dict[n]['children'] = [n for n in dag.successors(n)]
        # spouses
        mb_dict[n]['spouses'] = reduce_list([[s for s in dag.predecessors(c)] for c in mb_dict[n]['children']])

    return(mb_dict)

def reduce_list(in_list):
    return (sum(in_list, []))

def plot_2d_data_w_dag(dsc, s_i):
    print('constructing 2d plots')
    tmp_plot_dir = 'TEMPORARY_PLOT_DIR_d_n_{0}_s_i_{1}'.format(dsc.d_n, s_i)
    # check existence purge if so
    if os.path.exists(tmp_plot_dir):
        shutil.rmtree(tmp_plot_dir)
    # make new dir
    os.mkdir(tmp_plot_dir)
    # now write out hsmps_dict to a png file...
    order_to_generate = dsc.dag.topological_sorting()
    # need to get class variable name
    lab_i = np.where([d == 'label' for d in dsc.var_types])[0][0]
    label_var_name = dsc.var_names[lab_i]
    # before we do that, need to merge synthetic  data w train data
    # get idx of feature variable
    dsc.feature_alphan = {}
    dsc.fdict = {v: t for v, t in zip(dsc.var_names, dsc.var_types)}
    dsc.feature_alphan = copy.deepcopy(dsc.fdict)
    for x in dsc.feature_alphan.keys():
        relevant_features = []
        fsplit = [f.split('_')[0] for f in dsc.feature_varnames]
        for fsplit, f in zip(fsplit, dsc.feature_varnames):
            if fsplit == x:
                relevant_features.append(f)
        dsc.feature_alphan[x] = relevant_features
    synthetic_and_orig_data = dsc.merge_dat#[dsc.merge_dat.type != 'validation']

    synthetic_and_orig_data = synthetic_and_orig_data.sample(frac=1)  # shuffle for random
    # and then we should be able to use facet type to plot varying conditions:
    # unlabel,label,validation,synthetic,test
    for v_i in order_to_generate:
        # get ancestors for this variable
        source_edges = dsc.dag.es.select(_target=v_i)
        source_vertices = [s_e.source_vertex for s_e in source_edges]
        sv = [v.index for v in source_vertices]  # get index of them
        current_variable_label = dsc.var_names[v_i]
        # label var...
        if dsc.var_types[v_i] == 'label':
            xv = current_variable_label
            yv = ""
            fig = px.histogram(synthetic_and_orig_data,
                               x=current_variable_label,
                               color=current_variable_label,
                               facet_col="type")
            fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)
            # and get source also
            for s in sv:
                current_source_variable = dsc.var_names[s]
                xv = dsc.feature_alphan[current_source_variable][0]
                yv = dsc.feature_alphan[current_source_variable][1]
                fig = px.scatter(synthetic_and_orig_data,
                                 x=xv,
                                 y=yv,
                                 color=label_var_name, facet_col="type",
                                 color_continuous_scale="bluered_r", opacity=0.3)
                fig.update_yaxes(
                    scaleanchor="x",
                    scaleratio=1,
                )
                fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)

        else:
            # we have conditional variables, but get the marginal dist first
            xv = dsc.feature_alphan[current_variable_label][0]
            yv = dsc.feature_alphan[current_variable_label][1]
            # transpose x,y even tho x->y so we get better resolution in X
            # because the plots are strteched such that Y is larger than X
            fig = px.scatter(synthetic_and_orig_data,
                             x=xv,
                             y=yv,
                             color=label_var_name, facet_col="type",
                             color_continuous_scale="bluered_r", opacity=0.3)
            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )
            fig.write_image("./{1}/{0}.png".format(xv + '_' + yv, tmp_plot_dir), scale=5)

    # save image of DAG
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 18.5)
    print('graph structrure for dataset: {0}'.format(dsc.d_n))

    ig.plot(dsc.dag,
            vertex_label=dsc.var_names,
            vertex_label_size=50,
            vertex_size=50,
            edge_width=5,
            edge_arrow_width=5,
            vertex_color='green',
            vertex_shape='circle',
            layout=dsc.dag.layout_grid(), target=ax)

    fig.savefig("./{0}/dn_{1}_si_{2}_DAG.png".format(tmp_plot_dir, dsc.d_n, dsc.s_i))

    # split into unlabelled/validation/synthetic and plot
    # get all temporary

    output_img = glob.glob('./{0}/*.png'.format(tmp_plot_dir))
    output_img = [o for o in output_img if 'hyperparameters' not in o]  # removing pil_text_font...
    output_img = [Image.open(o) for o in output_img]

    images = output_img
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    total_images = len(output_img)  # use this to apportion width / height, based on how many plot we want...

    # take square root of this...and round up..
    sqrt_img = int(np.sqrt(total_images)) + 1  # round up by adding '1'

    new_image_height = max(heights) * sqrt_img
    new_image_width = max(widths) * sqrt_img

    new_im = Image.new('RGB', (new_image_width, new_image_height))
    x_offset = 0
    y_offset = 0

    im_idx = 0

    for r in range(sqrt_img):
        for c in range(sqrt_img):
            if im_idx < total_images:
                im = images[im_idx]
                new_im.paste(im, (x_offset, y_offset))
                x_offset += im.size[0]
                im_idx += 1
        y_offset += im.size[1]
        x_offset = 0

    # new_im.paste(dict_img, (y_offset, int(new_image_height/ 2)))

    # num_already = len(os.listdir(dspec.save_folder + '/TEMPORARY_PLOT_DIR/combined_plots/'))

    new_im.save('./{0}/combined.png'.format(tmp_plot_dir))
    # read in spec to set seed...
    master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)
    # write dataset spec shorthand
    dspec = master_spec['dataset_spec']
    dspec.set_index("d_n", inplace=True)  # set this index for easier
    dspec = dspec.loc[dsc.d_n]  # use this as reference..

    new_im.save("{0}/data_for_train_d_n_{1}_s_i_{2}_plot.png".format(dspec.save_folder, dsc.d_n, dsc.s_i))

    # now delete the directory (if poss)
    try:
        print('now removing plot dir if exists')
        shutil.rmtree(tmp_plot_dir)
    except:
        print('error no plot directory to remove')
    finally:
        print('exiting...')

    print('plot for synthetic and real data finished')

# get our random number gen
class ds:
    def __init__(self, adj_mat, labels, merge_dat, variable_types, class_varname='Y'):
        self.adj_mat = adj_mat
        self.labels = labels
        self.dag = igraph.Graph.Adjacency(self.adj_mat)
        self.all_vi = [i.index for i in self.dag.vs]
        self.merge_dat = merge_dat
        self.variable_types = variable_types
        self.class_varname = class_varname
        self.set_vertex_dict()

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

# check if matrix is square
# t1 = 02_03_2022 the dset we try after meeting to get more separation b/w classes
class DAG_dset:

    def __init__(self,**kwargs):

        for k in kwargs.keys():
            setattr(self, k, kwargs[k])
        self.NSAMP_RATIO=1
        self.n_samples=self.n_unlabelled+self.n_labelled+self.n_validation+self.n_test
        self.n_to_generate = int(self.NSAMP_RATIO * self.n_samples)
        self.adj_mat=np.array(self.adjacency_matrix)
        self.dag = igraph.Graph.Adjacency(self.adj_mat)  # get dag
        assert self.dag.is_dag()  # make sure is dag
        assert 'label' in self.var_types  # make sure we have at least one label
        assert self.adj_mat.shape[0] >= 2  # make sure at least two variables in DAG

        # get topological sorting of this one
        self.v_topsort = self.dag.topological_sorting()

        self.all_vi = [i.index for i in self.dag.vs]

        # read in spec to set seed...
        master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)
        # write dataset spec shorthand
        dspec = master_spec['dataset_spec']
        dspec.set_index("d_n", inplace=True)  # set this index for easier
        dspec = dspec.loc[self.d_n]  # use this as reference..

        #seed for deterministic

        self.rng = np.random.default_rng(int(str(dspec.dn_log) + str(self.s_i)))
        np.random.seed(int(str(dspec.dn_log) + str(self.s_i)))

        self.set_vertex_dict()
        self.synthesise_values()
        self.label_variables()
        self.convert_all_values_to_dict()
        self.concat_df()
        self.val_split()
        self.partition_data_df()
        self.subset_tensors()
        self.output_dataset_synthdag()
        print('pausing here')
        # self.variable_types=variable_types
        # self.feature_dim=feature_dim

    def synthesise_values(self):

        # get partitions of label/feature
        topsort_order = np.array([self.var_types[i] for i in self.v_topsort])
        # lab_idx
        lab_idx = np.where(topsort_order == 'label')[0]

        # partition into causal/label/effect index
        self.ce_dict = {'cause': self.v_topsort[:lab_idx[0]],
                        'lab': lab_idx,
                        'effect': self.v_topsort[lab_idx[0] + 1:]}

        # pd df
        self.vertex_idx = np.array([i for i in range(len(self.var_types))])
        self.graph_df = pd.DataFrame([self.v_topsort, self.var_types, self.vertex_idx]).transpose()
        self.graph_df.columns = ['topological_sorting', 'var_type', 'vertex_idx']

        self.set_vertex_dict()  # set the dictionary for vvertices etc

        print('pausing here')

        # set mu
        for vi in self.vertex_idx:
            self.dag.vs[vi]['mu'] = {}
            self.dag.vs[vi]['mu'][0] = np.random.uniform(-1, 1, self.feature_dim)

            # sample offset for class difference

            c_offset = 3
            sample_offset = np.random.uniform(-c_offset, c_offset)

            self.dag.vs[vi]['mu'][1] = np.random.uniform(-1 + sample_offset, 1 + sample_offset, self.feature_dim)

        # set sigma (wishart matrix)
        for vi in self.vertex_idx:
            self.dag.vs[vi]['sig'] = {}

            self.dag.vs[vi]['sig'][0] = sklearn.datasets.make_spd_matrix(self.feature_dim)
            self.dag.vs[vi]['sig'][0] = self.dag.vs[vi]['sig'][0] / np.sqrt(np.linalg.det(self.dag.vs[vi]['sig'][0]))

            sample_offset_0 = np.random.uniform(0.75, 1.25)  # sample offset for class-dependent covariance matrix
            self.dag.vs[vi]['sig'][0] *= sample_offset_0

            self.dag.vs[vi]['sig'][1] = sklearn.datasets.make_spd_matrix(self.feature_dim)
            self.dag.vs[vi]['sig'][1] = self.dag.vs[vi]['sig'][1] / np.sqrt(np.linalg.det(self.dag.vs[vi]['sig'][1]))

            sample_offset_1 = np.random.uniform(0.75, 1.25)  # sample offset for class-dependent covariance matrix
            self.dag.vs[vi]['sig'][1] *= sample_offset_1

        # set edge weights
        for e in self.dag.es:
            e['coef'] = {}
            e['coef'][0] = np.random.uniform(-1, 1, self.feature_dim)

            c_offset = 2
            sample_offset = np.random.uniform(-c_offset, c_offset)
            e['coef'][1] = np.random.uniform(-1 + sample_offset, 1 + sample_offset,
                                             self.feature_dim)  # add to sample offset for 0,1

        # now simulate...
        self.values = []

        self.variable_values = {}

        # 1. cause variable
        for i in self.ce_dict['cause']:
            # multidm gaussain
            current_mu = self.dag.vs[i]['mu'][0]
            current_sig = self.dag.vs[i]['sig'][0]
            # need to get wishart dist
            x_samples = np.random.multivariate_normal(current_mu, current_sig, self.n_to_generate)
            self.variable_values[i] = x_samples

        # 2. label variable
        for i in self.ce_dict['lab']:  # list of size 1 only
            # source index based on i
            source_vertices = self.vertex_dict['sv_i'][i]
            y_activation = np.array([0.0] * self.n_to_generate)

            # set our mu for bernoulli
            total_mu = 0.0  # total mu
            for sv in source_vertices:
                # identify edge based on sv and target index
                cur_edge = [e for e in self.dag.es.select(_source=sv, _target=i)][0]
                cur_edge_coef = cur_edge['coef'][0]
                cur_sv_mu = self.dag.vs.select(sv)['mu'][0][0]
                cur_weighted_mu = np.matmul(cur_edge_coef, cur_sv_mu)
                total_mu += cur_weighted_mu

            # now we set the function

            self.sigmoid_weighted = get_sigmoid_weighted(mu=total_mu, scale=0.25)

            for sv in source_vertices:
                # identify edge based on sv and target index
                cur_edge = [e for e in self.dag.es.select(_source=sv, _target=i)][0]
                cur_edge_coef = cur_edge['coef'][0]

                cur_vals = np.matmul(self.variable_values[sv], cur_edge_coef)
                y_activation += cur_vals

            # now get sigmoid
            sig_ya = self.sigmoid_weighted(y_activation)
            # now get bernoulli
            b_sig_ya = np.random.binomial(1, sig_ya)  # '''bernoulli sigmoid y activation'''
            # ok now we have y val
            self.variable_values[i] = b_sig_ya.reshape((-1, 1))

            # pass
            # get our edges into this one

        y_idx = self.ce_dict['lab'][0]
        # 3. effect variable
        for i in self.ce_dict['effect']:  # list of size 1 only
            # get source v for thise
            svc = []
            source_vertices = self.vertex_dict['sv_i'][i]

            cur_label = self.variable_values[y_idx][i][0]
            for sv in source_vertices:
                # if we are class-dependent,...
                if sv == y_idx:
                    # get edge connection
                    cur_edge = [e for e in self.dag.es.select(_source=sv, _target=i)][0]
                    # get edge weight
                    cur_edge_coef = cur_edge['coef'][cur_label]
                    # get sv value
                    source_value = self.variable_values[sv]
                    # multiply them together elementwise
                    sv_contrib = source_value * cur_edge_coef
                    svc.append(sv_contrib)  # append tolist of osurce variable contributions


                else:
                    # get edge connection
                    cur_edge = [e for e in self.dag.es.select(_source=sv, _target=i)][0]
                    # get edge weight
                    cur_edge_coef = cur_edge['coef'][0] * 0.5  # scale by 0.5 cos contrib from Y not super obvious
                    # get sv value
                    source_value = self.variable_values[sv]
                    # multiply them together elementwise
                    sv_contrib = source_value * cur_edge_coef
                    svc.append(sv_contrib)  # append tolist of osurce variable contributions
            print('pausing here')
            all_sv_contrib = np.sum(svc, 0)
            # multidm gaussain
            current_mu = self.dag.vs[i]['mu'][cur_label]
            current_sig = self.dag.vs[i]['sig'][cur_label]
            # need to get wishart dist
            x_samples = np.random.multivariate_normal(current_mu, current_sig, self.n_to_generate)
            self.variable_values[i] = x_samples + all_sv_contrib

        return (self)

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

    def label_variables(self):
        vnames = [""] * self.dag.vcount()
        feat_num = 1
        for v_i in self.dag.topological_sorting():
            if self.var_types[v_i] == 'label':
                vnames[v_i] = 'Y'
            else:
                vnames[v_i] = 'X{0}'.format(feat_num)
                feat_num += 1
        self.var_names = vnames
        return (self)

    def convert_all_values_to_dict(self):
        values_dict = {}
        for var_i in self.dag.topological_sorting():
            var_label = self.var_names[var_i]
            var_vals = self.variable_values[var_i]
            values_dict[var_label] = var_vals
        self.values_dict = values_dict
        return (self)

    def concat_df(self):
        # get all dfs
        self.dfs = [pd.DataFrame(self.values_dict[label]) for label in self.var_names]
        self.new_dfs = []  # name them
        for var, df in zip(self.var_names, self.dfs):
            df.columns = ['{0}_{1}'.format(var, dim) for dim in iter(df.columns)]
            self.new_dfs.append(df)
        self.dfs = self.new_dfs
        self.merge_dat = pd.concat(self.dfs, axis=1)  # concat them...
        return (self)

    def val_split(self):

        df = self.merge_dat

        df0=self.merge_dat[self.merge_dat.Y_0==0]
        df1=self.merge_dat[self.merge_dat.Y_0==1]

        df0_size=df0.shape[0]
        df1_size=df1.shape[0]



        total_n=self.n_labelled + self.n_unlabelled + self.n_validation+self.n_test

        ratio_class=1.0*df0.shape[0]/(df0.shape[0]+df1.shape[0])
        r_c0_labelled=int(np.rint(ratio_class*self.n_labelled))
        r_c0_unlabelled = int(np.rint(ratio_class * self.n_unlabelled))
        r_c0_val = int(np.rint(ratio_class * self.n_validation))
        r_c0_test= int(np.rint(ratio_class*self.n_test))

        r_c1_labelled=self.n_labelled-r_c0_labelled
        r_c1_unlabelled=self.n_unlabelled-r_c0_unlabelled
        r_c1_val=self.n_validation-r_c0_val
        r_c1_test=self.n_test-r_c0_test


        #nrow_d0=df0_sample.shape[0]

        #n_d0=r_c0_labelled + r_c0_unlabelled + r_c0_val + r_c0_test

        #nrow_d1 = df1_sample.shape[0]

        #n_d1 = r_c1_labelled + r_c1_unlabelled + r_c1_val + r_c1_test

        #don't need to subset in practice, cos NSAMP_RATIO=1, set to
        df0_sample = df0#.sample(n=r_c0_labelled + r_c0_unlabelled + r_c0_val + r_c0_test)  # subset to nlabel,label,validation
        df1_sample = df1#.sample(n=r_c1_labelled + r_c1_unlabelled + r_c1_val + r_c1_test)

        size_0=len(df0_sample)
        ratio_unlabelled=1.0 * r_c0_unlabelled/size_0
        ratio_labelled = 1.0 * r_c0_labelled / size_0
        ratio_validation=1.0 * r_c0_val/size_0

        ulab_split=ratio_unlabelled
        lab_split=ulab_split+ratio_labelled
        val_split=lab_split+ratio_validation

        unlabel_0, label_0, val_0, test_0 = np.split(df0_sample, [int(ratio_unlabelled*size_0),
                                                          int(lab_split * size_0),
                                                          int(val_split * size_0)])
        unlabel_0['type'] = 'unlabelled'
        label_0['type'] = 'labelled'
        val_0['type'] = 'validation'
        test_0['type'] = 'test'

        size_1 = len(df1_sample)
        ratio_unlabelled = 1.0 * r_c1_unlabelled/ size_1
        ratio_labelled = 1.0 * r_c1_labelled / size_1
        ratio_validation = 1.0 * r_c1_val / size_1

        ulab_split = ratio_unlabelled
        lab_split = ulab_split + ratio_labelled
        val_split = lab_split + ratio_validation

        unlabel_1, label_1, val_1, test_1 = np.split(df1_sample, [int(ratio_unlabelled * size_1),
                                                           int(lab_split * size_1),
                                                           int(val_split * size_1)])
        unlabel_1['type'] = 'unlabelled'
        label_1['type'] = 'labelled'
        val_1['type'] = 'validation'
        test_1['type'] = 'test'


        # combine all togeth
        self.merge_dat = pd.concat([unlabel_0, label_0, val_0, test_0,
                                    unlabel_1, label_1, val_1, test_1], ignore_index=True, axis=0)

        # print crosstabs out
        self.crosstabs = pd.crosstab(self.merge_dat.type, self.merge_dat.Y_0)
        print(self.crosstabs)

        return (self)

    def partition_data_df(self):
        self.label_features_df = self.merge_dat.drop(columns='Y_0')[self.merge_dat.type == 'labelled'].drop(
            columns='type')
        self.unlabel_features_df = self.merge_dat.drop(columns='Y_0')[self.merge_dat.type == 'unlabelled'].drop(
            columns='type')
        self.val_features_df = self.merge_dat.drop(columns='Y_0')[self.merge_dat.type == 'validation'].drop(
            columns='type')
        self.label_y_df = self.merge_dat.filter(items=['Y_0', 'type'])[self.merge_dat.type == 'labelled'].drop(
            columns='type')
        self.unlabel_y_df = self.merge_dat.filter(items=['Y_0', 'type'])[self.merge_dat.type == 'unlabelled'].drop(
            columns='type')
        self.val_y_df = self.merge_dat.filter(items=['Y_0', 'type'])[self.merge_dat.type == 'validation'].drop(
            columns='type')
        self.test_features_df = self.merge_dat.drop(columns='Y_0')[self.merge_dat.type == 'test'].drop(
            columns='type')
        self.test_y_df = self.merge_dat.filter(items=['Y_0', 'type'])[self.merge_dat.type == 'test'].drop(
            columns='type')
        return (self)

    def subset_tensors(self):
        all_vars = [c for c in self.merge_dat.columns]
        self.feature_varnames = [v for v in all_vars if 'X' in v]
        self.class_varname = [v for v in all_vars if 'Y' in v]
        self.train_label_dataset = self.merge_dat[self.merge_dat['type'] == 'labelled']
        self.train_unlabel_dataset = self.merge_dat[self.merge_dat['type'] == 'unlabelled']
        self.val_dataset = self.merge_dat[self.merge_dat['type'] == 'validation']
        self.test_dataset = self.merge_dat[self.merge_dat['type'] == 'test']

        # -----------------------------------
        # split into features and class label
        # -----------------------------------
        # -----------
        # labelled
        # -----------
        self.label_features = torch.Tensor(self.train_label_dataset[self.feature_varnames].values).float()
        self.label_y = torch.Tensor(self.train_label_dataset[self.class_varname].values)

        # -----------
        # unlabelled
        # -----------

        self.unlabel_features = torch.Tensor(self.train_unlabel_dataset[self.feature_varnames].values).float()
        self.unlabel_y = torch.Tensor(self.train_unlabel_dataset[self.class_varname].values)

        # -----------
        # validation
        # -----------

        self.val_features = torch.Tensor(self.val_dataset[self.feature_varnames].values).float()
        self.val_y = torch.Tensor(self.val_dataset[self.class_varname].values)

        # -----------
        # test
        # -----------

        self.test_features = torch.Tensor(self.test_dataset[self.feature_varnames].values).float()
        self.test_y = torch.Tensor(self.test_dataset[self.class_varname].values)

        # --------------------
        # convert y to one-hot and squeeze
        # --------------------

        self.label_y = torch.nn.functional.one_hot(self.label_y.type(torch.LongTensor), 2).squeeze(1)
        self.unlabel_y = torch.nn.functional.one_hot(self.unlabel_y.type(torch.LongTensor), 2).squeeze(1)
        self.val_y = torch.nn.functional.one_hot(self.val_y.type(torch.LongTensor), 2).squeeze(1)
        self.test_y = torch.nn.functional.one_hot(self.test_y.type(torch.LongTensor), 2).squeeze(1)

        return (self)

    def output_dataset_synthdag(self):
        current_dn = '{0}'.format(self.d_n)
        cd_dir = './data/dataset_{0}'.format(current_dn)

        os.makedirs(cd_dir, exist_ok=True)
        data_types = ['label_features',
                      'unlabel_features',
                      'val_features',
                      'label_y',
                      'unlabel_y',
                      'val_y',
                      'test_features',
                      'test_y']

        # saving all data types ie label/unlabel etc
        for dt in data_types:
            out_tens = getattr(self, dt)
            out_fn = '{0}/d_n_{1}_s_i_{2}_{3}.pt'.format(cd_dir, current_dn, self.s_i, dt)
            torch.save(out_tens, out_fn)
        # also save the whole class so can recover hyperparameter etc if need
        print('dataset tensors saved for {0}'.format(self.d_n))

        # rename Y_0 to Y
        all_c = [c for c in self.merge_dat.columns]
        if 'Y_0' in all_c:
            self.merge_dat.rename(columns={'Y_0': 'Y'}, inplace=True)
        # create class
        data_class = ds(adj_mat=self.adjacency_matrix,
                        labels=self.var_names,
                        merge_dat=self.merge_dat,
                        variable_types=self.var_types)
        # save it
        with open('{0}/d_n_{1}_s_i_{2}_dataset_class.pickle'.format(cd_dir, current_dn, self.s_i), 'wb') as file:
            pickle.dump(data_class, file)
        print('dataset saved to pickle for {0}'.format(self.d_n))

        # create dag and save dag fig
        dag_fn = '{0}/d_n_{1}_s_i_{2}_dataset_dag.png'.format(cd_dir, current_dn, self.s_i)
        dag_fig = self.plotting_dag(data_class, dn='testdag', target=dag_fn)

        # create new folders in the directory
        for n in newfolders:
            newdir = '{0}/{1}/'.format(cd_dir, n)
            os.makedirs(newdir, exist_ok=True)

    def plotting_dag(self, ds, dn, target):
        # print('graph structrure for dataset: {0}'.format(dn))
        igraph.plot(ds.dag,
                    bbox=(0, 0, 300, 300),
                    vertex_label=ds.labels,
                    vertex_label_size=13,
                    vertex_size=30,
                    vertex_color='white',
                    layout=ds.dag.layout_fruchterman_reingold(),
                    edge_curved=False,
                    target=target)

class DAG_dset_t36(DAG_dset):

    def synthesise_values(self):
        # @Mingming - here is where we synthesise mixture model values. [START]

        # get partitions of label/feature
        topsort_order = np.array([self.var_types[i] for i in self.v_topsort])
        # lab_idx
        self.lab_idx = np.where(topsort_order == 'label')[0]


        # need to partition into spouse also

        self.networkx_dag=self.dag.to_networkx() #converting to networkx object for easier
        self.mb_dict=return_mb_dict(self.networkx_dag)
        self.mb_label_var=self.mb_dict[self.lab_idx[0]]

        #ok now if any v common to spouse/parent, put in parent only
        for v in self.mb_label_var['parent']:
            if v in self.mb_label_var['spouses']:
                self.mb_label_var['spouses'].remove(v)

        #remove self from spouses
        try:
            self.mb_label_var['spouses'].remove(self.lab_idx[0])
        except:
            next
        finally:
            next

        # partition into causal/label/effect index
        self.ce_dict = {'cause': self.mb_label_var['parent'],
                        'spouse':self.mb_label_var['spouses'],
                        'lab': self.lab_idx,
                        'effect': self.mb_label_var['children']}




        # pd df
        self.vertex_idx = np.array([i for i in range(len(self.var_types))])
        self.graph_df = pd.DataFrame([self.v_topsort, self.var_types, self.vertex_idx]).transpose()
        self.graph_df.columns = ['topological_sorting', 'var_type', 'vertex_idx']

        self.set_vertex_dict()  # set the dictionary for vvertices etc

        print('pausing here')

        # now simulate...
        self.values = []

        self.variable_values = {}

        scale_factor_cause = self.rng.uniform(1,2)  # scale factor by which to stretch our dist

        sig_scale_cause = np.array([[scale_factor_cause, 0], [0, 1 / scale_factor_cause]])  # scale matrix

        #y_idx=1

        # synthesise x spouse


        offsets_dict={}



        for i in self.ce_dict['spouse']:  # list of size 1 only

            scale_factor_spouse = self.rng.uniform(1,2)  # scale factor by which to stretch our dist
            sig_scale_spouse = np.array([[scale_factor_spouse, 0], [0, 1 / scale_factor_spouse]])  # scale matrix
            xs_noise = np.random.multivariate_normal([0, 0], sig_scale_spouse, self.n_to_generate)
            self.variable_values[i] = xs_noise
            offsets_dict[i]={}
            offsets_dict[i][0]=np.array([0,0])
            offsets_dict[i][1] = np.array([0, scale_factor_spouse/2])
        # synthesise x cause

        for i in self.ce_dict['cause']:  # list of size 1 only
            xe_noise = np.random.multivariate_normal([0, 0], sig_scale_cause, self.n_to_generate)
            x1x2 = pd.DataFrame(xe_noise, columns=['X1', 'X2'])
            dsynth = return_d1(n_to_generate=self.n_to_generate, x1x2=x1x2,scale=1)['x1x2y']
            self.variable_values[i] = dsynth[['X1', 'X2']].values

            offsets_dict[i] = {}
            offsets_dict[i][0] = np.array([0, 0])
            offsets_dict[i][1] = np.array([0, scale_factor_cause / 2])

        # synthesise x2
        for i in self.ce_dict['lab']:  # list of size 1 only
            source_vertices = self.vertex_dict['sv_i'][i]

            if len(source_vertices)==1:

                self.variable_values[i] = dsynth[['class']].values

            else:
                sig_ya=[0.5]*self.n_to_generate
                b_sig_ya = np.random.binomial(1, sig_ya)  # '''bernoulli sigmoid y activation'''
                # ok now we have y val
                self.variable_values[i] = b_sig_ya.reshape((-1, 1))


        #synthesise x2
        for i in self.ce_dict['effect']: # list of size 1 only
            #get source v for thise
            svc=[]
            source_vertices = self.vertex_dict['sv_i'][i]
            all_labels = self.variable_values[self.lab_idx.tolist()[0]].flatten()

            print('pausing here')
            all_sv_contrib=np.sum(svc,0)

            scale_factor_effect=self.rng.uniform(4,6) #scale factor by which to stretch our dist

            sig_scale_effect = np.array([[scale_factor_effect, 0], [0, 1 / scale_factor_effect]]) #scale matrix


            rotation_angle=0#-45#self.rng.uniform(15,75)
            flip_rot=self.rng.binomial(1,0.5)
            muflip=1
            if flip_rot==1:
                rotation_angle=-rotation_angle
                muflip=-1

            x_offset=0
            y_offset=scale_factor_effect/2

            self.dag.vs[i]['mu']={}
            self.dag.vs[i]['mu'][0] = np.array([0, 0])
            self.dag.vs[i]['mu'][1] = np.array([x_offset, y_offset])

            # need to get wishart dist
            xe_noise = np.random.multivariate_normal([0,0], sig_scale_effect, self.n_to_generate)

            xe_n0=np.array([np.matmul(rmat_2d(rotation_angle),x) for x in xe_noise])
            xe_n0=np.array([self.dag.vs[i]['mu'][0] + x for x in xe_n0])

            xe_n1 = np.array([np.matmul(rmat_2d(rotation_angle), x) for x in xe_noise])
            xe_n1 = np.array([self.dag.vs[i]['mu'][1] + x for x in xe_n1])

            out_x=[]
            for idx,l in enumerate(all_labels):
                if l==0:
                    out_x.append(xe_n0[idx])
                else:
                    out_x.append(xe_n1[idx])

            self.variable_values[i] = np.array(out_x)

            def offset_cosine(x):
                return(np.cos(x))


            old_x=np.array(out_x)
            old_x1=old_x[:,0]
            new_x2_offset=4*offset_cosine(old_x1/2)
            new_x=old_x
            new_x[:,1]+=new_x2_offset
            self.variable_values[i]=new_x


            #now add class info if y->X_E and X_S->X_E
            source_vertices = self.vertex_dict['sv_i'][i]

            x_offset = 0
            y_offset = 1/scale_factor_effect

            mu_dict={}
            mu_dict[0]=np.array([0, 0])
            mu_dict[1]=np.array([x_offset, y_offset])

            if self.lab_idx in source_vertices:

                svc=copy.deepcopy(source_vertices)
                svc.remove(self.lab_idx)

                #and then copy to new_x, dependent on lab el

                yvals=self.variable_values[self.lab_idx.tolist()[0]]
                old_x = self.variable_values[i]

                for s in svc:
                    source_x=self.variable_values[s]
                    y_affine_offset = np.array([offsets_dict[s][yv] for yv in yvals.flatten().tolist()])
                    y_affine_offset.reshape(source_x.shape)
                    old_x=old_x+source_x+ y_affine_offset


                new_x=old_x

                self.variable_values[i] = new_x

        return(self)

# set some paramters before we generate synthetic data

base_synthetic_partition_spec = {
    'feat_dim': 2,
    'n_unlabelled': 1000,
    'n_labelled': 40,
    'n_validation': 40,
    'n_test': 1000,
    'num_si': 100,
    'dgen_func': DAG_dset_t36,
    'experiment': 36,
    'experiment_addendum':'',
}


base_synthetic_partition_spec_10000 = {
    'feat_dim': 2,
    'n_unlabelled': 10000,
    'n_labelled': 40,
    'n_validation': 40,
    'n_test': 1000,
    'num_si': 100,
    'dgen_func': DAG_dset_t36,
    'experiment': 36_10000,
    'experiment_addendum':'_10000',
}



base_synthetic_partition_spec_100000 = {
    'feat_dim': 2,
    'n_unlabelled': 100000,
    'n_labelled': 40,
    'n_validation': 40,
    'n_test': 1000,
    'num_si': 100,
    'dgen_func': DAG_dset_t36,
    'experiment': 36_100000,
    'experiment_addendum':'_100000',
}


base_synthetic_partition_spec_5000 = {
    'feat_dim': 2,
    'n_unlabelled': 5000,
    'n_labelled': 40,
    'n_validation': 40,
    'n_test': 1000,
    'num_si': 100,
    'dgen_func': DAG_dset_t36,
    'experiment': 36_5000,
    'experiment_addendum':'_5000',
}


n_to_generate_dict={
        1:100,    
        2:100,
        3:100,
        4:100,
        5:100,
        6:100,
        7:100,        
        1_10000:100,    
        2_10000:100,
        3_10000:100,
        4_10000:100,
        5_10000:100,
        6_10000:100,
        7_10000:100,
        1_100000:100,
        2_100000:100,
        3_100000:100,
        4_100000:100,
        5_100000:100,
        6_100000:100,
        7_100000:100,
        1_5000:100,
        2_5000:100,
        3_5000:100,
        4_5000:100,
        5_5000:100,
        6_5000:100,
        7_5000:100,
        
}

n_plots_dict={
        1:2,    
        2:2,
        3:2,
        4:2,
        5:2,
        6:2,
        7:2,
        1_10000:2,  
        2_10000:2,
        3_10000:2,
        4_10000:2,
        5_10000:2,
        6_10000:2,
        7_10000:2,
        1_100000:2,
        2_100000:2,
        3_100000:2,
        4_100000:2,
        5_100000:2,
        6_100000:2,
        7_100000:2,
        1_5000:2,
        2_5000:2,
        3_5000:2,
        4_5000:2,
        5_5000:2,
        6_5000:2,
        7_5000:2,
}

dset_spec_dict={'base':base_synthetic_partition_spec,
            'base_10000':base_synthetic_partition_spec_10000,
            'base_100000':base_synthetic_partition_spec_100000,
            'base_5000':base_synthetic_partition_spec_5000,}




#for dsk in ['base','base_10000','base_100000']:
for dsk in ['base_5000']:
    
    dset_spec=dset_spec_dict[dsk]


    # -----------------------------------
    #     Dataset 1: X1->Y
    # -----------------------------------

    print('synthesising for X1->Y')
    adjacency_matrix = [[0, 1],
                        [0, 0]]

    for s_i in range(n_to_generate_dict[1]):


        dset_kwargs={'adjacency_matrix':adjacency_matrix,
                        'var_types': ['feature', 'label'],
                        'feature_dim': dset_spec['feat_dim'],
                        'd_n':'n36_gaussian_mixture_d1{}'.format(dset_spec['experiment_addendum']),
                        's_i': s_i,
                        'NSAMP_RATIO':2}


        dset_kwargs.update(dset_spec)
        dset = dset_spec['dgen_func'](**dset_kwargs)




        if plotting_dset and s_i < n_plots_dict[1]:
            plot_2d_data_w_dag(dset, s_i)

        print('synthesise for: {0}'.format(s_i))



    # -----------------------------------
    #     Dataset 2: Y->X2
    # -----------------------------------

    print('synthesising for Y->X2')
    adjacency_matrix = [[0, 1],
                        [0, 0]]


    for s_i in range(n_to_generate_dict[2]):
        dset_kwargs={'adjacency_matrix':adjacency_matrix,
                    'var_types': ['label', 'feature'],
                    'feature_dim': dset_spec['feat_dim'],
                    'd_n':'n36_gaussian_mixture_d2{}'.format(dset_spec['experiment_addendum']),
                    's_i': s_i}


        dset_kwargs.update(dset_spec)
        dset = dset_spec['dgen_func'](**dset_kwargs)





        if plotting_dset and s_i < n_plots_dict[2]:
            plot_2d_data_w_dag(dset, s_i)

        print('synthesise for: {0}'.format(s_i))

        print('pauseing here')

    # -----------------------------------
    #     Dataset 3: X1->Y->X2
    # -----------------------------------

    print('synthesising for X1->Y->X2')
    adjacency_matrix = [[0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]]


    for s_i in range(n_to_generate_dict[3]):

        dset_kwargs = {'adjacency_matrix': adjacency_matrix,
                    'var_types':['feature', 'label', 'feature'],
                    'feature_dim': dset_spec['feat_dim'],
                    'd_n': 'n36_gaussian_mixture_d3{}'.format(dset_spec['experiment_addendum']),
                    's_i': s_i,
                    'NSAMP_RATIO': 2}

        dset_kwargs.update(dset_spec)
        dset = dset_spec['dgen_func'](**dset_kwargs)

        if plotting_dset and s_i < n_plots_dict[3]:
                plot_2d_data_w_dag(dset, s_i)

        print('synthesise for: {0}'.format(s_i))



    # -----------------------------------
    #     Dataset 4: X1->X2, Y->X2
    # -----------------------------------

    print('synthesising for Y->X2,X1->X2')
    adjacency_matrix = [[0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 0]]


    for s_i in range(n_to_generate_dict[4]):
        dset_kwargs = {'adjacency_matrix': adjacency_matrix,
                    'var_types':['feature', 'label', 'feature'],
                    'feature_dim': dset_spec['feat_dim'],
                    'd_n': 'n36_gaussian_mixture_d4{}'.format(dset_spec['experiment_addendum']),
                    's_i': s_i,
                    'NSAMP_RATIO': 2}

        dset_kwargs.update(dset_spec)
        dset = dset_spec['dgen_func'](**dset_kwargs)

        if plotting_dset and s_i < n_plots_dict[4]:
            plot_2d_data_w_dag(dset, s_i)

        print('synthesise for: {0}'.format(s_i))


    # -----------------------------------
    #     Dataset 5: X1->Y->X2, X1->X2
    # -----------------------------------

    print('synthesising for X1->Y->X2,X1->X2')
    adjacency_matrix = [[0, 1, 1],
                        [0, 0, 1],
                        [0, 0, 0]]


    for s_i in range(n_to_generate_dict[5]):
        dset_kwargs = {'adjacency_matrix': adjacency_matrix,
                    'var_types':['feature', 'label', 'feature'],
                    'feature_dim': dset_spec['feat_dim'],
                    'd_n': 'n36_gaussian_mixture_d5{}'.format(dset_spec['experiment_addendum']),
                    's_i': s_i,
                    'NSAMP_RATIO': 2}

        dset_kwargs.update(dset_spec)
        dset = dset_spec['dgen_func'](**dset_kwargs)

        if plotting_dset and s_i < n_plots_dict[5]:
            plot_2d_data_w_dag(dset, s_i)

        print('synthesise for: {0}'.format(s_i))


    # -----------------------------------
    #     Dataset 6: X1->Y->X3, X1->X3<-X2,
    # -----------------------------------

    print('synthesising for Dataset 6: X2->Y->X3, X1->X3<-X2')
    adjacency_matrix = [[0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]]


    for s_i in range(n_to_generate_dict[6]):
        dset_kwargs = {'adjacency_matrix': adjacency_matrix,
                    'var_types':['feature', 'feature', 'label', 'feature'],
                    'feature_dim': dset_spec['feat_dim'],
                    'd_n': 'n36_gaussian_mixture_d6{}'.format(dset_spec['experiment_addendum']),
                    's_i': s_i,
                    'NSAMP_RATIO': 2}

        dset_kwargs.update(dset_spec)
        dset = dset_spec['dgen_func'](**dset_kwargs)

        if plotting_dset and s_i < n_plots_dict[6]:
            plot_2d_data_w_dag(dset, s_i)

        print('synthesise for: {0}'.format(s_i))



    # -----------------------------------
    #     Dataset 7: X1->Y->X3<-X2
    # -----------------------------------

    print('synthesising for Dataset 7: X2->Y->X3, X1->X3<-X2')
    adjacency_matrix = [[0, 0, 1, 0], #XC
                        [0, 0, 0, 1], #XS
                        [0, 0, 0, 1], #Y
                        [0, 0, 0, 0]] #XE


    for s_i in range(n_to_generate_dict[7]):
        dset_kwargs = {'adjacency_matrix': adjacency_matrix,
                    'var_types':['feature', 'feature', 'label', 'feature'],
                    'feature_dim': dset_spec['feat_dim'],
                    'd_n': 'n36_gaussian_mixture_d7{}'.format(dset_spec['experiment_addendum']),
                    's_i': s_i,
                    'NSAMP_RATIO': 2}

        dset_kwargs.update(dset_spec)
        dset = dset_spec['dgen_func'](**dset_kwargs)

        if plotting_dset and s_i < n_plots_dict[7]:
            plot_2d_data_w_dag(dset, s_i)

        print('synthesise for: {0}'.format(s_i))







import sys

sys.exit()
