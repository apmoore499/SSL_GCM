

import sys
sys.path.append('..')
sys.path.append('../py/')

from benchmarks_utils import *
import glob
import pandas as pd
import numpy as np
import copy
import collections
SEED=10
np.random.seed(SEED)
debug=False


EXPORT_LATEK=True


############################
#
#     results_collate.py - collate results on synthetic data
#     set subset of datasets using rel_dn and these will be parsed into xlsx spreadsheet based on feature dim (fd)
#
############################


def get_acc_ulab(save_folder ,s_i ,model_type):
    if debug:
        set_trace()
    current_dn = '{0}/saved_models/{1}-s_i={2}_unlabel_acc.out'.format(save_folder,model_type,s_i)  # change to this 06_03_2022 ie, need '-' in front of *.ckpt
    acc=np.loadtxt(current_dn)
    return(acc)

def get_acc_test(save_folder ,s_i ,model_type):
    if debug:
        set_trace()
    current_dn = '{0}/saved_models/{1}-s_i={2}_test_acc.out'.format(save_folder,model_type,s_i)    # change to this 06_03_2022 ie, need '-' in front of *.ckpt
    acc=np.loadtxt(current_dn)
    return(acc)

def return_dag_image_fn(d_n ,s_i):
    retval ='<img src="synthetic_dags/d_{0}_s_i_{1}.png" width="100" height="100">'.format(d_n ,s_i)
    return(retval)

#return summary statistics, mean and standard deviation, for results
def get_summary_stats(in_df):
    #get summary stats of columns
    in_df_means=in_df.mean()
    in_df_std=in_df.std()
    #get idx of computed statistics
    idx_list=[i for i in in_df_means.index]
    #now, we want to have dictionary for outputs
    outputs_dict={}
    #for each idx, loop thru
    for idx in idx_list:
        outputs_dict[idx]='{:.3f} ± {:.3f}'.format(in_df_means.loc[idx]*100,
                                                   in_df_std.loc[idx]*100)
    #now get n cases
    outputs_dict['n']=in_df.shape[0]
    #return outputs_dict
    return(outputs_dict)

list_of_models = ['FULLY_SUPERVISED_CLASSIFIER',
                  'CGAN_BASIC_SUPERVISED_CLASSIFIER',
                  #'CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER',
                  'CGAN_GUMBEL_SUPERVISED_CLASSIFIER',
                  #'CGAN_GUMBEL_DJ_SUPERVISED_CLASSIFIER',
                  'SSL_GAN',
                  'TRIPLE_GAN',
                  'SSL_VAE',
                  'VAT',
                  'ENTROPY_MINIMISATION',
                  'LABEL_PROPAGATION',
                  'PARTIAL_SUPERVISED_CLASSIFIER']


# for replacing model names in latek table
mreplace_dict = {k: '' for k in list_of_models}
mreplace_dict['FULLY_SUPERVISED_CLASSIFIER']='F-SUP'
mreplace_dict['CGAN_BASIC_SUPERVISED_CLASSIFIER'] = 'CGAN-SSL'
mreplace_dict['CGAN_GUMBEL_SUPERVISED_CLASSIFIER'] = 'GCGAN-SSL'
mreplace_dict['SSL_GAN'] = 'SSL-GAN'
mreplace_dict['TRIPLE_GAN'] = 'TRIPLE-GAN'
mreplace_dict['SSL_VAE'] = 'SSL-VAE'
mreplace_dict['VAT'] = 'VAT'
mreplace_dict['ENTROPY_MINIMISATION'] = 'ENT-MIN'
mreplace_dict['LABEL_PROPAGATION'] = 'L-PROP'
mreplace_dict['PARTIAL_SUPERVISED_CLASSIFIER'] = 'P-SUP'

#for formmatting our latek table with legends
def create_model_colour_str(model_name):
    retval = '\\textcolor{MODEL_NAME}{\LARGE $\\blacksquare$}'.replace('MODEL_NAME', model_name)
    return (retval)

#create colour table reference
colour_df = pd.DataFrame([create_model_colour_str(m) for m in list_of_models])
colour_df.index = list_of_models
colour_df.columns = ['Key']

candidate_names = ['real_sachs_mek_log', 'real_sachs_raf_log','real_bcancer_diagnosis_zscore']
rename_keys={k:'' for k in candidate_names}
rename_keys['real_sachs_mek_log']='MEK'
rename_keys['real_sachs_raf_log']='RAF'
rename_keys['real_bcancer_diagnosis_zscore']='BCANCER'

#returns the data frame after add label key
def copy_legend_rename(results_df):
    # colour_df
    # now delete PARITAL_SUPERVISED_CLASSIFIER and rename psup_baseline
    latek_df = results_df.drop('PARTIAL_SUPERVISED_CLASSIFIER').rename(
        index={'psup_baseline': 'PARTIAL_SUPERVISED_CLASSIFIER'})

    oldcols=[c for c in latek_df.columns]

    latek_df = pd.concat([colour_df, latek_df], axis=1).fillna('')  # fill NaN with empty
    latek_df.rename(index=mreplace_dict, inplace=True)
    latek_df.columns = ['KEY'] + [rename_keys[k] for k in oldcols]
    return(latek_df)




if __name__=='__main__':
    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec=pd.read_excel('combined_spec.xls',sheet_name=None)
    dspec=master_spec['dataset_spec'] #write dataset spec shorthand
    dspec.set_index("d_n",inplace=True) #set idx for easier
    # all datasets
    all_dn=[str(i) for i in master_spec['dataset_spec'].index]
    gaussian_results_dict=collections.OrderedDict()

    rel_dn=np.array([n for n in all_dn if np.any([c in n for c in candidate_names])])

    for k in candidate_names:
        gaussian_results_dict[k]={'group':['real_dat'],
                                  'sheetname':'real_dat'}
    results_dict={}
    si_dict={}
    for d_n in rel_dn: #iterate thru target dataset results in d_n
        print('computing for dn: {0}'.format(d_n))
        csi=master_spec['dataset_si'][d_n].values
        candidate_si=csi[~np.isnan(csi)]
        candidate_si=[int(c) for c in candidate_si]


       # models = ['PARTIAL_SUPERVISED_CLASSIFIER']
        cspec=dspec.loc[d_n] #current spec
        model_accs={}
        all_si=[s for s in candidate_si]
        model_accs['s_i']=all_si
        for current_model in list_of_models:
            vaccs=[]
            for s_i in candidate_si:
                try:
                    vaccs.append(float(get_acc_ulab(dspec.save_folder.loc[d_n],s_i,current_model)))
                except:
                    print('warning error for s_i: {0} and model: {1}'.format(s_i,current_model))
                    vaccs.append(np.mean(vaccs))
            model_accs[current_model]=vaccs
        ma_df=pd.DataFrame(model_accs)
        ma_df.set_index(['s_i'],inplace=True)
        ma_df.to_csv('outputs/raw_results_dn_ulab{0}.csv'.format(d_n))
        md=ma_df
        c_all=[c for c in md.columns]
        mc=copy.deepcopy(md) #deep copy of it
        for c in c_all:
            mc[c]=md[c]-md['PARTIAL_SUPERVISED_CLASSIFIER']
        #transform by getting column - PARTIAL_SUPERVISED_CLASSIFIER
        summaries_c=get_summary_stats(mc)
        results_dict[d_n]=summaries_c
        in_df_means = md['PARTIAL_SUPERVISED_CLASSIFIER'].mean()
        in_df_std = md['PARTIAL_SUPERVISED_CLASSIFIER'].std()
        results_dict[d_n]['psup_baseline']='{:.3f} ± {:.3f}'.format(in_df_means * 100,in_df_std* 100)

    print('pausing here')



    writer = pd.ExcelWriter('outputs/collated_results_real.xlsx', engine='xlsxwriter')

    # from stack overflow: https://stackoverflow.com/questions/29463274/simulate-autofit-column-in-xslxwriter
    # for formatting column widths according to largest cell in column
    #gaussian_results_dict


    pd_dicts = [pd.DataFrame.from_dict(results_dict[k], orient='index') for k in results_dict.keys()]
    #for p, k in zip(pd_dicts, results_dict.keys()):
    #    p.columns = k
    if len(pd_dicts)>0:
        ulab_df_real = pd.concat(pd_dicts, axis=1)
    else:
        ulab_df_real = pd.DataFrame()

    ulab_df_real.columns=[k for k in results_dict.keys()]

    sname='REAL_DATA_RESULTS'
    ulab_df_real.to_excel(writer, sheet_name=sname,startrow=2, startcol=0)
    writer.sheets[sname].write_string(1, 0, 'unlabel_set')
    #get max col width
    idx_max = max([len(str(s)) for s in ulab_df_real.index.values] + [len(str(ulab_df_real.index.name))])

    col_widths = [idx_max] + [max([len(str(s)) for s in ulab_df_real[col].values] + [len(col)]) for col in ulab_df_real.columns]

    for i, width in enumerate(col_widths):
        writer.sheets[sname].set_column(i, i, width)
    writer.save()
    #convert to latek table

    #DROP disjoint models from real results
    drop_idx=['CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER','CGAN_GUMBEL_DJ_SUPERVISED_CLASSIFIER']

    if EXPORT_LATEK:
        latek_df=copy_legend_rename(ulab_df_real)
        with pd.option_context("max_colwidth", 1000):
            latek_df.to_latex("causal_ssl_gan_paper/tables/real_results/unlabelled_real_results.tex",encoding='utf-8', escape=False,column_format='rcccccccc')

