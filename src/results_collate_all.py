

import sys
sys.path.append('..')
sys.path.append('../src/')

from benchmarks_utils import *
import glob
import pandas as pd
import numpy as np
import copy
import collections
SEED=10
np.random.seed(SEED)
debug=False



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
                  #'CGAN_BASIC_SUPERVISED_CLASSIFIER',
                  'CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER',
                  #'CGAN_GUMBEL_SUPERVISED_CLASSIFIER',
                  'CGAN_GUMBEL_DJ_SUPERVISED_CLASSIFIER',
                  'SSL_GAN',
                  'TRIPLE_GAN',
                  'SSL_VAE',
                  'VAT',
                  'ENTROPY_MINIMISATION',
                  'LABEL_PROPAGATION',
                  'ASSFSCMR',
                  'SFAMCAMT',
                  'PARTIAL_SUPERVISED_CLASSIFIER']



drop_idx = ['n', 'P-SUP','F-SUP']


def get_colmax(in_col, drop_idx=None):
    # takes in_col as pd series (column) and returns the max value when interpreted as value+-error
    # drop some columns...
    #in_col.loc['F-SUP']='-100 ± 100'
    if drop_idx is not None:
        in_values = in_col.drop(drop_idx).values
    else:
        in_values=in_col
    in_values = [v.replace('nan ± nan','-100 ± 100') for v in in_values]
    in_values = [v.replace('-', '-100 ± 100') for v in in_values]
    values = [v.split(' ± ')[0] for v in in_values]
    values = [float(v) for v in values]
    values = np.array(values)
    max_idx = np.argmax(values)
    max_instance = in_values[max_idx]
    return (max_instance)


from functools import partial




def bold_formatter(x, value):
    """Format a number in bold when (almost) identical to a given value.

    Args:
        x: Input number.

        value: Value to compare x with.

        num_decimals: Number of decimals to use for output format.

    Returns:
        String converted output.

    """
    # Consider values equal, when rounded results are equal
    # otherwise, it may look surprising in the table where they seem identical
    if x == value:
        return f"\\textbf{{{x}}}"
    else:
        return x



# for replacing model names in latek table
mreplace_dict = {k: '' for k in list_of_models}
mreplace_dict['FULLY_SUPERVISED_CLASSIFIER']='F-SUP'
mreplace_dict['CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER'] = 'CGAN-SSL'
mreplace_dict['CGAN_GUMBEL_DJ_SUPERVISED_CLASSIFIER'] = 'GCGAN-SSL'
mreplace_dict['SSL_GAN'] = 'SSL-GAN'
mreplace_dict['TRIPLE_GAN'] = 'TRIPLE-GAN'
mreplace_dict['SSL_VAE'] = 'SSL-VAE'
mreplace_dict['VAT'] = 'VAT'
mreplace_dict['ENTROPY_MINIMISATION'] = 'ENT-MIN'
mreplace_dict['LABEL_PROPAGATION'] = 'L-PROP'
mreplace_dict['LABEL_PROPAGATION'] = 'L-PROP'
mreplace_dict['ASSFSCMR'] = 'Adapt-SSFS'
mreplace_dict['SFAMCAMT'] = 'SSFA-Cor'

mreplace_dict['PARTIAL_SUPERVISED_CLASSIFIER'] = 'P-SUP'

#for formmatting our latek table with legends
def create_model_colour_str(model_name):
    retval = '\\textcolor{MODEL_NAME}{\LARGE $\\blacksquare$}'.replace('MODEL_NAME', model_name)
    return (retval)

#create colour table reference
colour_df = pd.DataFrame([create_model_colour_str(m) for m in list_of_models])
colour_df.index = list_of_models
colour_df.columns = ['Key']

#returns the data frame after add label key
def copy_legend_rename(results_df):
    latek_df = results_df.drop('PARTIAL_SUPERVISED_CLASSIFIER').rename(
        index={'psup_baseline': 'PARTIAL_SUPERVISED_CLASSIFIER'})
    latek_df = pd.concat([colour_df, latek_df], axis=1).fillna('')  # fill NaN with empty
    latek_df.rename(index=mreplace_dict, inplace=True)
    latek_df.columns = ['KEY'] + ['CG{0}'.format(k) for k in range(1, 8)]
    return(latek_df)

if __name__=='__main__':
    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    master_spec=pd.read_excel('/media/krillman/240GB_DATA/codes2/SSL_GCM/combined_spec.xls',sheet_name=None)
    dspec=master_spec['dataset_spec'] #write dataset spec shorthand
    dspec.set_index("d_n",inplace=True) #set idx for easier
    # all datasets
    all_dn=[str(i) for i in master_spec['dataset_spec'].index]
    gaussian_results_dict=collections.OrderedDict()
    ds_idx=[d for d in [36]]#range(23,37)]
    candidate_names=['n{0}_gaussian'.format(k) for k in ds_idx]
    rel_dn=np.array([n for n in all_dn if np.any([c in n for c in candidate_names])])
    
    
    #remove '_10000' and '_100000'
    
    key_10000='_10000'
    key_100000='_100000'
    
    rel_dn_base=[]
    rel_dn_10000=[]
    rel_dn_100000=[]
    
    for r in rel_dn:
        if r.endswith(key_10000):
            rel_dn_10000.append(r)
        if r.endswith(key_100000):
            rel_dn_100000.append(r)
        else:
            rel_dn_base.append(r)


    rel_dn=rel_dn_base

    for k in ds_idx:
        gaussian_results_dict[k]={'group':[f for f in rel_dn if 'n{0}_gaussian'.format(k) in f],
                                  'sheetname':'n{0}_gaussian'.format(k)}

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
        ma_df.to_csv('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/raw_results_dn_ulab{0}.csv'.format(d_n))
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

    writer = pd.ExcelWriter('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/collated_results_synthetic.xlsx', engine='xlsxwriter')

    # from stack overflow: https://stackoverflow.com/questions/29463274/simulate-autofit-column-in-xslxwriter
    # for formatting column widths according to largest cell in column
    #gaussian_results_dict
    for k in gaussian_results_dict.keys():
        if k==16:
            print('pausing here')
        fd_list=gaussian_results_dict[k]['group']
        sname=gaussian_results_dict[k]['sheetname']
        pd_dicts = [pd.DataFrame.from_dict(results_dict[k], orient='index') for k in fd_list]
        for p, k in zip(pd_dicts, fd_list):
            p.columns = [k]
        if len(pd_dicts)>0:
            ulab_df_synthetic = pd.concat(pd_dicts, axis=1)
        else:
            ulab_df_synthetic = pd.DataFrame()




        ulab_df_synthetic.to_excel(writer, sheet_name=sname,startrow=2, startcol=0)
        writer.sheets[sname].write_string(1, 0, 'unlabel_set')
        #get max col width
        idx_max = max([len(str(s)) for s in ulab_df_synthetic.index.values] + [len(str(ulab_df_synthetic.index.name))])

        col_widths = [idx_max] + [max([len(str(s)) for s in ulab_df_synthetic[col].values] + [len(col)]) for col in ulab_df_synthetic.columns]

        for i, width in enumerate(col_widths):
            writer.sheets[sname].set_column(i, i, width)

        #convert to latek table

        #latek_df=copy_legend_rename(ulab_df_synthetic)
       
        
        #select cases with 10000
        
        
        cols=[c for c in ulab_df_synthetic]
        cols_10000=[c for c in cols if c.endswith('_10000')]
        latek_df_10000=copy_legend_rename(ulab_df_synthetic[cols_10000])
        #latek_df=copy_legend_rename(all_results)
        dset_cols=[c for c in latek_df_10000.columns if 'KEY' not in c]
        fmt_bold_max = {column: partial(bold_formatter, value=get_colmax(latek_df_10000[column],drop_idx=drop_idx)) for column in dset_cols}
        fmts = dict(**fmt_bold_max)
        
        
        ulab_synthetic_10000_fn="/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_10000.tex"
        
        
        with pd.option_context("max_colwidth", 1000):
            with open(ulab_synthetic_10000_fn, "w", encoding="utf-8") as fh:
                latek_df_10000.to_latex(buf=fh,
                                escape=False,
                                column_format='rcccccccc',
                                formatters=fmts)

        # with pd.option_context("max_colwidth", 1000):
        #     latek_df_10000.to_latex(ulab_synthetic_10000_fn,encoding='utf-8', escape=False,column_format='rcccccccc')

        import sys
        #exit here cos don't need real reasults anymore..............
        #sys.exit()
                
        with open(ulab_synthetic_10000_fn,'r') as f:
            lines=f.read()

        lines=lines.replace('nan ± nan','-')
        
        with open(ulab_synthetic_10000_fn,'w') as f:
            f.write(lines)
        
        # old_columns=[c for c in all_results.columns]
        
        # dropcols=[c for c in old_columns if '_5000' in c or '_10000' in c]
        
        
        # all_results=all_results.drop(columns=dropcols)

        # if EXPORT_LATEK:
        #     latek_df=copy_legend_rename(all_results)
        #     dset_cols=[c for c in latek_df.columns if 'KEY' not in c]
        #     fmt_bold_max = {column: partial(bold_formatter, value=get_colmax(latek_df[column])) for column in
        #                     dset_cols}
        #     fmts = dict(**fmt_bold_max)

        #     with pd.option_context("max_colwidth", 1000):
        #         with open("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/all_combined_results.tex", "w", encoding="utf-8") as fh:
        #             latek_df.to_latex(buf=fh,
        #                             escape=False,
        #                             column_format='rccccccccccc',
        #                             formatters=fmts)

            
        # latek_df=copy_legend_rename(all_results)
        # dset_cols=[c for c in latek_df.columns if 'KEY' not in c]
        # fmt_bold_max = {column: partial(bold_formatter, value=get_colmax(latek_df[column])) for column in
        #                 dset_cols}
        # fmts = dict(**fmt_bold_max)
            

        #lfn='/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_5000.tex'
        
        # with open(ulab_synthetic_10000_fn,'r') as f:
        #     lines=f.read()

        # lines=lines.replace('nan ± nan','-')
        
        # with open(ulab_synthetic_10000_fn,'w') as f:
        #     f.write(lines)
            

        
        
        
                
        cols=[c for c in ulab_df_synthetic]
        
        cols_5000=[c for c in cols if c.endswith('_5000')]
        
        
        latek_df_5000=copy_legend_rename(ulab_df_synthetic[cols_5000])
        
        #latek_df=copy_legend_rename(all_results)
        dset_cols=[c for c in latek_df_5000.columns if 'KEY' not in c]
        fmt_bold_max = {column: partial(bold_formatter, value=get_colmax(latek_df_5000[column],drop_idx=drop_idx)) for column in dset_cols}
        fmts = dict(**fmt_bold_max)
        
        
        #ulab_synthetic_10000_fn="/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_10000.tex"
        ulab_synthetic_5000_fn="/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_5000.tex"
        
        
        with pd.option_context("max_colwidth", 1000):
            with open(ulab_synthetic_5000_fn, "w", encoding="utf-8") as fh:
                latek_df_5000.to_latex(buf=fh,
                                escape=False,
                                column_format='rcccccccc',
                                formatters=fmts)
                

        
        with open(ulab_synthetic_5000_fn,'r') as f:
            lines=f.read()

        lines=lines.replace('nan ± nan','-')
        
        with open(ulab_synthetic_5000_fn,'w') as f:
            f.write(lines)
            
        #lfn='/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_5000.tex'
        
        # with open(ulab_synthetic_5000_fn,'r') as f:
        #     lines=f.read()

        # lines=lines.replace('nan ± nan','-')
        
        # with open(ulab_synthetic_5000_fn,'w') as f:
        #     f.write(lines)
            
            
            
        cols=[c for c in ulab_df_synthetic]
        cols_5000=[c for c in cols if c.endswith('_5000')]
        cols_10000=[c for c in cols if c.endswith('_10000')]
        
        exclude_cols=cols_5000+cols_10000
        
        remain_cols=[c for c in ulab_df_synthetic if c not in exclude_cols]
        
        
        latek_df_2000=copy_legend_rename(ulab_df_synthetic[remain_cols])
        
        
        
        #latek_df_5000=copy_legend_rename(ulab_df_synthetic[cols_5000])
        
        #latek_df=copy_legend_rename(all_results)
        dset_cols=[c for c in latek_df_2000.columns if 'KEY' not in c]
        fmt_bold_max = {column: partial(bold_formatter, value=get_colmax(latek_df_2000[column],drop_idx)) for column in dset_cols}
        fmts = dict(**fmt_bold_max)
        
        
        #ulab_synthetic_10000_fn="/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_10000.tex"
        ulab_synthetic_2000_fn="/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_2000.tex"
        
        
        with pd.option_context("max_colwidth", 1000):
            with open(ulab_synthetic_2000_fn, "w", encoding="utf-8") as fh:
                latek_df_2000.to_latex(buf=fh,
                                escape=False,
                                column_format='rcccccccc',
                                formatters=fmts)
        
        
        
        print('pausing here')
        
        
        
        
        
        # ulab_synthetic_remain_fn="/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_2000.tex"
        
        # with pd.option_context("max_colwidth", 1000):
        #     latek_df_2000.to_latex(ulab_synthetic_remain_fn,encoding='utf-8', escape=False,column_format='rcccccccc')
            
            
            

        #lfn='/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_5000.tex'
        
        with open(ulab_synthetic_2000_fn,'r') as f:
            lines=f.read()

        lines=lines.replace('nan ± nan','-')
        
        with open(ulab_synthetic_2000_fn,'w') as f:
            f.write(lines)
            


        # lfns='/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_5000_format.tex'


        #import sys
        
        #sys.exit()




        # cols=[c for c in test_df]
        # cols_5000=[c for c in cols if c.endswith('_5000')]
        # latek_df_5000=copy_legend_rename(test_df[cols_5000])
        
        # with pd.option_context("max_colwidth", 1000):
        #     latek_df_5000.to_latex("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_10000.tex",encoding='utf-8', escape=False,column_format='rcccccccc')














        # cols=[c for c in ulab_df_synthetic]
        # try:
        #     cols_100000=[c for c in cols if c.endswith('_100000')]
            
            
        #     latek_df_100000=copy_legend_rename(ulab_df_synthetic[cols_100000])
            
            
            
        #     with pd.option_context("max_colwidth", 1000):
        #         latek_df_100000.to_latex("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_synthetic_results_100000.tex",encoding='utf-8', escape=False,column_format='rcccccccc')
        # except:
        #     continue
        
        

        #writer.save()

    # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    #master_spec = pd.read_excel('combined_spec.xls', sheet_name=None)
    master_spec=pd.read_excel('/media/krillman/240GB_DATA/codes2/SSL_GCM/combined_spec.xls',sheet_name=None)
    
    dspec = master_spec['dataset_spec']  # write dataset spec shorthand
    dspec.set_index("d_n", inplace=True)  # set idx for easier
    # all datasets
    all_dn = [str(i) for i in master_spec['dataset_spec'].index]
    gaussian_results_dict = collections.OrderedDict()
    #ds_idx = [d for d in range(34, 35)]
    candidate_names = ['n{0}_gaussian'.format(k) for k in ds_idx]
    rel_dn = np.array([n for n in all_dn if np.any([c in n for c in candidate_names])])

    rel_dn=rel_dn_base


    for k in ds_idx:
        gaussian_results_dict[k] = {'group': [f for f in rel_dn if 'n{0}_gaussian'.format(k) in f],
                                    'sheetname': 'n{0}_gaussian'.format(k)}

    results_dict = {}
    si_dict = {}
    for d_n in rel_dn:  # iterate thru target dataset results in d_n
        print('computing for dn: {0}'.format(d_n))
        csi = master_spec['dataset_si'][d_n].values
        candidate_si = csi[~np.isnan(csi)]
        candidate_si = [int(c) for c in candidate_si]

        cspec = dspec.loc[d_n]  # current spec
        model_accs = {}
        all_si = [s for s in candidate_si]
        model_accs['s_i'] = all_si
        for current_model in list_of_models:
            vaccs = []
            for s_i in candidate_si:
                try:
                    vaccs.append(float(get_acc_test(dspec.save_folder.loc[d_n], s_i, current_model)))
                except:
                    print('warning error for s_i: {0} and model: {1}'.format(s_i, current_model))
                    vaccs.append(np.mean(vaccs))
            model_accs[current_model] = vaccs
        ma_df = pd.DataFrame(model_accs)
        ma_df.set_index(['s_i'], inplace=True)
        ma_df.to_csv('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/raw_results_dn_ulab{0}.csv'.format(d_n))
        md = ma_df
        c_all = [c for c in md.columns]
        mc = copy.deepcopy(md)  # deep copy of it
        for c in c_all:
            mc[c] = md[c] - md['PARTIAL_SUPERVISED_CLASSIFIER']
        # transform by getting column - PARTIAL_SUPERVISED_CLASSIFIER
        summaries_c = get_summary_stats(mc)
        results_dict[d_n] = summaries_c
        in_df_means = md['PARTIAL_SUPERVISED_CLASSIFIER'].mean()
        in_df_std = md['PARTIAL_SUPERVISED_CLASSIFIER'].std()
        results_dict[d_n]['psup_baseline'] = '{:.3f} ± {:.3f}'.format(in_df_means * 100, in_df_std * 100)

    print('pausing here')

    #writer = pd.ExcelWriter('outputs/collated_results_gauss_mix_newsim.xlsx', engine='xlsxwriter')

    # from stack overflow: https://stackoverflow.com/questions/29463274/simulate-autofit-column-in-xslxwriter
    # for formatting column widths according to largest cell in column
    # gaussian_results_dict
    for k in gaussian_results_dict.keys():
        if k == 16:
            print('pausing here')
        fd_lst = gaussian_results_dict[k]['group']
        sname = gaussian_results_dict[k]['sheetname']
        pd_dicts = [pd.DataFrame.from_dict(results_dict[k], orient='index') for k in fd_list]
        for p, k in zip(pd_dicts, fd_list):
            p.columns = [k]
        if len(pd_dicts) > 0:
            test_df = pd.concat(pd_dicts, axis=1)
        else:
            test_df = pd.DataFrame()
        nrows = test_df.shape[0]

        writer.sheets[sname].write_string(nrows + 4, 0, 'test_set')
        test_df.to_excel(writer, sheet_name=sname, startrow=nrows + 5, startcol=0)

        # get max col width
        idx_max = max([len(str(s)) for s in test_df.index.values] + [len(str(test_df.index.name))])

        col_widths = [idx_max] + [max([len(str(s)) for s in test_df[col].values] + [len(col)]) for col in test_df.columns]

        for i, width in enumerate(col_widths):
            writer.sheets[sname].set_column(i, i, width)




        #convert to latek table
        #latek_df=copy_legend_rename(test_df)



        cols=[c for c in test_df]
        cols_10000=[c for c in cols if c.endswith('_10000')]
        latek_df_10000=copy_legend_rename(test_df[cols_10000])
        
        with pd.option_context("max_colwidth", 1000):
            latek_df_10000.to_latex("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/test_synthetic_results_10000.tex",encoding='utf-8', escape=False,column_format='rcccccccc')



        cols=[c for c in test_df]
        cols_5000=[c for c in cols if c.endswith('_5000')]
        latek_df_5000=copy_legend_rename(test_df[cols_5000])
        
        with pd.option_context("max_colwidth", 1000):
            latek_df_5000.to_latex("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/test_synthetic_results_5000.tex",encoding='utf-8', escape=False,column_format='rcccccccc')



        try:
            cols_100000=[c for c in cols if c.endswith('_100000')]
            
            
            latek_df_100000=copy_legend_rename(test_df[cols_100000])
            
            
            
            with pd.option_context("max_colwidth", 1000):
                latek_df_100000.to_latex("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/test_synthetic_results_100000.tex",encoding='utf-8', escape=False,column_format='rcccccccc')
        except:
            continue
        



        # with pd.option_context("max_colwidth", 1000):
        #     latek_df.to_latex("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/test_synthetic_results.tex",encoding='utf-8', escape=False,column_format='rcccccccc')

    #import sys
    
    #sys.exit()
    
    writer.close()

    print('pausing here')

    # we need to write example of the colour into latek table


    # doing real data

    list_of_models = ['FULLY_SUPERVISED_CLASSIFIER',
                      'CGAN_BASIC_SUPERVISED_CLASSIFIER',
                        'CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER',
                      'CGAN_GUMBEL_SUPERVISED_CLASSIFIER',
                      'CGAN_GUMBEL_DJ_SUPERVISED_CLASSIFIER',
                      'SSL_GAN',
                      'TRIPLE_GAN',
                      'SSL_VAE',
                      'VAT',
                      'ENTROPY_MINIMISATION',
                      'LABEL_PROPAGATION',
                        'ASSFSCMR',
                    'SFAMCAMT',
                      'PARTIAL_SUPERVISED_CLASSIFIER']

    #candidate_names = ['real_sachs_mek_log', 'real_sachs_raf_log', 'real_bcancer_diagnosis_zscore']
    candidate_names = ['real_sachs_raf_log', 'real_sachs_mek_log','real_bcancer_diagnosis_zscore']
    
    rename_keys = {k: '' for k in candidate_names}
    rename_keys['real_sachs_mek_log'] = 'MEK'
    rename_keys['real_sachs_raf_log'] = 'RAF'
    rename_keys['real_bcancer_diagnosis_zscore'] = 'BCANCER'
    rename_keys['n36_gaussian_mixture_d1'] = 'CG1'
    rename_keys['n36_gaussian_mixture_d2'] = 'CG2'
    rename_keys['n36_gaussian_mixture_d3'] = 'CG3'
    rename_keys['n36_gaussian_mixture_d4'] = 'CG4'
    rename_keys['n36_gaussian_mixture_d5'] = 'CG5'
    rename_keys['n36_gaussian_mixture_d6'] = 'CG6'
    rename_keys['n36_gaussian_mixture_d7'] = 'CG7'
    
    rename_keys['n36_gaussian_mixture_d1_10000'] = 'CG1'
    rename_keys['n36_gaussian_mixture_d2_10000'] = 'CG2'
    rename_keys['n36_gaussian_mixture_d3_10000'] = 'CG3'
    rename_keys['n36_gaussian_mixture_d4_10000'] = 'CG4'
    rename_keys['n36_gaussian_mixture_d5_10000'] = 'CG5'
    rename_keys['n36_gaussian_mixture_d6_10000'] = 'CG6'
    rename_keys['n36_gaussian_mixture_d7_10000'] = 'CG7'
    # for replacing model names in latek table
    mreplace_dict = {k: '' for k in list_of_models}
    mreplace_dict['FULLY_SUPERVISED_CLASSIFIER'] = 'F-SUP'
    mreplace_dict['CGAN_BASIC_SUPERVISED_CLASSIFIER'] = 'CGAN-SSL'
    mreplace_dict['CGAN_GUMBEL_SUPERVISED_CLASSIFIER'] = 'GCGAN-SSL'
    mreplace_dict['SSL_GAN'] = 'SSL-GAN'
    mreplace_dict['TRIPLE_GAN'] = 'TRIPLE-GAN'
    mreplace_dict['SSL_VAE'] = 'SSL-VAE'
    mreplace_dict['VAT'] = 'VAT'
    mreplace_dict['ENTROPY_MINIMISATION'] = 'ENT-MIN'
    mreplace_dict['LABEL_PROPAGATION'] = 'L-PROP'
    mreplace_dict['ASSFSCMR'] = 'Adapt-SSFS'
    mreplace_dict['SFAMCAMT'] = 'SSFA-Cor'

    mreplace_dict['PARTIAL_SUPERVISED_CLASSIFIER'] = 'P-SUP'


    # for formmatting our latek table with legends
    def create_model_colour_str(model_name):
        retval = '\\textcolor{MODEL_NAME}{\LARGE $\\blacksquare$}'.replace('MODEL_NAME', model_name)
        return (retval)


    # create colour table reference
    colour_df = pd.DataFrame([create_model_colour_str(m) for m in list_of_models])
    colour_df.index = list_of_models
    colour_df.columns = ['Key']

    # returns the data frame after add label key
    def copy_legend_rename(results_df):
        # colour_df
        # now delete PARITAL_SUPERVISED_CLASSIFIER and rename psup_baseline
        latek_df = results_df.drop('PARTIAL_SUPERVISED_CLASSIFIER').rename(
            index={'psup_baseline': 'PARTIAL_SUPERVISED_CLASSIFIER'})

        oldcols = [c for c in latek_df.columns]

        latek_df = pd.concat([colour_df, latek_df], axis=1).fillna('')  # fill NaN with empty
        latek_df.rename(index=mreplace_dict, inplace=True)
        latek_df.columns = ['KEY'] + [rename_keys[k] for k in oldcols]
        return (latek_df)

   # get dataspec, read in as dictionary
    # this is the master dictionary database for parsing different datasets / misc modifications etc
    #master_spec=pd.read_excel('combined_spec.xls',sheet_name=None)
    master_spec=pd.read_excel('/media/krillman/240GB_DATA/codes2/SSL_GCM/combined_spec.xls',sheet_name=None)
    
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
        ma_df.to_csv('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/raw_results_dn_ulab{0}.csv'.format(d_n))
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



    writer = pd.ExcelWriter('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/collated_results_real.xlsx', engine='xlsxwriter')

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
    writer.close()
    #convert to latek table

    #DROP disjoint models from real results
    drop_idx=['CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER','CGAN_GUMBEL_DJ_SUPERVISED_CLASSIFIER']

    EXPORT_LATEK=True
    
    
    
    
    
    
    if EXPORT_LATEK:
        latek_df=copy_legend_rename(ulab_df_real)
        with pd.option_context("max_colwidth", 1000):
            latek_df.to_latex("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/unlabelled_real_results.tex",encoding='utf-8', escape=False,column_format='rcccccccc')


    #set that nice index

    ulab_df_synthetic.index = ulab_df_real.index


    #now export all togeth

    all_results=pd.concat([ulab_df_synthetic,ulab_df_real],axis=1)



    drop_idx = ['n', 'P-SUP','F-SUP']



    #replace values in results



    all_results=all_results.replace("nan ± nan", "-")

    #DROPPING MEK RESULTS FOR PAPER - 13_02_2023 ARCHER


    import sys
    #exit here cos don't need real reasults anymore..............
    #sys.exit()
    
    
    old_columns=[c for c in all_results.columns]
    
    #dropcols=[c for c in old_columns if '_5000' in c or '_10000' in c]
    dropcols=[c for c in old_columns if ('_10000' not in c and 'real' not in c)]
    
    
    all_results=all_results.drop(columns=dropcols)

    if EXPORT_LATEK:
        latek_df=copy_legend_rename(all_results)
        dset_cols=[c for c in latek_df.columns if 'KEY' not in c]
        fmt_bold_max = {column: partial(bold_formatter, value=get_colmax(latek_df[column],drop_idx=drop_idx)) for column in
                        dset_cols}
        fmts = dict(**fmt_bold_max)

        with pd.option_context("max_colwidth", 1000):
            with open("/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/all_combined_results.tex", "w", encoding="utf-8") as fh:
                latek_df.to_latex(buf=fh,
                                    escape=False,
                                    column_format='rccccccccccc',
                                    formatters=fmts)



    print('pausing here')





#define colours
import json


#color_dict_fn = 'causal_ssl_gan_paper/.json'


#read in these colours

tex_cd_fn='/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/model_colours.tex'

with open(tex_cd_fn,'r') as f:
    cldict={k:'#'+v for k,v in [l.replace('\definecolor{','').replace('}{HTML}{',' ').replace('}\n','').split(' ') for l in f.readlines()]}



#read in our latek



#make DISJOINT same color as regular version


#extra_colours={'CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER':'#EF553B',
#'CGAN_GUMBEL_DJ_SUPERVISED_CLASSIFIER':'#00CC96'}

#color_dict.update(extra_colours)

#with open("jp/colours_dict_bk.json", "w") as fp:
#    json.dump(color_dict,fp)


#add in other two model names...

list_of_models.append('CGAN_BASIC_SUPERVISED_CLASSIFIER')
list_of_models.append('CGAN_GUMBEL_SUPERVISED_CLASSIFIER')

list_of_models=list(set(list_of_models))

#and then save to a json

def define_colours(list_of_models,color_dict_fn):
    # Opening JSON file
    f = open(color_dict_fn)
    # returns JSON object as
    # a dictionary
    color_dict = json.load(f)
    ret_dict={m:color_dict[m] for m in list_of_models}
    return ret_dict




#cd=define_colours(list_of_models=list_of_models,color_dict_fn=color_dict_fn)

cd=cldict
#format out file..



def create_cstring_latek(model_name,hex_str):
    retval='\definecolor{model_name}{HTML}{hex_str}'
    retval=retval.replace('model_name',model_name)
    retval=retval.replace('hex_str',hex_str.replace('#',''))
    return(retval)

model_latek_colours=[create_cstring_latek(m,cd[m]) for m in list_of_models]

with open('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/model_colours.tex','w') as f:
    model_latek_colours=[m+'\n' for m in model_latek_colours]
    f.writelines(model_latek_colours)

f.close()

