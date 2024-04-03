import pandas as pd

import glob

results_in_folder=glob.glob('/media/krillman/240GB_DATA/codes2/SSL_GCM/collating_results/revision_method_1_results/*.csv')[0:1]


rf_dict={r.split('/')[-1].replace('_results.csv',''):r for r in results_in_folder}
model_name='ASSFSCMR'


for r in rf_dict.keys():
    
    current_result=pd.read_csv(rf_dict[r])
    
    current_result.columns=['s_i','ulab_acc']
    
    current_result.set_index('s_i',inplace=True)
    #/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_n36_gaussian_mixture_d2_5000/saved_models/CGAN_BASIC_DJ_SUPERVISED_CLASSIFIER-s_i=0_test_acc.out
    
    for si in current_result.index:
        ulab_acc=current_result.loc[si].ulab_acc.item()
        
        out_fn=f'/media/krillman/240GB_DATA/codes2/SSL_GCM/data/dataset_{r}/saved_models/{model_name}-s_i={si}_ulab_acc.out'
        with open(out_fn,'w') as f:
            f.write(str(ulab_acc)+'\n')
            
    print(r)
    print(si)

    