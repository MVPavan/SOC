import numpy as np
import pandas as pd
import seaborn as sns
import pickle

pkl_file = "/home/pavanmv/Pavan/study/SOC/DDP/DATA/soc_db_v3.pkl"
pkl_file_df = "/home/pavanmv/Pavan/study/SOC/DDP/DATA/soc_db_v3_df.pkl"
pkl_file_df_1 = "/home/pavanmv/Pavan/study/SOC/DDP/DATA/soc_db_v3_df_1.pkl"
pkl_file_df_stacked = "/home/pavanmv/Pavan/study/SOC/DDP/DATA/soc_db_v3_df_stacked.pkl"

def concat_df(soc_db, axis=1):
    soc_db_temp={}
    for key, value in soc_db.items():
        soc_db_temp[key] = pd.concat(value, axis=axis)
    soc_db_df = pd.concat(soc_db_temp, axis=axis)
    with open(pkl_file_df, 'wb+') as fp:
        pickle.dump(soc_db_df, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return soc_db_df

def stacker(soc_db):
    with open(pkl_file_df_stacked,'wb+') as f:
        soc_db.columns = soc_db.columns.rename("dataset", level=0)
        soc_db.columns = soc_db.columns.rename("cycles", level=1)
        soc_db.columns = soc_db.columns.rename("params", level=2)
        df = soc_db.stack().stack().stack().unstack("params")\
            .reset_index(leve=[1,2]).rename(columns={"level_0":"samples"})
        soc_db = pickle.dump(df,f)


# with open(pkl_file,'rb') as f:
#     soc_db = pickle.load(f)
# soc_db = concat_df(soc_db)
# stacker(soc_db)

with open(pkl_file_df_stacked,'rb') as f:
    soc_db = pickle.load(f)

# with open(pkl_file_df_1,'rb') as f:
#     soc_db_1 = pickle.load(f)



print("wow")
