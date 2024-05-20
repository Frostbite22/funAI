import os
import glob
import dspy
import pandas as pd

def load_scone(dirname):
    dfs = []
    for filename in glob.glob(os.path.join(dirname + "/*.csv")):
        df = pd.read_csv(filename,index_col=0)
        df['category'] = os.path.basename(filename).replace('.csv','')
        dfs.append(df)
    data_df = pd.concat(dfs)
