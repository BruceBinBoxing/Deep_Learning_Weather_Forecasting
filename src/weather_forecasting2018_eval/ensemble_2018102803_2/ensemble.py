
# coding: utf-8

# In[8]:


import glob 
import pandas as pd
  
#获取指定目录下的所有图片 
files = glob.glob('seq2seq*')
dfs=[]

print("Write ...")
for f in files:
    print(f)
    dfs.append(pd.read_csv(f))
df_ensemble = pd.DataFrame(columns=[dfs[0].columns])
df_ensemble.FORE_data = dfs[0].FORE_data
df_ensemble[['       t2m', '      rh2m', '      w10m']] = 0

df_ensemble = pd.DataFrame(columns=[dfs[0].columns])
df_ensemble.FORE_data = dfs[0].FORE_data
df_ensemble[['       t2m', '      rh2m', '      w10m']] = 0

for i in range(len(dfs)):
    df_ensemble[['       t2m', '      rh2m', '      w10m']] += dfs[i][['       t2m', '      rh2m', '      w10m']].values
df_ensemble[['       t2m', '      rh2m', '      w10m']] = df_ensemble[['       t2m', '      rh2m', '      w10m']].values / len(dfs)

df_ensemble.to_csv('./ensemble_avg.csv', index=False)
print("Into ",'./ensemble_avg.csv')
print("Now you can re-name the file for submitting.")
