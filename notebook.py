import pandas as pd
from sklearn.preprocessing import LabelEncoder

#%%
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

df = pd.concat([df_train.drop(columns = ['target']), df_test], axis=0).sort_values(by = 'id')

#%%
print(df_train.info())
print(df_train.isna().sum())

#%%
cat_cols = [col for col in df_train.columns if 'cat' in col]
cont_cols = [col for col in df_train.columns if 'cont' in col]

#%% Label Encoding

lbl = LabelEncoder()

for col in cat_cols:
    lbl.fit(df[col])
    df_train[col] = lbl.transform(df_train[col])
    df_test[col] = lbl.transform(df_test[col])

