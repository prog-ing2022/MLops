import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

print("Standartization")
df_train = pd.read_csv('train/train.csv', delimiter = ',')
df_test = pd.read_csv('test/test.csv', delimiter = ',')
X_train,y_train = df_train.drop(columns = ['x']).values, df_train['x'].values
X_test,y_test = df_test.drop(columns = ['x']).values, df_test['x'].values
scaler = StandardScaler()
scaler.fit_transform(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
df_train = pd.DataFrame(data=np.column_stack([X_train.flatten(), y_train]))
df_test = pd.DataFrame(data=np.column_stack([X_test.flatten(), y_test]))
df_train.to_csv('content/new_train.csv',index=True)
df_test.to_csv('content/new_test.csv',index=True)