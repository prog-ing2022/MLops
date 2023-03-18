import pandas as pd
import pickle

print("Result")

df_test = pd.read_csv('content/new_test.csv', delimiter = ',').drop('Unnamed: 0', axis=1)
df_test.rename(columns = {'0':'y', '1':'x'}, inplace = True )
X_test,y_test = df_test.drop(columns = ['x']).values, df_test['x'].values
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model.score(X_test, y_test))