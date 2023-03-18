import pickle
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RandomizedSearchCV

print("Training")
df_train = pd.read_csv('content/new_train.csv', delimiter = ',').drop('Unnamed: 0', axis=1)
df_train.rename(columns = {'0':'y', '1':'x'}, inplace = True )
X_train,y_train = df_train.drop(columns = ['x']).values, df_train['x'].values
param_dist = {"eta0": [ .001, .003, .01, .03, .1, .3, 1, 3]}
linear_regression_model = SGDRegressor(tol=.0001)
n_iter_search = 8
random_search = RandomizedSearchCV(linear_regression_model,
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   cv=3,
                                   scoring='neg_mean_squared_error')
random_search.fit(X_train, y_train)

log_regression_model = SGDRegressor(tol=.0001, eta0 = random_search.best_params_['eta0'])
log_regression_model.fit(X_train, y_train)
filename = 'finalized_model.sav'
pickle.dump(log_regression_model, open(filename, 'wb'))