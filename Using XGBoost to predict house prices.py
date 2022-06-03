# Using XGBoost to predict house prices


import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

# loading dataset
boston = load_boston()

print(boston.keys()) #boston variable is a dictionary

# checking shape of the dataset
print(boston.data.shape)

print(boston.feature_names)
print (boston.DESCR)

# converting to pandas DAtaFrame
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

# appending Price column
data['PRICE'] = boston.target

# Information about dataset
data.info()

# summary statistics
data.describe()

# separating target variable and other variables
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# converting dataset into XGBoost Dmatrix data structure
data_dmatrix = xgb.DMatrix(data = X, label = y)

# Creating train-test set for cross-validation of results
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiating XGBoost regressor object
xg_reg = xgb.XGBRegressor(objective = 'reg:linear', colsample_bytree = 0.3, max_depth = 5, alpha = 10, n_estimators = 10)

# fitting regressor on the training set and making predictions on the test set
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

# computing rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print('RMSE: {}'.format(rmse)) # RMSE comes out to be around 4.73 per $1000

# k-fold Cross Validation with XGBoost
# creating hyper-parameter dictionary
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

# building 3-fold cross validation by using XGBoost's cv() method
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

cv_results.head()

# printing final boosting round metrix
print(cv_results['test-rmse-mean'].tail(1)) # RMSE comes out to be 3.997 (better than earlier)

# visualizing boosting trees and feature importance

# train model using XGBOost 
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

# plotting first tree with matplotlib library
import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

# examining the importance of each feature within the model
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show() #RM has been given the highest importance score. Thus, XGBoost allows for feature selection this way
