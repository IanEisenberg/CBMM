import numpy as np
from os import path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns

# load data
data_loc = path.join('Data', 'meaningful_variables_imputed.csv')
target_loc = path.join('Data', 'targets.csv')

targets = pd.DataFrame.from_csv(target_loc, index_col=0)
data_df = pd.read_csv(data_loc, index_col=0)
data = data_df.values

# linear predictor
rgr = make_pipeline(StandardScaler(), LassoCV())
rgr.fit(data,targets.Age)
plt.scatter(targets.Age,rgr.predict(data))

# no cross validation
rgr = make_pipeline(StandardScaler(), LinearRegression())
rgr.fit(data,targets.Age)
plt.scatter(targets.Age,rgr.predict(data))

# support vector
rgr = make_pipeline(StandardScaler(), SVR())
param_grid = [
  {'svr__C': [1, 10], 'svr__kernel': ['linear']},
  {'svr__C': [1, 10], 'svr__gamma': [0.001, 0.0001], 'svr__kernel': ['rbf']},
 ]
rgr = GridSearchCV(rgr, param_grid, verbose=1, n_jobs=4)

rs = rgr.fit(data, targets.Age)

plt.scatter(targets.Age,rgr.predict(data))

joblib.dump(rgr, path.join('SVM_Age_predictor.pkl'))