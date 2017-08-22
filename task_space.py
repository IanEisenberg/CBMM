from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from os import path
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# load AE
encoder_loc = path.join('output', 'data_augmentation_encoder.h5')
encoder = load_model(encoder_loc)

decoder_loc = path.join('output', 'data_augmentation_decoder.h5')
decoder = load_model(decoder_loc)

AE_loc = path.join('output', 'data_augmentation_autoencoder.h5')
AE = load_model(AE_loc)

# load data
# old data
data_loc = path.join('Data', 'behavioral_data.csv')
trained_data_df = pd.read_csv(data_loc, index_col=0)
# new data
data_loc = path.join('Data', 'independent_behavioral_data.csv')
new_data_df = pd.read_csv(data_loc, index_col=0)

# reorder data
for column in set(new_data_df.columns)-set(trained_data_df.columns):
    new_data_df.drop(column, axis=1, inplace=True)
for column in set(trained_data_df.columns)-set(new_data_df.columns):
    new_data_df.loc[:,column]=0
joint_index = set(new_data_df.index)&set(trained_data_df.index)
new_data_df = new_data_df.loc[joint_index, trained_data_df.columns]
trained_data_df = trained_data_df.loc[joint_index,:]

# helper functions
def CV_autoencode(X, y, model, CV):
    CV.split(X,y)
    CV_scores = []
    for train_i, test_i in CV.split(X,y):
        xtrain = X[train_i,:]; ytrain=y[train_i,:]
        xtest = X[test_i,:]; ytest=y[test_i,:]
        xtrain = model.predict(scale(xtrain))
        xtest = model.predict(scale(xtest))
        rgr = MultiTaskElasticNetCV()
        rgr.fit(xtrain,ytrain)
        score = rgr.score(xtest,ytest)
        CV_scores.append(score)
    rgr = MultiTaskElasticNetCV()
    rgr.fit(model.predict(scale(X)),y)
    return CV_scores, rgr

# prediction
tasks = np.unique([x.split('.')[0] for x in new_data_df.columns])
scores = {k:{} for k in tasks}
models = {k:{} for k in tasks}
KF = KFold(5)
datasets = {'within': new_data_df, 'across': trained_data_df}
AE_reuse_quotient = pd.Series(index=trained_data_df.columns,
                                       name='reuse', data=0)
for task in tasks:
    print(task)
    for k,v in datasets.items():
        print(k)
        target = new_data_df.filter(regex = task)
        predictors = v.drop(target.columns, axis=1)
        AE_predictors = v.copy()
        AE_predictors.loc[:,target.columns]=0
        # use to look up AE representation
        target_col_index = [v.columns.get_loc(i) for i in target.columns]
        
        print('Running PCA')
        PCA_pipe = make_pipeline(StandardScaler(), PCA(),MultiTaskElasticNetCV())
        scores[task]['PCA_' + k] =  np.mean(cross_val_score(PCA_pipe, 
                                              predictors, target, cv=KF))
        models[task]['AE_' + k] = PCA_pipe
        
        print('Running native')
        native_pipe = make_pipeline(StandardScaler(), MultiTaskElasticNetCV())
        scores[task]['native_' + k] = np.mean(cross_val_score(native_pipe, 
                                     predictors, target, cv=KF))
        models[task]['AE_' + k] = PCA_pipe
    
        print('Running encoded')
        CV_scores, encoded_pipe = CV_autoencode(AE_predictors.values, target.values, 
                                                encoder, KF)
        scores[task]['encoded_' + k] = np.mean(CV_scores)
        models[task]['AE_' + k] = encoded_pipe
        
        print('Running AE')
        CV_scores, AE_pipe = CV_autoencode(AE_predictors.values, target.values, 
                                                    AE, KF)
        scores[task]['AE_' + k] = np.mean(CV_scores)
        models[task]['AE_' + k] = AE_pipe
        
        # print beta weight percentage in original variables
        for i,coef in enumerate(AE_pipe.coef_): 
            reuse = np.sum(abs(coef[target_col_index]))/sum(abs(coef))
            AE_reuse_quotient.loc[target.columns[i]] = reuse
            

pickle.dump(scores, open(path.join('output', 'regression_results.pkl'),'wb'))
pickle.dump(models, open(path.join('output', 'regression_models.pkl'),'wb'))

f = plt.figure(figsize=(30,12))
AE_reuse_quotient.plot(marker='o', xticks=range(0,186))
plt.xticks(rotation=90)
f.savefig(path.join('Plots','AE_reuse_quotient.png'))

        
        
    
        
        
    