
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[7]:

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

data = pd.read_csv('Recommendation_Data.csv',  parse_dates=['dd_executiontime'], date_parser=dateparse)


# In[8]:

data.info()
data.head(10)


# In[6]:

data['dd_reportname'].groupby([])


# In[22]:

data['report_fact'] = data['dd_factname']+"-"+data['dd_reportname']
data['report_fact'].head()


# In[36]:

prev = None
newindex = 0
report_fact = []
output = []
for i, value in data['report_fact'].iteritems():
    if len(output) != 0 and output[-1] != value:
        prev = output[-1]
    output.append(value)
    report_fact.append(prev)


# In[221]:

tmp = pd.DataFrame(report_fact, output).reset_index()
tmp.columns = ["target","prev_click"]
tmp = tmp[["prev_click","target"]]
tmp = tmp[1:]


# In[222]:

tmp.head()


# <h2> Preprocess

# In[223]:

import sys
sys.path.extend(['/Users/sshetty/Dropbox/Datahack/LoanPrediction/code'])

import preprocess_util

prev_label = preprocess_util.label_to_numeric(tmp['prev_click'])
tmp['prev_click'] = prev_label[1]

# target_label = preprocess_util.label_to_numeric(tmp['target'])
# tmp['target'] = target_label[1]



# <h2> Modelling 

# In[224]:

train_tmp = tmp[0:int(len(tmp)*.8)]
test_tmp = tmp[int(len(tmp)*.8):]


# In[225]:

train_tmp.head()


# In[226]:

def validateModel(model, X, y, parameters={}):
    from sklearn.grid_search import GridSearchCV
    from sklearn import cross_validation
    # Simple K-Fold cross validation. 10 folds.
    cv = cross_validation.StratifiedKFold(y, n_folds=10)
    
    model = GridSearchCV(estimator= model, cv=5, scoring ="accuracy", param_grid=parameters,n_jobs=-1,verbose=True)
    fit = model.fit(X,y)
    print model.best_score_
    print model.best_params_
    return model,fit
#     scores = cross_validation.cross_val_score(estimator=eclf, X=X, y=y,cv=cv,scoring='accuracy',n_jobs=-1)


# In[306]:

get_ipython().magic(u'pinfo LogisticRegression')


# In[ ]:

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import EnsembleClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

# # Create Bayes
# bayes = GaussianNB()

# model =bayes
# Create Gradient booster
gbm =GradientBoostingClassifier(n_estimators=100)
model =gbm

# # Create LR
# regr = LogisticRegression(random_state=45, max_iter= 200, solver = 'newton-cg',n_jobs=-1)
# # model = regr
# eclf1 = EnsembleClassifier(clfs=[regr,gbm,bayes], voting='hard')

# cv = cross_validation.StratifiedKFold(train_tmp['target'], n_folds=10)
# grid= GridSearchCV(estimator = eclf1,cv=cv, param_grid={})
# grid.fit(train_tmp[['prev_click']],train_tmp['target'])

# fit = model.fit(train_tmp[['prev_click']],train_tmp['target'])




# In[301]:

test_X= test_tmp.loc[test_tmp.prev_click.isin(train_tmp.prev_click),"prev_click"]
test_Y= test_tmp.loc[test_tmp.prev_click.isin(train_tmp.prev_click),"target"]


# In[302]:

predict_prob = model.predict_proba(pd.DataFrame(test_X))
predit_df = pd.DataFrame(predict_prob, columns=model.classes_)
# Get top 3 predictions
tmp_predit_df = predit_df.apply(lambda x: ','.join(x.sort(inplace=False,ascending=False).head(3).index), axis=1)
predit_df = pd.DataFrame(list(tmp_predit_df.apply(lambda x: x.split(','))))


# In[303]:

from sklearn.metrics import accuracy_score
print "Giving next three recommendaiton of report to the user gives a accuracy of {}%".format(100* sum(predit_df.apply(lambda x: accuracy_score(test_Y,x)) ))


# In[ ]:



