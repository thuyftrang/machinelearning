#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set(style='whitegrid')


# In[2]:


# Read the data file
transactions = pd.read_csv('creditcarddata.csv')
transactions.describe()


# In[3]:


# Check if there is any null value in the file.
# Luckily, there is no null value
transactions.isnull().any().any()


# In[4]:


# Check the balance of the class
# 0 is nonfraudulent transaction, 1 is fraud transaction
transactions['Class'].value_counts()


# In[5]:


# Check the percentage of each type of transaction
transactions['Class'].value_counts(normalize=True)


# In[6]:


# Split off the data set into features and response variables
X = transactions.drop(labels='Class', axis=1) 
y = transactions.loc[:,'Class']               
del transactions                              


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


# The test size is 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
del X, y


# In[9]:


X_train.shape


# In[10]:


X_test.shape


# In[11]:


# Prevent view warnings
X_train.is_copy = False
X_test.is_copy = False


# In[12]:


# Descrbe 'time' variables
X_train['Time'].describe()


# In[13]:


X_train.loc[:,'Time'] = X_train.Time / 3600
X_test.loc[:,'Time'] = X_test.Time / 3600


# In[14]:


X_train['Time'].max() / 24


# In[15]:


# Descriptive statistics of variable 'Amount'
X_train['Amount'].describe()


# In[16]:


# Check the skewness
X_train['Amount'].skew()


# In[17]:


X_train.loc[:,'Amount'] = X_train['Amount'] + 1e-9
X_train.loc[:,'Amount'], maxlog, (min_ci, max_ci) = sp.stats.boxcox(X_train['Amount'], alpha=0.01)


# In[18]:


maxlog


# In[19]:


(min_ci, max_ci)


# In[20]:


# Check the variable 'Amount' again after shifting all amounts by 10^-9
X_train['Amount'].describe()


# In[21]:


X_train['Amount'].skew()


# In[22]:


X_test.loc[:,'Amount'] = X_test['Amount'] + 1e-9 # Shift all amounts by 1e-9
X_test.loc[:,'Amount'] = sp.stats.boxcox(X_test['Amount'], lmbda=maxlog)


# In[23]:


# Plot the relationship between the transactions amount and time of day
sns.jointplot(X_train['Time'].apply(lambda x: x % 24), X_train['Amount'], 
              kind='hex', stat_func=None, size=12, xlim=(0,24), 
              ylim=(-7.5,14)).set_axis_labels('Time of Day (hr)','Transformed Amount')


# In[24]:


# Compare the descriptive stats of the PCA variables
pca_vars = ['V%i' % k for k in range(1,29)]


# In[25]:


# Describe statistics
X_train[pca_vars].describe()


# In[26]:


from sklearn.feature_selection import mutual_info_classif


# In[27]:


mutual_infos = pd.Series(data=mutual_info_classif(X_train, y_train, discrete_features=False, random_state=1), 
                         index=X_train.columns)


# In[28]:


# Calculate mutual information of each variable
mutual_infos.sort_values(ascending=False)


# In[29]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.linear_model import SGDClassifier


# In[30]:


pipeline_sgd = Pipeline([
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])


# In[31]:


# Conduct a grid search that uses 5 folds for train/ validation splits
param_grid_sgd = [{
    'model__loss': ['log'],
    'model__penalty': ['l1', 'l2'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20)
}, {
    'model__loss': ['hinge'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20),
    'model__class_weight': [None, 'balanced']
}]


# In[32]:


# Use MCC as scoring metric
MCC_scorer = make_scorer(matthews_corrcoef)
grid_sgd = GridSearchCV(estimator=pipeline_sgd, param_grid=param_grid_sgd, scoring=MCC_scorer,
                        n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)


# In[33]:


# Start using random forest model
pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])


# In[34]:


param_grid_rf = {'model__n_estimators': [75]}


# In[35]:


grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=MCC_scorer,
                       n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)


# In[36]:


# Perform grid search
grid_rf.fit(X_train, y_train)


# In[37]:


# Check the performance
grid_rf.best_score_


# In[38]:


grid_rf.best_params_


# In[39]:


from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, cohen_kappa_score, accuracy_score, average_precision_score, roc_auc_score


# In[40]:


# Evaluate random forest on the test set
def classification_eval(estimator, X_test, y_test):
    """
    Print several metrics of classification performance of an estimator, given features X_test and true labels y_test.
    
    Input: estimator or GridSearchCV instance, X_test, y_test
    Returns: text printout of metrics
    """
    y_pred = estimator.predict(X_test)
    
    # Number of decimal places based on number of samples
    dec = np.int64(np.ceil(np.log10(len(y_test))))
    
    print('CONFUSION MATRIX')
    print(confusion_matrix(y_test, y_pred), '\n')
    
    print('CLASSIFICATION REPORT')
    print(classification_report(y_test, y_pred, digits=dec))
    
    print('SCALAR METRICS')
    format_str = '%%13s = %%.%if' % dec
    print(format_str % ('MCC', matthews_corrcoef(y_test, y_pred)))
    if y_test.nunique() <= 2: # Additional metrics for binary classification
        try:
            y_score = estimator.predict_proba(X_test)[:,1]
        except:
            y_score = estimator.decision_function(X_test)
        print(format_str % ('AUPRC', average_precision_score(y_test, y_score)))
        print(format_str % ('AUROC', roc_auc_score(y_test, y_score)))
    print(format_str % ("Cohen's kappa", cohen_kappa_score(y_test, y_pred)))
    print(format_str % ('Accuracy', accuracy_score(y_test, y_pred)))


# In[41]:


# Print out the result of the accuracy in the test set
classification_eval(grid_rf, X_test, y_test)

