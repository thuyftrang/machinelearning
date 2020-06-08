#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Import Libraries
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


# Read the data file
transactions = pd.read_csv('creditcarddata.csv')
df = pd.DataFrame(transactions)


# In[25]:


df = pd.DataFrame(transactions) 


# In[26]:


# Describe the data
df.describe()


# In[27]:


# Calculate the correlation coefficients 
df_corr = df.corr()
print(df_corr)


# In[28]:


rank = df_corr['Class']
df_rank = pd.DataFrame(rank) 
df_rank = np.abs(df_rank).sort_values(by='Class',ascending=False)
df_rank.dropna(inplace=True)


# In[29]:


# We build the train data set
df_train_all = df[0:150000] 
# Split in two the original dataset and separate it with frauds and no frauds transaction
df_train_1 = df_train_all[df_train_all['Class'] == 1]
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('We have ' + str(len(df_train_1)) +" frauds in the data set")
df_sample=df_train_0.sample(300)
# Gather frauds with no frauds transactions
df_train = df_train_1.append(df_sample)
# Mix the data set
df_train = df_train.sample(frac=1)


# In[30]:


X_train = df_train.drop(['Time', 'Class'],axis=1)
y_train = df_train['Class']
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)


# In[31]:


# Test to see whether the model learns correctly
df_test_all = df[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'],axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)


# In[32]:


X_train_rank = df_train[df_rank.index[1:11]]
X_train_rank = np.asarray(X_train_rank)


# In[33]:


# Test to see whether the model learns correctly
X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)


# In[34]:


# Set SVM Classifier
classifier = svm.SVC(kernel='linear') 


# In[35]:


# Train the model
classifier.fit(X_train, y_train)


# In[36]:


# Predict the data set
prediction_SVM_all = classifier.predict(X_test_all)


# In[37]:


# Print out the confusion matrix
cm = confusion_matrix(y_test_all, prediction_SVM_all)
print(cm)


# In[38]:


print('Our criterion give a result of ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[39]:


print('We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nSo, the probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("the accuracy is : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[40]:


classifier.fit(X_train_rank, y_train) # Then we train our model, with our balanced data train.
prediction_SVM = classifier.predict(X_test_all_rank) #And finally, we predict our data test.


# In[41]:


# Print out the confusion matrix
cm = confusion_matrix(y_test_all, prediction_SVM)
print(cm)


# In[42]:


print('Our criterion give a result of ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[45]:


print('There are ' + str(cm[1][1]) + ' frauds out of' + str(cm[1][1]+cm[1][0]) + ' total frauds detected.')
print('\nThe probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("The accuracy: "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[50]:


classifier_b = svm.SVC(kernel='linear',class_weight={0:0.60, 1:0.40})


# In[51]:


# Train model with the balanced data train
classifier_b.fit(X_train, y_train)


# In[52]:


# Predict all the data set
prediction_SVM_b_all = classifier_b.predict(X_test_all)


# In[53]:


# Print out the confusion matrix
cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
print(cm)


# In[54]:


print('Our criterion give a result of ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[55]:


print('There are ' + str(cm[1][1]) + ' frauds out of' + str(cm[1][1]+cm[1][0]) + ' total frauds detected.')
print('\n The probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("The accuracy in our predicted data: "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[56]:


# Rebalance class weigh
classifier_b = svm.SVC(kernel='linear',class_weight={0:0.60, 1:0.40})


# In[57]:


# Train the model again with balanced data
classifier_b.fit(X_train_rank, y_train)


# In[58]:


# Predict the data test
prediction_SVM = classifier_b.predict(X_test_all_rank)


# In[59]:


# Predict with out test data
cm = confusion_matrix(y_test_all, prediction_SVM)
print(cm)


# In[60]:


print('Our criterion give a result of ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[61]:


print('There are ' + str(cm[1][1]) + ' frauds out of ' + str(cm[1][1]+cm[1][0]) + ' total frauds detected.')
print('\nThe probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("The accuracy in out test data: "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))

