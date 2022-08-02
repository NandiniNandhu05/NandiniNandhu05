#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df['Species'].value_counts()


# In[8]:


sns.countplot(x='Species', data=df)


# In[9]:


sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', data=df, hue='Species')
plt.show()


# In[10]:


df.corr()


# In[11]:


sns.heatmap(df.corr())


# In[12]:


sns.pairplot(df,hue='Species')


# In[13]:


X = df.drop('Species', axis=1)
y = df['Species']


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[16]:


scaler = StandardScaler()


# In[17]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[19]:


log_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)


# In[20]:


penalty = ['l1', 'l2', 'elasticnet']
l1_ratio = np.linspace(0,1,20)
C = np.logspace(0,10,20)   


param_grid = {'penalty': penalty,
             'l1_ratio': l1_ratio,
             'C': C}


# In[21]:


grid_model = GridSearchCV(log_model, param_grid=param_grid)


# In[22]:


grid_model.fit(scaled_X_train, y_train)


# In[23]:


grid_model.best_params_


# In[24]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


# In[25]:


y_pred = grid_model.predict(scaled_X_test)


# In[26]:


y_pred


# In[27]:


accuracy_score(y_test, y_pred)


# In[28]:


confusion_matrix(y_test, y_pred)


# In[29]:


plot_confusion_matrix(grid_model, scaled_X_test, y_test)


# In[30]:


print(classification_report(y_test,y_pred))

