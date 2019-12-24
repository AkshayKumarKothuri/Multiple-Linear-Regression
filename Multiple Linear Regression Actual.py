#!/usr/bin/env python
# coding: utf-8

# # Building Machine Learning Model

# # Part1-Data Preprocessing

# Step1- Importing Libraries for Preprocessing

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 
# Step2- Import Dataset

# In[2]:


dataset =pd.read_csv('50_Startups.csv')


# In[3]:


type(dataset)


# In[4]:


dataset


# In[5]:


dataset.head()


# In[6]:


dataset.head(10)


# Step3- Split Independent and Dpendent Variables

# In[7]:


x= dataset.iloc[:,:4]


# In[8]:


x


# In[9]:


type(x)


# In[10]:


x= dataset.iloc[:,:4].values #convert from dataframe to numpy array


# In[11]:


x


# In[12]:


y= dataset.iloc[:,4:].values


# In[13]:


y


# In[14]:


x.ndim #mandatory to be in 2 dimesion for Linear Regression


# In[15]:


type(x)


# In[16]:


dataset.corr()


# In[17]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[18]:


x[:,3]=lb.fit_transform(x[:,3])


# In[19]:


x


# In[20]:


from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(categorical_features=[3])
x=oh.fit_transform(x).toarray()  


# In[21]:


x


# In[22]:


y= dataset.iloc[:,4].values


# In[23]:


x=x[:,1:]


# In[24]:


x


# In[25]:


import seaborn as sns
sns.heatmap(dataset.corr(),annot=True)


# In[26]:


dataset.hist()


# In[27]:


y


# In[28]:


#y= dataset.iloc[:,1:].values


# In[29]:


#y

If you have null values in the dataset 
# Step6- Split Test and Train Data

# In[30]:


from sklearn.model_selection import train_test_split                #previously cros_validation was used in sklearn
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[31]:


x_train


# In[32]:


x_test


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


lr=LinearRegression()


# In[35]:


lr.fit(x_train,y_train)


# In[36]:


y_predict=lr.predict(x_test)


# In[37]:


y_predict


# In[38]:


y_test


# In[39]:


lr.predict(np.array([[10000,24000,40000,1,0]]))


# In[42]:


from sklearn.metrics import r2_score
r2= r2_score(y_predict,y_test)
r2

