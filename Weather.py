#!/usr/bin/env python
# coding: utf-8

# In[17]:


#importation de numpy at pandas

import numpy as np # linear algebra
import pandas as pd # data processing


# In[18]:


#reading data
data = pd.read_csv(r'C:\Users\ghanm\Downloads\634_1203_compressed_weatherHistory.csv\weatherHistory.csv')
data.head()


# In[19]:


#dropping loud couver and daily summary columns
data=data.drop(['Loud Cover','Daily Summary'],axis=1)
data.head()


# In[20]:


#extracting month from date column
data['Formatted Date']=pd.to_datetime(data['Formatted Date'],utc=True)
data['month']=data['Formatted Date'].dt.month
print(data['month'])


# In[21]:


#dropping formatted date column
data=data.drop('Formatted Date',axis=1)
data.info()


# In[22]:


#label encoding precip type column
print(data['Precip Type'].value_counts())
data['Precip Type']=data['Precip Type'].replace(['rain','snow'],[0,1])
print(data['Precip Type'].value_counts())


# In[23]:


#label encoding summary type column
print(data['Summary'].value_counts())
data['Summary']=data['Summary'].replace(data['Summary'].unique(),range(0,len(data['Summary'].unique())))
print(data['Summary'].value_counts())


# In[26]:


#skew
for i in data.columns.values:
    print(i+"\t"+str(data[i].skew()))


# In[27]:


#replacing nan values
data['Precip Type']=data['Precip Type'].replace(np.nan,0)


# In[28]:


#reducing skewness
print(np.sqrt(data['Summary']).skew())
data['Summary']=np.sqrt(data['Summary'])
print(np.sqrt(data['Wind Speed (km/h)']).skew())
data['Wind Speed (km/h)']=np.sqrt(data['Wind Speed (km/h)'])
print(data.shape)
data=data.loc[data['Pressure (millibars)']>0,:]
print(data.shape)
print(data['Pressure (millibars)'].skew())


# In[29]:


y=pd.DataFrame()
y=data['Apparent Temperature (C)']
data=data.drop('Apparent Temperature (C)',axis=1)


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
X_train,X_test,y_train,y_test=train_test_split(data,y,random_state=0)


# In[31]:


reg=DecisionTreeRegressor(random_state=0)
reg.fit(X_train,y_train)


# In[32]:


# error
from sklearn.metrics import mean_squared_error
from math import sqrt
weather=sqrt(mean_squared_error(y_test,reg.predict(X_test)))
print(weather)


# In[33]:


#error
print(reg.score(X_test,y_test))
print(reg.score(X_train,y_train))

