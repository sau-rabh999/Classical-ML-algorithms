#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # importing dataset

# In[2]:


dataset = pd.read_csv('C:/Users/tanis/Downloads/ML_intern_resources/1.linear_regression/insurance.csv')
dataset.head()


# In[3]:


dataset.shape


# # defining the feature and dependent variable

# In[4]:


x = dataset.iloc[:,0].values
print(x)


# In[5]:


y = dataset.iloc[:,-1].values
print(y)


# # splitting into training and test set

# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[7]:


x_train.shape


# In[8]:


x_test.shape


# # training the simple linear reg model on the training set

# In[9]:


x_train = x_train.reshape(-1, 1)


# In[10]:


x_test = x_test.reshape(-1, 1)


# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# # predicting the test set results

# In[12]:


y_pred = regressor.predict(x_test)


# # visualizing training set results

# In[13]:


plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Plotting the training set')
plt.xlabel('Age (in yrs.)')
plt.ylabel('Insurance Charges')
plt.show()


# # visualizing test set results

# In[14]:


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Plotting the test set')
plt.xlabel('Age (in yrs.)')
plt.ylabel('Insurance Charges')
plt.show()


# # predicting a value

# In[17]:


print(x_test[:5])


# In[18]:


print(y_test[:5])


# In[19]:


regressor.predict([[19]])


# In[20]:


regressor.predict([[57]])


# In[25]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# # parameters (intercept and coefficient)

# In[21]:


regressor.intercept_


# In[22]:


regressor.coef_

