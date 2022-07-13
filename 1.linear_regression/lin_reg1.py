#!/usr/bin/env python
# coding: utf-8

# x = np.array([12,13,15,17,22,28])
# y = np.array([3310,4500,3990,3680,5560,6655])
# n = np.size(x)
# x_mean = np.mean(x)
# y_mean = np.mean(y)
# num = np.sum(x*y)-n*x_mean*n*y_mean
# den = np.sum(x*x)-n*x_mean*x_mean
# b1 = num/den
# b0 = y_mean-b1*x_mean
# x = int(input('Enter the no. of rooms: '))
# y_pred = b0+b1*x
# print('The rent/day is: ', y_pred)

# # using libraries

# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.array([12,13,15,17,22,28])
y = np.array([3310,4500,3990,3680,5560,6655])


# In[43]:


x = x.reshape(-1,1)


# In[44]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)


# In[45]:


y_pred = regressor.predict(x)


# In[46]:


plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Visualizing the results')
plt.xlabel('No. of days')
plt.ylabel('Rent/day')
plt.show()


# In[47]:


regressor.predict([[35]])


# In[48]:


regressor.intercept_


# In[49]:


regressor.coef_


# In[50]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # importing dataset

# In[51]:


dataset = pd.read_csv('C:/Users/tanis/Downloads/ML_intern_resources/1.linear_regression/insurance.csv')
dataset.head()


# In[52]:


dataset.shape


# # defining the feature and dependent variable

# In[53]:


x = dataset.iloc[:,0].values
print(x)


# In[54]:


y = dataset.iloc[:,-1].values
print(y)


# # splitting into training and test set

# In[55]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[56]:


x_train.shape


# In[57]:


x_test.shape


# # training the simple linear reg model on the training set

# In[58]:


x_train = x_train.reshape(-1, 1)


# In[59]:


x_test = x_test.reshape(-1, 1)


# In[60]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# # predicting the test set results

# In[61]:


y_pred = regressor.predict(x_test)


# # visualizing training set results

# In[62]:


plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Plotting the training set')
plt.xlabel('Age (in yrs.)')
plt.ylabel('Insurance Charges')
plt.show()


# # visualizing test set results

# In[63]:


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Plotting the test set')
plt.xlabel('Age (in yrs.)')
plt.ylabel('Insurance Charges')
plt.show()


# # predicting a value

# In[64]:


print(x_test)


# In[65]:


print(y_test)


# In[66]:


regressor.predict([[19]])


# In[67]:


regressor.predict([[57]])


# In[68]:


regressor.predict([[51]])


# In[69]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# # parameters (intercept and coefficient)

# In[22]:


regressor.intercept_


# In[23]:


regressor.coef_

