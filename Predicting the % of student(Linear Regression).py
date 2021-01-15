#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Url = "http://bit.ly/w-data"
dataset = pd.read_csv(Url)
print("Data Imported")


# In[6]:


#exploring data
dataset.shape


# In[8]:



dataset.describe()


# In[9]:


dataset.head()


# In[10]:


#Plotting of data
dataset.plot(x="Hours",y="Scores",style="o")
plt.title("Student Marks Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Percentage Scores")
plt.show()


# In[30]:


#preparing data
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
print("Data Prepared")


# In[31]:


#Spliting of data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[32]:


#Training Algorith(Linear regression)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)


# In[33]:


print(regressor.intercept_)


# In[34]:


print(regressor.coef_)


# In[35]:


#predicting model
y_pred = regressor.predict(X_test)


# In[36]:


df = pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})
df
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[37]:


df = pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})


# In[38]:


df


# In[42]:


#Ploting best fit line
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Study hours V/s percentage (using linear regression)")
plt.xlabel("Study hours")
plt.ylabel("percentage")
plt.show()


# In[44]:


#evaluating algo
from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,y_pred))
print('Mean Square Error:',metrics.mean_squared_error(Y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(Y_test,y_pred)))


# In[52]:


#predicting for 9.25 hours
Hours = 9.25
Pred_percentage= regressor.predict([[Hours]])
print(f'no. of hours={Hours}')
print(f'Predicted Score={Pred_percentage[0]}')


# In[ ]:




