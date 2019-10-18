#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pickle


# In[2]:


data = pd.read_csv("dataset/Insurance.csv")


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


sns.distplot(data.expenses, hist=False, rug=True);


# In[6]:


male_exp_M = data[['sex','age','expenses']]
male_exp_M = male_exp_M[male_exp_M.sex=='male']
male_exp_M.nlargest(10,'expenses')


# In[7]:


sns.scatterplot(x='age',y='expenses',data = data, hue= 'smoker')


# In[8]:


sns.scatterplot(x='age',y='expenses',data = data, hue= 'sex')


# In[9]:


data.groupby('sex')['sex','expenses'].sum()


# In[10]:


data.groupby('sex')['sex','smoker'].count()


# In[11]:


data.head()


# In[12]:


corr = data.corr()
corr


# In[13]:


sns.heatmap(corr, cmap="YlGnBu")


# In[14]:


data.replace(('yes', 'no'), (1, 0), inplace=True)
data.replace(('male', 'female'), (1, 0), inplace=True)
data.replace(('northeast', 'northwest','southeast','southwest'), (1, 2,3,4), inplace=True)
data.head()


# In[15]:


# Import the dataset
x = data[['age','bmi','smoker','sex','children','region']]
y = data.iloc[:,-1].values
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size = 0.30)


# In[16]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

### MODELFITSTRAIN - use the LinearRegression() function on my training data 
lm.fit(x_train,y_train)


# In[17]:


print(lm.intercept_)


# In[18]:


pd.DataFrame(data=lm.coef_,index=x.columns,columns=['Coeffecients'])


# In[19]:


pred_y = lm.predict(x_test)


# In[20]:


### Visualtion of predictive results, y_test is actual y, pred_y is the predicted 
sns.scatterplot(y_test,pred_y)


# In[21]:


sns.distplot((y_test-pred_y))


# In[22]:


## import the sklearn metrics package
from sklearn import metrics


# In[23]:


metrics.mean_absolute_error(y_test,pred_y)


# In[24]:


numpy.sqrt(metrics.mean_squared_error(y_test,pred_y))


# In[25]:


lm.score(x_test,y_test)


# In[26]:


x_test.head()


# In[27]:


pred_y = lm.predict(x_test.iloc[43:44])


# In[28]:


pred_y


# In[29]:


x_test.iloc[43:44]


# In[31]:


# Saving model to disk
pickle.dump(lm, open('C:/Data Science Material/Deployment Project/Insurance Deployment/model.pkl','wb'))


# In[33]:


# Loading model to compare the results
model = pickle.load(open('C:/Data Science Material/Deployment Project/Insurance Deployment/model.pkl','rb'))
print(model.predict([[46, 19.9, 0,1,0,2]]))


# In[ ]:




