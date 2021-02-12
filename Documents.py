#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION 
# 
# 
# ## DATA SCIENCE AND BUSIENESS ANALYTICS 
# 
# 
# 
# - NAME :- ANAND GEED
# 
# 
# - TASK NO:-1--->Predict the percentage of an student based on the no. of study hours.
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


url = ("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data = pd.read_csv(url)


# In[3]:


print(data.shape)
data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[32]:


data.plot(x= "Hours", y= "Scores",style= "o")
plt.title("Hours Vs precentages")
plt.xlabel("Hours studies")
plt.ylabel("percentages scores")


# In[35]:


data.corr(method ="pearson")


# In[34]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[24]:


from sklearn.model_selection import train_test_split
x_test, x_train , y_test , y_train = train_test_split(x,y, test_size =0.2, random_state=20)


# In[25]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
print("training is completed")


# In[26]:


line= reg.coef_*x + reg.intercept_
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[27]:


print(x_test)
y_pred= reg.predict(x_test)


# In[28]:


df = pd.DataFrame({"actual": y_test, "predict": y_pred})
df


# In[29]:


h = 9.25
pred = reg.predict([[h]])
print("if student studies {} per days then he/she score {}".format(h,pred))


# In[36]:


sns.set_style("whitegrid")
sns.distplot(np.array(y_test-y_pred))
plt.show


# In[42]:


from sklearn import metrics
from sklearn.metrics import r2_score
print("Mean Absolute Error",metrics.mean_absolute_error(y_test,y_pred))
print("R2 score:",metrics.r2_score(y_test,y_pred))


# In[ ]:




