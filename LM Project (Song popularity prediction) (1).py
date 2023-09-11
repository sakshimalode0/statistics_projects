#!/usr/bin/env python
# coding: utf-8

# ### Importing Required Libraries

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Setting the dimensions of visualizations

# In[3]:


sns.set(rc={'figure.figsize':(8,6)})


# ### Importing Data Set

# In[4]:


df=pd.read_csv('song data.csv')


# In[ ]:


df


# * Hit song = 1
# * Flop song = 0

# ### Shuffling the data
# * For unbiasedness

# In[5]:


df2=df.sample(frac=1)


# In[6]:


df2


# ### Knowing the data

# In[51]:


df.info()


# ### Checking Null values

# In[52]:


df.isnull().sum()


# * No null value in any column

# ### Descriptive Statistics

# * Summary of Numerical variables (By default)

# In[53]:


des_stat=df.describe()
print(des_stat)


# ### Dropping unwanted columns

# In[10]:


col=['id','name']
df=df.drop(columns=col,axis=1)


# In[11]:


df


# ### EXPLORATORY DATA ANALYSIS

# ### Finding Correlation

# In[54]:


cor_mat=df.corr()
cor_mat


# In[13]:


sns.heatmap(cor_mat,cmap='YlGnBu',annot=True)


# In[55]:


sns.distplot(df['loudness'],kde=False,bins=8)


# In[56]:


sns.boxplot('Status','energy',data=df,showmeans=True,meanprops={'marker':'*','markerfacecolor':'white','markeredgecolor':'black','markersize':'10'}) #corr=0.48


# In[57]:


sns.boxplot('Status','duration',data=df,showmeans=True,meanprops={'marker':'*','markerfacecolor':'white','markeredgecolor':'black','markersize':'10'}) #corr=-0.7


# In[58]:


sns.boxplot('Status','loudness',data=df,showmeans=True,meanprops={'marker':'*','markerfacecolor':'white','markeredgecolor':'black','markersize':'10'}) #corr=0.56


# In[59]:


sns.distplot(df['duration'],bins=10)


# In[60]:


plt.hist(df['duration'],color='purple',edgecolor='white',bins=10)
plt.title('Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()


# In[89]:


sns.countplot(df['mode'])


# In[20]:


Index=[]
for i in range(0,200):
    Index.append(i)
#Index


# In[21]:


df['index']=Index
df


# In[22]:


sns.regplot(y=df['duration'],x=df['index'],fit_reg=True)


# * Duration is strongly Significant

# In[23]:


sns.regplot(y=df['energy'],x=df['index'],fit_reg=True)


# * Energy is Significant

# In[24]:


sns.regplot(y=df['key'],x=df['index'],fit_reg=True)  


# * Key is not Significant

# In[25]:


sns.regplot(y=df['loudness'],x=df['index'],fit_reg=True)


# * Loudness is Significant

# In[26]:


sns.regplot(y=df['mode'],x=df['index'],fit_reg=True)


# * Mode is Not Significant

# In[27]:


sns.regplot(y=df['speechiness'],x=df['index'],fit_reg=True)


# * Weak Significant

# In[28]:


sns.regplot(y=df['acousticness'],x=df['index'],fit_reg=True)


# * Weakly Significant

# In[29]:


sns.regplot(y=df['liveness'],x=df['index'],fit_reg=True)


# * Not Significant

# In[30]:


sns.regplot(y=df['valence'],x=df['index'],fit_reg=True)


# * Not Significant

# In[31]:


sns.regplot(y=df['tempo'],x=df['index'],fit_reg=True)


# * Not Significant

# In[32]:


sns.regplot(y=df['danceability'],x=df['index'],fit_reg=True)


# * Not Significant

# # Logistic Regression

# * dependent variable = Status
# * independent variables - duration, energy, loudness, speechiness, acousticness 

# In[61]:


all_var=list(df.columns)


# In[62]:


print(all_var)


# ### Independent Variables X

# In[63]:


indp_var=list(set(all_var)-set(['key','mode','liveness','valence','tempo','danceability','Status','index']))


# In[64]:


indp_var


# In[65]:


# Extracting independent variable values


# In[66]:


x=df[indp_var].values


# In[67]:


x.shape


# ### Dependent Variable Y

# In[69]:


y = df['Status'].values


# In[70]:


y.shape


# ### Splitting the data 

# In[71]:


from sklearn.model_selection import train_test_split # to partition the data
from sklearn.linear_model import LogisticRegression # For logistic regression
from sklearn.metrics import accuracy_score, confusion_matrix # importing performance matrix for accuracy score and confusion matix


# In[72]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
# test data = 20% for both x and y
# random_state=0 means output of data will not change each time if the command is run several times


# In[73]:


print(train_x.shape)
# 80% of x


# In[74]:


test_x.shape


# In[75]:


train_y.shape


# In[76]:


test_y.shape


# ### Fit the Logistic Regression Model

# In[77]:


logistic=LogisticRegression() # Making the model easy to call


# In[78]:


logistic.fit(train_x,train_y) # train_x is independent variable, train_y is dependent variable


# ### Coefficient and Intercept

# In[79]:


logistic.coef_


# In[80]:


logistic.intercept_


# ### Prediction on Test data

# * Put test_x in prediction function so we will get predicted values of y
# * Then these values can be compared with the actual test_y data set

# In[81]:


prediction=logistic.predict(test_x) # prediction = predicted values of y


# In[82]:


prediction.shape


# ### Confusion Matrix

# * Used to check the accuracy of predicted values
# * Usually return the correct and incorrect predictions
# * Diagonal values are correctly classified values
# * Off diagonal values are incorrectly classified values

# In[83]:


conf_mat=confusion_matrix(test_y,prediction)
print(conf_mat)


# ###### Accuracy Score

# In[84]:


acc=accuracy_score(test_y,prediction)


# In[85]:


acc


# In[ ]:





# In[ ]:





# # Linear Regression

# In[87]:


# Dataset
df2 #shuffle


# In[99]:


sns.regplot(df['energy'],df['loudness'],color='purple') # corr = 0.83 


# * There is a strong correlation between loudness and energy
# * We can predict energy by measuring loudness in a song
# * If loudness is high, energy will be high, then the song should be hit
# * This can be done using simple linear regression

# * Dependent variable - energy
# * Independent variable - loudness

# In[103]:


sns.distplot(df['energy'],bins=5)


# In[105]:


sns.distplot(df['loudness'],bins=5)


# ### Extracting Values of dependent y and independent variable x

# In[139]:


y1=pd.array(df['energy']) #dependent
y1


# In[140]:


x1=pd.array(df['loudness']).reshape(-1,1) #independent
x1


# In[141]:


train_x1,test_x1,train_y1,test_y1=train_test_split(x1,y1,test_size=0.2,random_state=0)


# In[142]:


print(train_x1.shape)
print(test_x1.shape)
print(train_y1.shape)
print(test_y1.shape)


# In[143]:


from sklearn.linear_model import LinearRegression


# In[144]:


lr=LinearRegression()


# In[145]:


lr.fit(train_x1,train_y1)


# In[146]:


lr.coef_


# In[147]:


lr.intercept_


# ### Accuracy Score

# In[149]:


r_sq=lr.score(x1,y1)
print(r_sq)


# * 68.66% is the accuaracy score for this model

# In[1]:


y2=pd.array(df['energy'])


# In[ ]:


x1=pd.array(df['loudness']).reshape(-1,1)

