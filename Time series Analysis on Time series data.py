#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[6]:


data = pd.read_csv("C:\\Users\\Dell\\Downloads\\gld_price_data.csv")
gold_data = data[['Date','GLD']]
gold_data.columns = ['date','gold_price']
gold_data.head()


# The first step is to set date column as the index in the dataframe.
# For this, first we convert dates in pandas datatime using
# pd.to_datetime function and then set it as the index of dataframe.

# In[7]:


gold_data.index = pd.to_datetime(gold_data['date'])
gold_data.drop('date',axis=1,inplace = True)


#  We can select data based on year, month and date by setting the date as index. 
# It is very useful if you set date as index in pandas Dataframe.
# Here I am selecting January,2018 to see what are the prices of gold in that particular month.

# In[11]:


gold_data['2018-01']


#  Now we plot a chart to see how the prices of gold varies from 2008 to 2018. 
# Plotting a chart of series help us to visualise data take a quick glance to get an intution of dataset. 
# Here we can clearly see that gold prices are very high in the time period of 2011 to 2013. 
# For plotting a graph we use of python matplotlib library.
# 

# In[12]:


import matplotlib.pyplot as plt
plt.plot(gold_data)
plt.xlabel('Year')
plt.ylabel('Price [USD]')
plt.title('Gold Prices')
plt.grid()
plt.show()


#  AutoCorrelation :
# Autocorrelation is the correlation of a point in the series with a point with lag taken as one day of the same series.
# If a series show a positive autocorrelation then we series is momentum or trend following and if a series show
# negative correlation then we say series is mean reversing.
# In python we can use pandas autocorr() function to calculate autocorrelation of a series.

# In[14]:


print(gold_data['gold_price'].autocorr())


#  Here the autocorrelation is positive so we can conclude that the series here is a trend following series.

# Autocorrelation Function :
# The sample autocorrelation function function show entire correlation function for different lags. So the autocorrelation function is a funtion of lag. By using statsmodels python library we visualise aucorrelation for different lags. Autocorelation fuction help us to choose how many lags we can use for forcasting.

# In[16]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(gold_data['gold_price'],lags=20,alpha=0.5)
plt.show()


# In[ ]:





# We can see that we can use lag of 20 days to forcast gold values because they have strongly positive autocorrelated.
# 
# White Noise :
# A noise time series simply a sequence of uncorrelated random variable that was identically distributed and follows a normal distribution with mean as zero.. Stock market are often modeled as white noise.
# 
# We can't forecast future observations for white noise based on past values because autocorrelations at all lags are 0.

# Random Walk :
# A series is said to be a random walk if a point of a series does not depend to previous point or entry. Every next point is random and we can not correlate them. Example : Today's price = Yesterday's price + Noise
#         Pt    =     Pt-1   +  et
# 
# Change in price in is white noise :
#         Pt    -   Pt-1   =   et
# 
# We can not forcast a random walk
# Best forcast for a Random walk is today price or current point

# Augmented Dickey Fuller test :
# This test is usefull to check weather a series is random walk or not. The null hypothesis is that the series follow a random walk. Therefore, a low p-value (say less then 5%) means we can reject null hyposthesis that time series is a random walk.
# Test :    Pt   -   Pt-1   =  α +   β Pt-1   +  et
# The eqution is only for one lag which for simple Dickey Fuller Test. We can add more lags to make the test augmented dickey fuller test.

# In[17]:


from statsmodels.tsa.stattools import adfuller
results = adfuller(gold_data['gold_price'])
print('The p-value of the test on prices is: ' + str(results[1]))


# The p-value we get is less then 5% so we reject this null hypothesis. Percent change in gold prices not follow random walk
# 
# Stationarity :
# Strong Stationary : A series is strong sationary if the entire distribution is time invariant. Joint distribtuion of the observations does not depend on time.
# Weak Stationary : A stationary process has the property that the mean, variance and autocorrelation structure do not change over time. For autocorrelation, corr( Xt, Xt -  τ
#  )is only function of  τ
#   and  τ
#   is a notation for lag.
# If a process non stationary then it become difficult to model.
# 
# If parameters vary with time then it results in too many parameters that needs to be estimate.
# We can only estimate model with a few parameters.
# Random walk is a common type of non-stationary.
# Seasonal data is also non-stationary beacause mean varies with time.
# 
# Therefore few technique to make a series stationary.
# 
# We can take difference (df.diff() in python) then new series is a stationary series.
# If we take only log of the series then it remove only growth of the time series and if we take first log and then difference the it become more stationary.

# In[18]:


#here we take diff() to make gold data stationary.
diff_gold_data = gold_data.diff().dropna()
diff_gold_data.plot()


# AR (Auto Regression Model) :
# In a auto regression model today's value is equal to mean plus fraction of yesterday's value plus noise.
# 
# Mathmetical description of AR(1) model :Rt =μ+ ϕRt-1 + ϵt
# AR parameter  ϕ:
# if  ϕ
#   = 0 then Rt is simply a white noise.
# if  ϕ
#   = 1 then Rt is a random walk.
# for stationary process -1 <  ϕ
#   < 1
# if  ϕ
#   < 0 then Mean reversion and if  ϕ
#   = 1 then Momentum which we discuss previously.
# 
# 
# Higher order AR models :
#      AR(1) :       Rt =  μ+ ϕ1Rt-1 + ϵ t
#      AR(2) :       Rt =  μ + ϕ1Rt-1 + ϕ2Rt-2 + ϵ t
#      AR(3) :       Rt =  μ+ ϕ1Rt-1 + ϕ2Rt-2 + ϕ3Rt-3 + ϵt
# ...
# 
# Estimating a AR Model :

# In[29]:


import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(gold_data,order=(1,0,0)) #here order attribute is used to select AR model
result = model.fit()
print(result.summary())


# Choosing a Right Model :
# Identifying the right order AR mdoel gives us best AR model to forcast the values.
# 
# The order of an AR model ususally be unkown.
# Two techniques to determine order :
#   - Partial autocorrelation function (PACF)
#   - Information Criteria

# Partial Autocorrelation Function :
# Partial autocorelation function measures the incremental benefit of adding lag. 
# Basically it provides us information if we add one more lag to our model then it would benefit the model or not.

# In[30]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(gold_data['gold_price'],lags=20,alpha=0.10)
plt.show()


# In this graph we can see that after lag 1 partical autocorrelation is approx zero. So we can use only one lag in the model.
# 
# Information Criteria :
# Generally if we have higher order AR model then it can be better fit for our dataset. But may result model overfit our dataset.
# So information is used to by adjusting goodness of fit measures by imposing penalty based on number of parameters that are used.Two popular adjusted goodness of fit measures.
# 
# AIC (Akaike Information Criteria)
# BIC (Baysian Information Criteria)
# We choose the model which have lowest aic or bic values.

# In[ ]:




