
# coding: utf-8

# # Assignment on Time series modeling
# ## Problem Statement
# In this assignment students have to make ARIMA model over shampoo sales data and
# check the MSE between predicted and actual value.
# Student can download data in .csv format from the following link:
# ### Url link:- https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period#!ds=22r0&display=line
# 

# ## Load important libraries  into our project

# In[1]:


#importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# ### Step 1. Loading the csv file data in correct format in pandas series

# In[2]:


# loading csv file into series and formating the date
def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0,squeeze=True, date_parser=parser)
series.head()


# ### Step 2. Exploring the data and visualization regarding stationarity

# In[3]:


plt.figure(figsize=(15,5))
plt.plot(series,color='red', linewidth=2,label='Shampoo sales')
plt.legend(loc='upper left')
plt.show()


# ### Step 2.1 Plot rolling statistics and Dicky fuller test

# In[4]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(8).mean()
    rolstd = timeseries.rolling(8).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

   


# In[5]:


test_stationarity(series)


# #### we can observe that the rolling mean is increasing over time. Also in dicky fuller test test-statistics is greater than critical value. so we accept the null hypothesis(H0) i.e. the timeseries is not stationary.

# ### Step 3:- Eliminating Trend and Seasonality

# In[6]:


# Applying log function
series_log = np.log(series)
series_log.head()


# In[7]:


# visualizing the logged value
plt.figure(figsize=(15,5))
plt.plot(series_log,color='red', linewidth=2,label='Shampoo sales')
plt.legend(loc='upper left')
plt.show()


# In[8]:


# checking for stationarity on the logged value
test_stationarity(series_log)


# #### we see that we have constant rolling standard deviation but increasing trend in the rolling mean. Also Test statistics is greater than critical value so we accept null hypothesis. i.e. the time series is not stationary

# In[9]:


# Applying first order differencing on the log value
series_log_diff = series_log - series_log.shift(periods=1)
series_log_diff.head()


# In[10]:


# visualizing the logged value
plt.figure(figsize=(15,5))
plt.plot(series_log_diff[1:],color='red', linewidth=2,label='Shampoo sales')
plt.legend(loc='upper left')
plt.show()


# In[11]:


# checking for stationarity on the first order differential logged value
test_stationarity(series_log_diff[1:])


# #### we see that we have constant rolling mean but fluctuating trend in the rolling standard deviation. Also Test statistics is greater than critical value so we accept null hypothesis. i.e. the time series is not stationary

# In[12]:


# Applying second order differencing on the log value
series_log_diff_sec = series_log_diff - series_log_diff.shift(periods=1)
series_log_diff_sec.head()


# In[13]:


# visualizing the logged value
plt.figure(figsize=(15,5))
plt.plot(series_log_diff_sec[1:],color='red', linewidth=2,label='Shampoo sales')
plt.legend(loc='upper left')
plt.show()


# In[14]:


# checking for stationarity on the first order differential logged value
test_stationarity(series_log_diff_sec[2:])


# #### we see that Test statistics is greater than critical value so we reject null hypothesis. i.e. we can say with 99 % confidence level that the time series is stationary

# ### Step 4:- Forecasting a Time Series

#  The data on which we will do the the forecast is second order differenced logged value

# In[15]:


series_log_diff_sec.values[2:]


# ### step 4.1  plot ACF and PACF graph

# In[16]:


#ACF and PACF plots:
import statsmodels.api as sm

# show plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(series_log_diff_sec.values.squeeze()[2:], lags=10, ax=ax1)

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(series_log_diff_sec.values.squeeze()[2:], lags=10, ax=ax2)


# ### Observation
# In ARIMA(p,d,q), p is calculated by looking from PACF plot and q is calulated from ACF plot.
# 
# from PACF plot we choose p=4
# 
# from ACF plot we choose q=1
# 
# so we will plot for 3 sets of p,d,q (4,2,0),(0,2,1) and (4,2,1)

# ### step 4.2 Train the model and predict the value

# In[17]:


import warnings
warnings.filterwarnings('ignore')


# In[18]:


# for model ARIMA(4,2,0) i.e. AR Model
model = ARIMA(series_log, order=(4, 2, 0))  
results_ARIMA = model.fit(disp=-1)  

plt.figure(figsize=(15,5))
plt.plot(series_log_diff_sec,color='blue', linewidth=1,label='Original series')
plt.plot(results_ARIMA.fittedvalues,color='red', linewidth=2,label='Predicted Series')
plt.legend(loc='upper left')
plt.title('MSE: %.4f'% mean_squared_error(series_log_diff_sec.values[2:],results_ARIMA.fittedvalues) )
plt.show()


# In[19]:


# for model ARIMA(0,2,1) i.e MA Model
model = ARIMA(series_log, order=(0, 2, 1))  
results_ARIMA = model.fit(disp=-1)  

plt.figure(figsize=(15,5))
plt.plot(series_log_diff_sec,color='blue', linewidth=1,label='Original series')
plt.plot(results_ARIMA.fittedvalues,color='red', linewidth=2,label='Predicted Series')
plt.legend(loc='upper left')
plt.title('MSE: %.4f'% mean_squared_error(series_log_diff_sec.values[2:],results_ARIMA.fittedvalues) )
plt.show()


# In[20]:


# for model ARIMA(4,2,1) i.e. ARIMA Model
model = ARIMA(series_log, order=(4, 2, 1))  
results_ARIMA = model.fit(disp=-1)  
#future_ARIMA =results_ARIMA.predict(start=pd.to_datetime('1904-01-01'), dynamic=False)
plt.figure(figsize=(15,5))
plt.plot(series_log_diff_sec,color='red', linewidth=2,label='Original series')
plt.plot(results_ARIMA.fittedvalues,color='blue', linewidth=2,label='Predicted Series')
#plt.plot(future_ARIMA,color='green', linewidth=2,label='Forecasted Series')
plt.legend(loc='upper left')
plt.title('MSE: %.4f'% mean_squared_error(series_log_diff_sec.values[2:],results_ARIMA.fittedvalues) )
plt.show()


# ### Observation
# In ARIMA(p,d,q)
# 
# for ARIMA(4,2,0):-This is Just like AR model where MSE = 0.0945 
# 
# for ARIMA(0,2,1):-This is Just like MA model where MSE = 0.1536 
# 
# for ARIMA(4,2,1):-This is Just like ARIMA model where MSE = 0.0764
# 
# ### Hence, ARIMA(4,2,1) is best model which forecasts the time series very well.
