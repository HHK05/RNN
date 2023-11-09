#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# In[4]:


dataset_training =pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset_training.iloc[:,1:2].values


# In[10]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)


# In[12]:


print(training_set_scaled)


# In[13]:


print(training_set)


# In[42]:


X_train=[]
Y_train=[]

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])
X_train,Y_train=np.array(X_train),np.array(Y_train)


# In[16]:


X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


# In[17]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[18]:


regressor=Sequential()


# In[21]:


regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))


# In[22]:


regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))


# In[23]:


regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))


# In[24]:


regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))


# In[26]:


regressor.add(Dense(units=1))


# In[31]:


regressor.compile(optimizer='adam',loss='mean_squared_error')


# In[32]:


regressor.fit(X_train,Y_train,epochs=100,batch_size=32)


# In[36]:


dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values


# In[44]:


dataset_total=pd.concat((dataset_training['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)


# In[47]:


X_test=[]

for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


# In[49]:


plt.plot(real_stock_price,color='red',label='Real_Google_Stock_Price')
plt.plot(predicted_stock_price,color='Blue',label='Pridicted_Stock_Price')
plt.title('Stock_Market_visulization')
plt.xlabel('Time')
plt.ylabel('Google_Stock_Price')
plt.legend()
plt.show()


# In[57]:


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))


# In[58]:


print(rmse)


# In[ ]:




