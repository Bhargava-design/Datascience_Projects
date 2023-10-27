#!/usr/bin/env python
# coding: utf-8

# In[1]:


#install the Tiingo package if you haen't installed it
#pip install tiingo and tensor flow
#Also install the necessary libraries
get_ipython().system('pip install tensorflow')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tiingo import TiingoClient


# In[4]:


#set your Tiingo API Key
api_key ="fcd1dda20ddf992ad55de36f2b3fe86ce71539ea"


# In[5]:


#now intialize the Tiingo Client
config = {
    'session' : True,
    'api_key' : api_key
}
client = TiingoClient(config)


# In[8]:


#Fetch TSLA stock Data
ticker = 'TSLA'
st_date ='2022-06-01'
ed_date = '2023-06-30'
df = client.get_dataframe(tickers = ticker, startDate=st_date, endDate=ed_date)


# In[10]:


#save the data to CSV file
df.to_csv(f'{ticker}.CSV')


# In[11]:


#Display some rows in CSV file
print(df.head(10))


# In[12]:


#now extract the adjclose' column and store it in df1
df1 = df['adjClose']


# In[13]:


#Now plot the data
plt.plot(df1)
plt.show()


# In[16]:


#reshape and scale the data using MinMaxScaler
s = MinMaxScaler(feature_range=(0, 1))
df1 = s.fit_transform(np.array(df1).reshape(-1, 1))


# In[17]:


#Display the df1 after scaler happens
print(df1)


# In[18]:


#Now split the datainto training and testing sets
training_size = int(len(df1)*0.65)
train_data, test_data = df1[0: training_size, :], df1[training_size:len(df1), :]


# In[19]:


#Display the sizes of training and testing data
print("Training data Size:",len(train_data))
print("Testing data Size:",len(test_data))


# In[21]:


#display the content oof testing and training data
print("Training data:",train_data)
print("Testing data:",test_data)


# In[23]:


#Now create a dataset for training and testing datas
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[24]:


print("X_train shape:",X_train.shape)


# In[25]:


print("y_train shape:",y_train.shape)


# In[26]:


print("X_test shape:",X_test.shape)


# In[27]:


print("y_test shape:",y_test.shape)


# In[49]:


# Reshape the input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
print(X_train)


# In[50]:


# Reshape the input to be [samples, time steps, features] which is required for LSTM
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
print(y_train)


# In[51]:


from tensorflow.keras.models import Sequential


# In[52]:


from tensorflow.keras.layers import LSTM

#Now create a Stacked LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[54]:


#Make Predicitons
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[55]:


print(data)


# In[56]:


#Now Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)


# In[60]:


print(data)


# In[61]:


#Make Predicitons
train_predict = model.predict(X_train)
test_predict = model.predict(y_train)


# In[62]:


print(train_predict)


# In[64]:


# Before that we have inverse the transform predictions
train_predict = s.inverse_transform(train_predict)
test_predict = s.inverse_transform(test_predict)


# In[100]:


x_input = test_data[96:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = [x for x in range(1, 301)]
n_steps = 100
i = 0
while i < 10:
    if len(temp_input) > 100:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input = temp_input[1:]
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
    else:
        break
print("Predict values for the Next 10 Days:", lst_output)


# In[101]:


len(test_data)


# In[102]:


x_input.shape


# In[108]:


day_new = np.arange(1, 100)
day_pred = np.arange(101, 131)


# In[109]:


len(df1)


# In[111]:


#Define the indices of slicing df1
start_index = 1158
end_index = start_index + len(lst_output)

#Check slicing indices are valid or not
if start_index < len(df1) and end_index >= len(df1):
    plt.plot(day_new, s.inverse_transform(df1[start_index:end_index]))
    plt.plot(day_pred, s.inverse_transform(lst_output))
    plt.show()
else:
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label = "Sine Wave", color='red')
    plt.title("Sample Sine Wave")
    plt.xlabel("X-Axis")
    plt.ylabel("Y_Axis")
    plt.legend()
    plt.grid(True)
    
plt.show()


# In[112]:


print(df1)


# In[116]:


df3 = df1.tolist()
plt.plot(df3[12:])#This is the final Output


# In[ ]:




