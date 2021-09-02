#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import DataFrame, read_csv, concat
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, GRU,ConvLSTM2D, Flatten
from matplotlib import pyplot as plt
from numpy import concatenate, reshape, array
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sys import argv
import csv
import datetime
import time
import pickle
from keras.models import model_from_json


# In[21]:


fileName = 'BCH' #asset name
modelNumber = 4 #insert model number with best results
json_file = open('models/{}/{}_{}_model.json'.format(fileName,fileName, modelNumber), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('models/{}/{}_{}_model.h5'.format(fileName,fileName, modelNumber))
print("Loaded model from disk")


# In[22]:


# Series to Supervised Learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
    # print("I: ",i)
        cols.append(df.shift(i))
        # print("Column: ",cols)
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # print("Names: ",names)
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        # print("COls: ",cols)
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # print("Names: ",names)

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg


# In[23]:


# Read Data and Extract Values
# Read Data and Extract Values



 #write crypto name
dataset = read_csv('final_datasets/'+fileName+'.csv', parse_dates=['time']) 


startIndex = 3  #start from 3rd column
nrows = dataset.shape[0]
values = dataset.iloc[:,startIndex:].values #Getting values - Total Sentiment and BTC Values
valuesCrypto = dataset.iloc[:,-1:].values #Getting values -  C Values
# For predicting with just Cryptocurrency values, we have just 1 input variable. 
# Incorporating sentiment values will make input variables=2

# Comment the below line if there are multiple features / input variable.
# values = values.reshape(-1,1) #Only do this if you have 1 input variable


# In[24]:


startDate_index =dataset.loc[dataset['time'] == '2020-12-01'].index[0] # use the date from which the testing data should start
total_index= dataset.iloc[[-1]].index[0]
train_percent=startDate_index/total_index


# In[25]:


# Scaling
scaler = MinMaxScaler(feature_range = (0,1))
scaler = scaler.fit(values)
scaled = scaler.fit_transform(values)


# In[28]:


# Input and Output Sequence Length
input_sequence = 1
output_sequence = 1


# Call Series to Supervised Function
reframed = series_to_supervised(scaled, input_sequence, output_sequence)


# In[29]:


# Drop current sentiment/any other feature that might be added in the future(at time t)

#The actual asset value should be the last column of the dataset for this to work
dropColumns = []
for i in range(values.shape[1]-1):
    dropColumns.append('var{}(t)'.format(i+1))
reframed=reframed.drop(columns=dropColumns) 


# In[30]:


# Ignore the headers
reframedValues = reframed.values


# In[34]:


#Splitting data into train and test sets

n_train_days = int(train_percent*nrows)
train = reframedValues[:n_train_days, :]
test = reframedValues[n_train_days:nrows, :]
# valuesCrypto = reframed.iloc[:,-1:].values #Getting values -  C Values

#Assigning inputs and output datasets
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

#Reshaping input to be 3 dimensions (samples, timesteps, features)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# In[35]:


# Predicition using the loaded model

model = loaded_model
model_prediction = model.predict(test_X)


# In[36]:


# Inverse Scale
scalerCrypto = MinMaxScaler(feature_range = (0,1))
scalerCrypto = scaler.fit(valuesCrypto)
scaledCrypto = scaler.fit_transform(valuesCrypto)


model_prediction_unscale = scalerCrypto.inverse_transform(model_prediction)
predictedValues = reshape(model_prediction_unscale, model_prediction_unscale.shape[0])

actualValues = valuesCrypto[n_train_days+input_sequence:] #test_y+input_sequence:


actualValues = reshape(actualValues, actualValues.shape[0])


# In[37]:


actual= DataFrame(actualValues, columns= ['Actual Value'])
predicted=DataFrame(predictedValues, columns= ['Predicted Value'])


# In[38]:


#Calculating RMSE and MAE
errorDF=concat([actual,predicted], axis=1)
errorDF.dropna(inplace=True)
rmse = sqrt(mean_squared_error(errorDF.iloc[:,0], errorDF.iloc[:,1]))
mae = mean_absolute_error(errorDF.iloc[:,0], errorDF.iloc[:,1])
print('Test MAE: %.3f' % mae)
print('Test RMSE: %.3f' % rmse)


# In[39]:


# Write to csv
writeFileName = "--Results.csv"
timestamp = DataFrame(dataset['time'][n_train_days:], columns= ['time'])
timestamp.reset_index(drop=True, inplace=True)
results=concat([timestamp,actual,predicted], axis=1)
print("Head: ",results.head())
print("Tail: ",results.tail())
results.dropna(inplace=True)
results.to_csv('Prediction Tables/'+fileName+writeFileName, index= False)


# In[ ]:




