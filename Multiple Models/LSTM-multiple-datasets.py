#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[9]:



from tqdm import tqdm

fileNames=['BCH','BTC', 'ETC', 'ETH','EOS','LINK', 'LTC','DASH', 'MKR','OMG','XLM','XTZ','ZRX']

for fileName in fileNames:

    for at in tqdm(range(50)):

        dataset = read_csv('final_datasets/'+fileName+'.csv', parse_dates=['time']) 


        startIndex = 3  #start from 3rd column
        nrows = dataset.shape[0]
        values = dataset.iloc[:,startIndex:].values #Getting values - Total Sentiment and BTC Values
        valuesCrypto = dataset.iloc[:,-1:].values #Getting values -  C Values
        # For predicting with just Cryptocurrency values, we have just 1 input variable. 
        # Incorporating sentiment values will make input variables=2

        # Comment the below line if there are multiple features / input variable.
        # values = values.reshape(-1,1) #Only do this if you have 1 input variable
        num =dataset.loc[dataset['time'] == '2020-12-01'].index[0]

        num2= dataset.iloc[[-1]].index[0]

        percent=num/num2

        scaler = MinMaxScaler(feature_range = (0,1))
        scaler = scaler.fit(values)
        scaled = scaler.fit_transform(values)


        # Input and Output Sequence Length
        input_sequence = 1
        output_sequence = 1


        # Call Series to Supervised Function
        reframed = series_to_supervised(scaled, input_sequence, output_sequence)


        # Drop current sentiment/any other feature that might be added in the future(at time t)
        dropColumns = []
        for i in range(values.shape[1]-1):
            dropColumns.append('var{}(t)'.format(i+1))
        reframed=reframed.drop(columns=dropColumns)

        # Drop cuurent sentiment
        #reframed=reframed.drop(columns=['var2(t-1)'])
        # Ignore the headers
        reframedValues = reframed.values


        #Splitting data into train and test sets

        n_train_days = int(percent*nrows) #90% data is train, 10% test
        train = reframedValues[:n_train_days, :]
        test = reframedValues[n_train_days:nrows, :]
        # valuesCrypto = reframed.iloc[:,-1:].values #Getting values -  C Values

        #Assigning inputs and output datasets
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        #Reshaping input to be 3 dimensions (samples, timesteps, features)
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        #Building LSTM Neural Network model


        model = Sequential()
        model.add(Bidirectional(GRU(50, activation='relu', return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]))))
        model.add(LSTM(50,activation ='tanh')) 
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse',metrics=['acc'])


        # Uncomment below line to get summary of the model
        # print(model.summary(line_length=None, positions=None, print_fn=None))


        #Fitting model
        history = model.fit(train_X, train_y, epochs = 100, batch_size=64, validation_data=(test_X, test_y), verbose=0, shuffle=False) #Best so far: 100 neurons, epochs = 400, batch_size = 53

        #saving model 
        model_json = model.to_json()
        with open('models/'+fileName+'/'+fileName+"_"+str(at)+"_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('models/'+fileName+'/'+fileName+"_"+str(at)+"_model.h5")
        print("Saved " + fileName+"_"+str(at)+"_model.h5 to disk")
        
        
        # Predicition
        model_prediction = model.predict(test_X)

        # Inverse Scale
        scalerCrypto = MinMaxScaler(feature_range = (0,1))
        scalerCrypto = scaler.fit(valuesCrypto)
        scaledCrypto = scaler.fit_transform(valuesCrypto)


        model_prediction_unscale = scalerCrypto.inverse_transform(model_prediction)
        predictedValues = reshape(model_prediction_unscale, model_prediction_unscale.shape[0])

        actualValues = valuesCrypto[n_train_days+input_sequence:] #test_y+input_sequence:


        actualValues = reshape(actualValues, actualValues.shape[0])

        #Plotting training loss vs validation loss
        # plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='validation')
        # plt.legend()
        # plt.show()

        #Visualising Results (Actual vs Predicted)
        # plt.plot(actualValues, color = 'red', label = 'Actual '+ fileName + ' Value')
        # plt.plot(predictedValues, color = 'blue', label = 'Predicted '+ fileName + ' Value') #[1:38]
        # plt.title(fileName+' Trend Prediction')
        # plt.xlabel('Time Interval (1 interval = 3 hours)')
        # plt.ylabel('Price')
        # plt.legend()

        # Uncomment below line to save the figure
        # plt.savefig('Trend_Graphs/'+'Trend Graph for '+fileName+'.png', dpi=700)

        # plt.show()

        actual= DataFrame(actualValues, columns= ['Actual Value'])
        predicted=DataFrame(predictedValues, columns= ['Predicted Value'])

        # Write to csv
        writeFileName = "--Results.csv"
        timestamp = DataFrame(dataset['time'][n_train_days:], columns= ['time'])
        timestamp.reset_index(drop=True, inplace=True)
        results=concat([timestamp,actual,predicted], axis=1)
        # print("Head: ",results.head())
        # print("Tail: ",results.tail())
        results.dropna(inplace=True)

        results.to_csv('Prediction Tables/'+fileName+'/'+fileName+'_'+str(at)+writeFileName, index= False)


# In[ ]:




