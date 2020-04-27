#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[45]:


pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[46]:


data = pd.read_csv('./Data Collected/covid_19_india.csv')
data


# In[47]:


data.shape


# In[48]:


import datetime as dt
dates = data['Date']
x = [dt.datetime.strptime(d,'%d/%m/%y').date() for d in dates]


# In[49]:


cases_numbers = data['Confirmed']
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
figure(num=None, figsize=(30, 10), dpi=80, facecolor='w', edgecolor='k')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x,cases_numbers)
plt.gcf().autofmt_xdate()
plt.xlabel("dates")
plt.ylabel("Number of cases")
plt.show()


# In[50]:


import re
state_uts = []
for index,row in data.iterrows():
    state_ut = row['State/UnionTerritory']
    state_ut = re.sub('[^A-Za-z0-9]+', '', state_ut)
    state_uts.append(state_ut)


# In[51]:


data['State/UnionTerritory'] = state_uts
# data


# In[52]:


unique_dates = data['Date'].unique().tolist()
unique_states = set(data['State/UnionTerritory'].tolist())
# unique_states


# In[53]:


df1 = pd.DataFrame(columns = ['Date', 'Confirmed'])
df1['Date'] = unique_dates
df1['Confirmed'] = 0
# df1


# In[54]:


final = {}
for state in unique_states:
    dates = []
    confirmeds = []
    df2 = pd.DataFrame()
    for index, row in df1.iterrows():
        datevalue = row['Date']
        dates.append(datevalue)
        a = data[(data['State/UnionTerritory']==state) & (data['Date']==datevalue)]
        if(a.empty):
          confirmeds.append(0)
        else:
          conf = (a['Confirmed'].values[0])
          confirmeds.append(conf)
    df2 = pd.DataFrame()
    df2['Date'] = dates
    df2['Confirmed'] = confirmeds
    final[state] = df2


# In[55]:


final


# In[56]:


input_region_name = input("Enter the name of region: ")


# In[57]:


data_for_input_region = final[input_region_name]


# In[58]:


data_for_input_region_ori = data_for_input_region.copy()
data_for_input_region


# In[59]:


cases_numbers = data_for_input_region['Confirmed']
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
import datetime as dt
dates = data_for_input_region['Date']
x = [dt.datetime.strptime(d,'%d/%m/%y').date() for d in dates]
figure(num=None, figsize=(30, 15), dpi=80, facecolor='w', edgecolor='k')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x,cases_numbers)
plt.gcf().autofmt_xdate()
plt.xlabel("dates")
plt.ylabel("Number of cases")
plt.show()


# In[60]:


dates = data_for_input_region['Date']
date_features = [dt.datetime.strptime(d,'%d/%m/%y').date() for d in dates]

# Get the list with seconds since earliest event
date_features = [(i - min(date_features)).total_seconds() for i in date_features]
# Normalize data so it lies between 0 and 1
date_features = [i/max(date_features) for i in date_features]
# print(date_features)
data_for_input_region['Date'] = date_features
data_for_input_region


# In[61]:


data_for_input_region_ori


# In[62]:


from sklearn.model_selection import train_test_split
X = data_for_input_region
Y = data_for_input_region['Confirmed']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=42)


# In[63]:


print("shape of training data: ",X_train.shape)
print("shape of testing data: ",X_test.shape)
print("shape of training data label: ",y_train.shape)
print("shape of testing data label: ",y_test.shape)


# # Multilayer Perceptron Model

# In[130]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-12, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train,y_train)


# In[131]:


y_pred = clf.predict(X_test)


# In[132]:


y_pred


# In[134]:


from sklearn.metrics import r2_score
print(r2_score(y_test,  y_pred))


# # Long Short Term Memory (LSTM)

# In[69]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
import numpy
import datetime
from numpy import concatenate


# In[70]:


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    return df


# In[71]:


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)

    return pd.Series(diff)


# In[72]:


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# In[73]:


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# In[74]:


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# In[75]:


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons, timesteps):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], timesteps, 1)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
        
    return model


# In[76]:


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, len(X), 1)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


# In[77]:


# run a repeated experiment
def experiment(repeats, series, timesteps):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, timesteps)
    supervised_values = supervised.values[timesteps:,:]
    # split data into train and test-sets
    train, test = supervised_values[0:-12, :], supervised_values[-12:, :]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the base model
        lstm_model = fit_lstm(train_scaled, 1, 500, 1, timesteps)
        # forecast test dataset
        predictions = list()
        for i in range(len(test_scaled)):
            # predict
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
    return error_scores, predictions


# In[78]:


# execute the experiment
# load dataset
series = data_for_input_region_ori.copy()
series = series.set_index('Date')
# experiment
repeats = 10
results = pd.DataFrame()
# run experiment
timesteps = 1
results['results'],predictions = experiment(repeats, series, timesteps)
# summarize results
print(results.describe())


# In[79]:


results.describe()


# In[80]:


results


# In[81]:


predictions = np.array(predictions)


# In[82]:


predictions


# In[83]:


import math
predictions = list(map(math.floor, predictions)) 


# In[84]:


predictions = np.array(predictions)
predictions


# In[85]:


test_portion = Y[-12:]
test_portion


# In[86]:


predictions=predictions.reshape(-1)


# In[87]:


from sklearn.metrics import mean_squared_error
sqrt(mean_squared_error(test_portion, predictions))


# In[88]:


print(r2_score(y_test, predicted))


# # Multinomial Naive Bayes

# In[90]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)


# In[91]:


predicted = model.predict(X_test)


# In[93]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, predicted)
print("Accuracy: ",acc*100)


# In[94]:


print(r2_score(y_test, predicted))


# # Gaussian process regression

# In[191]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings; warnings.simplefilter('ignore')

# Instantiate a Gaussian Process model
kernel = C(1.0, (10.0, 100.0)) * RBF(30.0, (1000000000.0, 10000000000.0))+RBF(20.0, (1000000000.0, 10000000000.0))
# kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(X_test, return_std=True)

print("R2 score: ",r2_score(y_test, y_pred))

