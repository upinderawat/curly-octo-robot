import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def predict(X_train, y_train, X_test):
  model = LinearRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  return int(round(y_pred[0]))



def train(path):
  data = pd.read_csv(path)
  # data

  state_uts = []
  for index,row in data.iterrows():
    state_ut = row['State/UnionTerritory']
    state_ut = re.sub('[^A-Za-z0-9]+', '', state_ut)
    state_uts.append(state_ut)

  data['State/UnionTerritory'] = state_uts
  # data

  unique_dates = data['Date'].unique().tolist()
  unique_states = set(data['State/UnionTerritory'].tolist())
  # print(len(unique_dates))
  # print(len(unique_states))
  # unique_states

  df1 = pd.DataFrame(columns = ['Date', 'Confirmed'])
  df1['Date'] = unique_dates
  df1['Confirmed'] = 0
  # df1

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


  delete = []
  for key, value in final.items():
    confirmed_list = value['Confirmed'].tolist()
    length = len(confirmed_list)
    # print(length)
    # print(key, confirmed_list[length-1])
    if(confirmed_list[length-1]==0):
      delete.append(key)

  for i in delete:
    del final[i]

  # final

  state_confirmed_dict = {}
  for key, values in final.items():
    confirmed_list = values['Confirmed'].tolist()
    state_confirmed_dict [key] = confirmed_list

  print(state_confirmed_dict)
  # print(differences)
  # len(differences)

  state_prediction = {}
  for key, value in state_confirmed_dict.items():
    X_state = np.array(value)
    X_state = np.trim_zeros(X_state)
    # print(X_state.shape)
    window_size = 5
    Xt = np.array([0] * (window_size + 1))

    row = 0
    while(row < (X_state.shape[0] - window_size)):
      # row = random.randrange(len(X) - window_size - 1)
      temp = X_state[row:row+window_size+1].tolist()
      temp = [float(item) for item in temp]
      Xt = np.vstack((Xt,temp)) 
      row = row + 1

    Xt = Xt[1:,:]
    # Xt = np.trim_zeros(Xt)
    X_train = Xt[:,0:window_size]
    y_train = Xt[:,window_size]

    # X_train = preprocessing.scale(X_train)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_train)
    X_test = X_train[X_train.shape[0]-1 , :]
    X_test = X_test.reshape((1,-1))
    # print(X_test))
    prediction = predict(X_train, y_train, X_test)
    state_prediction[key] = prediction
    # state_prediction[key] = int(round(prediction))

  print(state_prediction)
