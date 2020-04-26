class covid_predictions:

  def __init__(self, window_size = 5):
    self.model = LinearRegression()
    self.window_size = window_size

  def set_data(self, data):
    self.data = data

  def fit_model(self, X_train, y_train):
    self.model.fit(X_train, y_train)

  def make_model_prediction(self, X_test):
    y_pred = self.model.predict(X_test)
    return int(round(y_pred[0]))

  def remove_special_characters(self, data):
    state_uts = []
    for index,row in data.iterrows():
      state_ut = row['State/UnionTerritory']
      state_ut = re.sub('[^A-Za-z0-9]+', '', state_ut)
      state_uts.append(state_ut)
    return state_uts

  def create_dates_dataframe(self, unique_dates):
    df1 = pd.DataFrame(columns = ['Date', 'Confirmed'])
    df1['Date'] = unique_dates
    df1['Confirmed'] = 0
    return df1

  def get_statewise_confirmed_cases(self, unique_dates, unique_states):
    df1 = create_dates_dataframe(unique_dates)

    final = {}
    for state in unique_states:
      dates = []
      confirmeds = []
      df2 = pd.DataFrame()
      for index, row in df1.iterrows():
        datevalue = row['Date']
        dates.append(datevalue)
        a = self.data[(self.data['State/UnionTerritory']==state) & (self.data['Date']==datevalue)]
        if(a.empty):
          confirmeds.append(0)
        else:
          conf = (a['Confirmed'].values[0])
          confirmeds.append(conf)
      df2 = pd.DataFrame()
      df2['Date'] = dates
      df2['Confirmed'] = confirmeds
      final[state] = df2

    return final

  def remove_states_with_no_cases(self, state_cases):
    delete = []
    for key, value in state_cases.items():
      confirmed_list = value['Confirmed'].tolist()
      length = len(confirmed_list)
      if(confirmed_list[length-1] == 0):
        delete.append(key)

    for i in delete:
      del state_cases[i]

    return state_cases

  def get_statewise_cases(self, state_cases):
    state_confirmed_dict = {}
    for key, values in state_cases.items():
      confirmed_list = values['Confirmed'].tolist()
      state_confirmed_dict[key] = confirmed_list

    return state_confirmed_dict

  def fit_train(self, X_state):
    X_state = np.trim_zeros(X_state)
    Xt = np.array([0] * (self.window_size + 1))
    row = 0
    while(row < (X_state.shape[0] - self.window_size)):
      temp = X_state[row:row+self.window_size+1].tolist()
      temp = [float(item) for item in temp]
      Xt = np.vstack((Xt,temp)) 
      row = row + 1

    Xt = Xt[1:,:]
    X_train = Xt[:,0:self.window_size]
    y_train = Xt[:,self.window_size]

    return X_train, y_train

  def fit_test(self, X_train):
    X_test = X_train[X_train.shape[0]-1 , :]
    X_test = X_test.reshape((1,-1))
    return X_test      

  def predict(self, state_confirmed_dict):
    
    state_prediction = {}

    for key, value in state_confirmed_dict.items():
      X_state = np.array(value)
      X_train, y_train = self.fit_train(np.array(value))
      X_test = self.fit_test(X_train)
      self.fit_model(X_train, y_train)
      state_prediction[key] = self.make_model_prediction(X_test)

    return state_prediction

  def train(self, path):

    data = pd.read_csv(path)
    self.set_data(data)

    data['State/UnionTerritory'] = self.remove_special_characters(data)

    unique_dates = data['Date'].unique().tolist()
    unique_states = set(data['State/UnionTerritory'].tolist())

    state_cases = self.get_statewise_confirmed_cases(unique_dates, unique_states)
    
    state_cases = self.remove_states_with_no_cases(state_cases)

    state_confirmed_dict = self.get_statewise_cases(state_cases)

    state_prediction = self.predict(state_confirmed_dict)

    print(state_prediction)

