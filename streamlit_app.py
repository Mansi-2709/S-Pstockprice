import tensorflow as tf
!pip install -q keras
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

stock_data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/all_stocks_5yr.csv')

stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['Name'].unique()

# Get latest record per stock
latest = stock_data.sort_values('date').groupby('Name').tail(1)

# Get previous day for % change
prev = stock_data.sort_values('date').groupby('Name').nth(-2).reset_index()

# Merge
merged = latest.merge(prev[['Name', 'close']], on='Name', suffixes=('', '_prev'))

# Calculate % change
merged['change'] = ((merged['close'] - merged['close_prev']) / merged['close_prev']) * 100

fig = px.treemap(
    merged,
    path=['Name'],                  # Only stocks (no sector available)
    values='volume',                # Size of box
    color='change',                 # Color based on % change
    color_continuous_scale=[
        "#8B0000", "#111111", "#00C853"
    ],
    color_continuous_midpoint=0
)

fig.update_layout(
    paper_bgcolor="#0b1f24",
    plot_bgcolor="#0b1f24",
    font_color="white",
    margin=dict(t=40, l=10, r=10, b=10)
)

fig.show()

grouped = stock_data.groupby('Name')
data1 = grouped.get_group('BAC')
data1

data2=data1[['date', 'open']].copy()
data2=data2.set_index('date')
fig = go.Figure(data=[go.Candlestick(x=data1['date'],
                open=data1['open'],
                high=data1['high'],
                low=data1['low'],
                close=data1['close'])])

fig.show()
data1=data1['open']
data1

# LSTM is sensitive to the scale of data so we use MinMaxScaler to fit data in range of 0 to 1
scaler=MinMaxScaler(feature_range=(0,1))
data1=scaler.fit_transform(np.array(data1).reshape(-1,1))
training_size=int(len(data1)*0.65) # splitting data into train test split
test_size=len(data1)-training_size
train_data,test_data=data1[0:training_size,:],data1[training_size:len(data1),:1]

"""
This is an important step in training our model there are two components X(input features) and Y(output)
so here we use the 'open' column to make the entire data so the logic is that we use data of 100 days as input feature(X) and the next day's data as output(Y)
same is repeated again and again for all the data. Then we train our model with this data with X and Y that we generated.
here time_step is 100 but we can take this value as per our data
"""
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))
### Plotting
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(data1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(data1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# XGBoost Regressor ---------------------------------------------------

def create_features(df):
    """
    Create time series features based on time series index. we have converted date into index so now we take
    each date(each entry) and get features like quarter, week, month, year, day of week, day of year, day of month
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def add_lags(df):
    target_map = data2['open'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('35 days')).map(target_map)   # lag features are when we use past data as one of the features
    df['lag2'] = (df.index - pd.Timedelta('63 days')).map(target_map)   # eg. here we have 35 days lag so starting from first entry we go 35 days in past for every date(entry) and use data for that date as the feature
    df['lag3'] = (df.index - pd.Timedelta('91 days')).map(target_map)   # now for the very first data we don't have past data so we put NaN there
    df['lag4'] = (df.index - pd.Timedelta('126 days')).map(target_map)  # same goes for all lags 63 days, 91 days, 126 days ....
    df['lag5'] = (df.index - pd.Timedelta('154 days')).map(target_map)
    df['lag6'] = (df.index - pd.Timedelta('182 days')).map(target_map)
    return df

data2 = create_features(data2)
data2 = add_lags(data2)

tss = TimeSeriesSplit(n_splits=5, test_size= 180, gap=24)  #splitting data into test and train
data2 = data2.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(data2):
    train = data2.iloc[train_idx]
    test = data2.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)
    FEATURES = ['dayofyear', 'dayofmonth', 'weekofyear', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2','lag3', 'lag4','lag5','lag6']
    TARGET = 'open'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)
    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

test['prediction'] = reg.predict(X_test)
data2 = data2.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = data2[['open']].plot(figsize=(15, 5))
data2['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()


