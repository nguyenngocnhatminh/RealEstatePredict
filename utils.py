import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.tabular.all import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#importing required libraries
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from matplotlib.pylab import rcParams

def ReadData():
    rcParams['figure.figsize'] = 20, 10
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = pd.read_csv("Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_mon.csv")
    return df

def GetAllRegion():
    df = ReadData()
    region = df[["RegionID","RegionName"]]
    data = []
    for i in range(0,len(region)):
        data.append({
            "RegionID": int(region["RegionID"][i]),
            "RegionName": region["RegionName"][i]
        })
    print(data)
    return data

def GetDataByRegion(regionID):
    df = ReadData()
    predata = df[df["RegionID"] == regionID]
    time = df.columns.tolist()[5:]
    value = predata.values.tolist()[0]
    value = value[5:]
    data = pd.DataFrame(index=range(0,len(time)),columns=['Time', 'Value'])
    for i in range(0,len(time)):
        data["Time"][i] = time[i]
        data["Value"][i] = value[i]
    return data

def change_to_json(train, valid, preds):
    data = []
    for i in range(0, len(train)):
        data.append({
            "Time": train['Time'][i],
            "Value": train['Value'][i],
            "Type": 1
        })
    for i in range(0, len(valid)):
        data.append({
            "Time": valid['Time'][i+len(train)],
            "Value": valid['Value'][i+len(train)],
            "Type": 2
        })
    for i in range(0, len(valid)):
        data.append(
            {
                "Time": valid['Time'][i+len(train)],
                "Value": preds[i],
                "Type": 3
            }
        )
    return data

def change_to_json_from_model(train, valid, preds):
    data = []
    for i in range(0, len(train)):
        data.append({
            "Time": train.index[i],
            "Value": train['Value'][i],
            "Type": 1
        })
    for i in range(0, len(valid)):
        data.append({
            "Time": valid.index[i],
            "Value": valid['Value'][i],
            "Type": 2
        })
    for i in range(0, len(valid)):
        data.append(
            {
                "Time": valid.index[i],
                "Value": preds.item(i),
                "Type": 3
            }
        )
    return data

def Moving_Avage(regionID):
    data = GetDataByRegion(regionID)
    train = data[:240]
    valid = data[240:]
    preds = []
    for i in range(0, len(valid)):
        a = train['Value'][len(train) - 60 + i:].sum() + sum(preds)
        b = a / 60
        preds.append(b)
    # checking the results (RMSE value)
    mape = np.mean(np.abs((valid['Value'] - preds) / valid['Value'])) * 100
    print("Moving_Average")
    print("mape")
    print(mape)
    data = change_to_json(train,valid,preds)
    print(data)
    return data

def Prophet(region_id):
    #import require libraries
    from fbprophet import Prophet
    #Lấy dữ liệu được clean vào
    data = GetDataByRegion(region_id)
    #Fomat data
    time = data['Time'].tolist()
    data['Time'] = pd.to_datetime(data.Time, format='%Y/%m/%d')
    data.index = data['Time']
    #Chuẩn bị dữ liệu
    data.rename(columns={'Value': 'y', 'Time': 'ds'}, inplace=True)
    #Chia dữ liệu
    train = data[:240]
    valid = data[240:]
    #Sử dụng thư viện Prophet để tạo model
    model = Prophet(daily_seasonality=True)
    model.fit(train)
    # Dự đoán
    close_prices = model.make_future_dataframe(periods=len(valid))
    forecast = model.predict(close_prices)
    forecast_valid = forecast['yhat'][240:]
    mape = np.mean(np.abs((valid['y'].values - forecast_valid.values) / valid['y'].values))*100
    print("Prophet")
    print("mape")
    print(mape)
    data.rename(columns={'y': 'Value', 'ds': 'Time'}, inplace=True)
    data['Time'] = time
    data.index = data['Time']
    train = data[:240]
    valid = data[240:]
    data = change_to_json_from_model(train,valid,forecast_valid.values)
    print(data)
    return data


def lstmPrediction(regionID):
    data = GetDataByRegion(regionID)
    data.index = data.Time
    data.drop('Time', axis=1, inplace=True)

    # creating train and test sets
    dataset = data.values

    train = dataset[0:240, :]
    valid = dataset[240:, :]

    # converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=40, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=40))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # predicting 246 values, using past 60 from the train data

    inputs = data[len(data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
    # for plotting
    train = data[:240]
    valid = data[240:]
    print(valid)
    data = []
    print(valid)
    for i in range(0, len(train)):
        data.append({
            "Time": train.index[i],
            "Value": train['Value'][i],
            "Type": 1
        })
    for i in range(0, len(valid)):
        data.append({
            "Time": valid.index[i],
            "Value": valid['Value'][i],
            "Type": 2
        })
    for i in range(0, len(valid)):
        data.append(
            {
                "Time": valid.index[i],
                "Value": closing_price.item(i),
                "Type": 3
            }
        )
    return data

