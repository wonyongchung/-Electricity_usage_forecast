import math
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv
import time

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

for TYPE in ["InterpolatedData(Linear)"]:
    datalist = ["Academic.csv","Green.csv","Science1.csv","Science2.csv"]
    f = open("./Experimental Results.csv", 'a', newline='')
    w = csv.writer(f)
    w.writerow(['Building Cluster', 'Model', 'RMSE (kWh)', 'MAE (kWh)', 'MAPE (%)'])
    for csvfile in datalist:

        file_name = "./" + csvfile
        data = pd.read_csv(file_name, engine='python')

        if csvfile == "Science2.csv":
            train_data = data.iloc[672:116928, :]  # Academic, Green, Science1: 672:140256; Science2: 672:116928
            test_data = data.iloc[116928:151968, :]  # Academic, Green, Science1: 140256:175296; Science2: 116928:151968
        else:
            train_data = data.iloc[672:140256, :] # Academic, Green, Science1: 672:140256; Science2: 672:116928
            test_data = data.iloc[140256:175296:, :] # Academic, Green, Science1: 140256:175296; Science2: 116928:151968

        print("Building a Forecasting Model Using " + csvfile  )

        X_Data = train_data.iloc[:, :-1]
        scaler = StandardScaler().fit(X_Data)
        X_Data = scaler.transform(X_Data)
        Y_Data = train_data.iloc[:, -1]

        X_Test = test_data.iloc[:, :-1]
        X_Test = scaler.transform(X_Test)
        Y_Test = test_data.iloc[:, -1]

        model = RandomForestRegressor(n_estimators=128, n_jobs=-1, random_state=42, max_features='auto')
        result = []
        result1 = []
        model.fit(X_Data, Y_Data.values.ravel())

        Y_Pred = model.predict(X_Test)
        Y_True = Y_Test

        rmse = math.sqrt(mean_squared_error(Y_Pred, Y_True))
        mae = mean_absolute_error(Y_Pred, Y_True)
        mape = mean_absolute_percentage_error(Y_True, Y_Pred)
        print("Finished.")
        print("MAPE (%): ", mape)
        print(time.ctime())

        f = open("./Experimental Results.csv", 'a', newline='')
        w = csv.writer(f)
        w.writerow([csvfile[:-4],"RF",rmse,mae,mape])

        f.close()


        Y_Pred = pd.DataFrame(Y_Pred, columns=["Forecast"])
        Y_True.to_csv("./experiment/result/" + TYPE + csvfile + "_real.csv", header=True, index=False)
        Y_Pred.to_csv("./experiment/result/" + TYPE + csvfile + "_forecast.csv", header=True, index=False)

        result.append(["RF", rmse, mae, mape])
        result = np.asarray(result)
        result = pd.DataFrame(result, columns=["Model", "RMSE", "MAE", "MAPE"])
        result.to_csv("./experiment/result/" + TYPE + csvfile + "_RF_result.csv", header=True, index=False)
