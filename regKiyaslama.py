import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import neighbors
from sklearn.svm import SVR
import warnings

def compML(data, y, alg):
    #train-test ayrimi
    y = data[y]
    x_ = data.drop(data, axis=1).astype('float64')
    x = pd.concat([x_, data], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
    #modelleme
    model = alg().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    model_ismi = alg.__name__
    print(model_ismi, "Modeli Test Hatası:",RMSE)


column_name = ["gun", "mevsim", "talep", "elektrikfiyat", "gazfiyat", "uretim"]
data = pd.read_csv("elekdata3.csv", names=column_name, na_values="?", comment="\t", sep=";",
                   skipinitialspace=True, decimal=',')

models = [XGBRegressor,
          GradientBoostingRegressor,
          RandomForestRegressor,
          DecisionTreeRegressor,
          MLPRegressor,
          KNeighborsRegressor,
          SVR]

for i in models:
    compML(data, "elektrikfiyat", i)
