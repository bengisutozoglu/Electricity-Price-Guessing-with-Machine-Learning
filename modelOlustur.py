import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
import warnings




warnings.filterwarnings('ignore')

column_name = ["gun", "mevsim", "talep", "elektrikfiyat", "gazfiyat", "uretim"]
data = pd.read_csv("elekdata3.csv",names=column_name, sep=";",comment="\t",decimal=",")
##data = pd.read_csv("elekdata3.csv", names=column_name, na_values="?", comment="\t", sep=";",
##                   skipinitialspace=True, decimal=',')

y = data.elektrikfiyat
x = data.drop(["elektrikfiyat"], axis=1)

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=123,shuffle=1)
print(X_train)
exit()
sonuclar = dict()

#knn model ve tahmin
knn_model = KNeighborsRegressor()
# knn Model Tuning
knn_params = {"n_neighbors": np.arange(1,30,1)}
knn_cv_model = GridSearchCV(knn_model, knn_params, cv = 10).fit(X_train, y_train)
#final model
knn_tuned = KNeighborsRegressor(n_neighbors = knn_cv_model.best_params_["n_neighbors"]).fit(X_train, y_train)
knnPred = knn_tuned.predict(x_test)
#Hata kareler ortalamasının krekökünün hesaplanması
knnSonuc = np.sqrt(mean_squared_error(y_test, knnPred))
sonuclar["KNN"] = knnSonuc
save_classifier = open("models/knnModel.pickle","wb")
pickle.dump(knn_tuned,save_classifier)
save_classifier.close


# SVR
svr_model = SVR("linear")
# Tuning
svr_params = {"C": [0.1,0.5,1,3]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv = 5).fit(X_train, y_train)
svr_tuned = SVR("linear", C = svr_cv_model.best_params_["C"]).fit(X_train, y_train)
svrPred = svr_tuned.predict(x_test)
knnSonuc = np.sqrt(mean_squared_error(y_test, svrPred))
sonuclar["SVR"] = knnSonuc
save_classifier = open("models/SVRModel.pickle","wb")
pickle.dump(svr_tuned,save_classifier)
save_classifier.close

# MLP Model
mlp_model = MLPRegressor()
#Tuning
mlp_params = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001], 
             "hidden_layer_sizes": [(10,20), (5,5), (100,100)]}
mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10, verbose = 2, n_jobs = -1).fit(X_train, y_train)
mlp_tuned = MLPRegressor(alpha = mlp_cv_model.best_params_["alpha"], hidden_layer_sizes = mlp_cv_model.best_params_["hidden_layer_sizes"]).fit(X_train, y_train)
mlpPred = mlp_tuned.predict(x_test)
knnSonuc = np.sqrt(mean_squared_error(y_test, mlpPred))
sonuclar["MLP"] = knnSonuc
save_classifier = open("models/MLPModel.pickle","wb")
pickle.dump(mlp_tuned,save_classifier)
save_classifier.close

#Random Forest
rf_model = RandomForestRegressor()
#Tuning
rf_params = {"max_depth": [5,8,10],
            "max_features": [2,5,10],
            "n_estimators": [200, 500, 1000, 2000],
            "min_samples_split": [2,10,80,100]}
rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
rf_model = RandomForestRegressor(random_state = 42, 
                                 max_depth = rf_cv_model.best_params_["max_depth"],
                                max_features = rf_cv_model.best_params_["max_features"],
                                min_samples_split = rf_cv_model.best_params_["min_samples_split"],
                                 n_estimators = rf_cv_model.best_params_["n_estimators"])
rf_tuned = rf_model.fit(X_train, y_train)
rfPred = rf_tuned.predict(x_test)
knnSonuc = np.sqrt(mean_squared_error(y_test, rfPred))
sonuclar["Random"] = knnSonuc
save_classifier = open("models/rfModel.pickle","wb")
pickle.dump(rf_tuned,save_classifier)
save_classifier.close


#Gradient Boosting Machines
gbm_model = GradientBoostingRegressor()
#Tuning
gbm_params = {"learning_rate": [0.001,0.1,0.01],
             "max_depth": [3,5,8],
             "n_estimators": [100,200,500],
             "subsample": [1,0.5,0.8],
             "loss": ["ls","lad","quantile"]}
gbm_cv_model = GridSearchCV(gbm_model, 
                            gbm_params, 
                            cv = 10, 
                            n_jobs=-1, 
                            verbose = 2).fit(X_train, y_train)
gbm_tuned = GradientBoostingRegressor(learning_rate = gbm_cv_model.best_params_["learning_rate"],
                                     loss = gbm_cv_model.best_params_["loss"],
                                     max_depth = gbm_cv_model.best_params_["max_depth"],
                                     n_estimators = gbm_cv_model.best_params_["n_estimators"],
                                     subsample = gbm_cv_model.best_params_["subsample"]).fit(X_train, y_train)
gbmPred = gbm_tuned.predict(x_test)
knnSonuc = np.sqrt(mean_squared_error(y_test, gbmPred))
sonuclar["Gradient Boosting"] = knnSonuc
save_classifier = open("models/GradientModel.pickle","wb")
pickle.dump(gbm_tuned,save_classifier)
save_classifier.close



# XGBoost
import xgboost
from xgboost import XGBRegressor
xgb = XGBRegressor().fit(X_train, y_train)
#Tuning
xgb_params = {"learning_rate": [0.1,0.01,0.5],
             "max_depth": [2,3,4,5,8],
             "n_estimators": [100,200,500,1000],
             "colsample_bytree": [0.4,0.7,1]}
xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
xgb_tuned = XGBRegressor(colsample_bytree = xgb_cv_model.best_params_["colsample_bytree"], 
                         learning_rate = xgb_cv_model.best_params_["learning_rate"], 
                         max_depth = xgb_cv_model.best_params_["max_depth"], 
                         n_estimators = xgb_cv_model.best_params_["n_estimators"]).fit(X_train, y_train)

xgbPred = xgb_tuned.predict(x_test)
knnSonuc = np.sqrt(mean_squared_error(y_test, xgbPred))

sonuclar["XGB"] = knnSonuc
save_classifier = open("models/XGBModel.pickle","wb")
pickle.dump(xgb_tuned,save_classifier)
save_classifier.close



from lightgbm import LGBMRegressor
lgb_model = LGBMRegressor().fit(X_train, y_train)
#Tuning
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "n_estimators": [20,40,100,200,500,1000],
              "max_depth": [1,2,3,4,5,6,7,8,9,10]}
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)
lgbm_tuned = LGBMRegressor(learning_rate = lgbm_cv_model.best_params_["learning_rate"], 
                          max_depth = lgbm_cv_model.best_params_["max_depth"], 
                          n_estimators = lgbm_cv_model.best_params_["n_estimators"]).fit(X_train, y_train)

lbgPred = lgbm_tuned.predict(x_test)
knnSonuc = np.sqrt(mean_squared_error(y_test, lbgPred))
sonuclar["LGB"] = knnSonuc
save_classifier = open("models/lgbModel.pickle","wb")
pickle.dump(lgbm_tuned,save_classifier)
save_classifier.close

print(x_test)
print(sonuclar)

##fig = plt.figure(figsize=(15,5))
##plt.plot(knnPred,label="knnPred")
##plt.plot(cvPred,label="cvPred")
##plt.plot(svrPred,label="svrPred")
##plt.plot(mlpPred,label="mlpPred")
##plt.plot(rfPred,label="rfPred")
##plt.plot(gbmPred,label="gbmPred")
##plt.plot(xgbPred,label="xgbPred")
##plt.plot(lbgPred,label="lbgPred")
##plt.plot(catbPred,label="catbPred")
##fig.legend(loc="best")
##plt.show()






