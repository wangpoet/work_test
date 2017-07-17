import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
###################################################
import time 
start_time = time.time()
test_data = pd.read_csv("test_data.csv")
test_y = test_data.y
test_X =test_data.drop(['y'],axis=1) 
xgb_test = xgb.DMatrix(test_X,label=test_y)
model = xgb.Booster(model_file='./model/xgb.model')  
y_hat=model.predict(xgb_test) 
print y_hat
