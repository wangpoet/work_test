import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
###################################################

train_data = pd.read_csv("train_data.csv")
#test_data = pd.read_csv("test_data4.csv")

########################################################input data

y = train_data.y
X = train_data.drop(['y'],axis=1)
#val_y = train_data.y
#val_X =train_data.drop(['y'],axis=1)
#test_y = test_data.y
#test_X =test_data.drop(['y'],axis=1) 


#xgb
#xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
#xgb_test = xgb.DMatrix(test_X,label=test_y)

###############################################data transfor


params={'max_depth':2,'eta':0.1,'silent':1,'booster':'gblinear'} 
plst = list(params.items())
num_rounds = 5000 
watchlist = [(xgb_train, 'train')]
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
model.save_model('./model/xgb.model') 
print "best best_ntree_limit",model.best_ntree_limit
#y_hat=model.predict(xgb_test) 
#print y_hat
############################################################train 
