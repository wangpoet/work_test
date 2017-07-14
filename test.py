import  xgboost  as  xgb


x=[1,2,3,4,5]
y=[2,4,6,8,9]
x_test=[[6],[1]]
y_test=[]
###################################input data

x1=[0,1,1,1,1]
x_train=[x,x1]
#print  x_train
y_train=y
###################################data build 

data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)   
watch_list = [(data_test, 'eval'), (data_train, 'train')]  
param = {'max_depth': 2, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}  
bst = xgb.train(param, data_train, num_boost_round=2, evals=watch_list)  
y_hat = bst.predict(data_test)  
print y_hat
