
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import accuracy_score
import sklearn

df=pd.read_excel(r"C:\Users\ancha\OneDrive\Documents\SVM\rough_data\Concrete_Data.xls")
#df= np.array(df)


cols = len(df.axes[1])
rows = len(df.axes[0])
#print(cols)
#print(rows)

#checking missing values
#print(norm_df.isnull())
mean_col=list(df.mean(axis=0))
#print(mean_col)

if df.isnull=='True':
   for i in range(cols):
       (df.iloc[:,i]).fillna(mean_col[i])
    

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_df= pd.DataFrame(x_scaled)
#print(norm_df)

X=norm_df.iloc[:,0:cols-1]
y=norm_df.iloc[:,cols-1]
#print(len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_test=np.array(X_test)
#print(y_train)
#print(X_train)
def rbf_kernel(gamma,C,X_train,y_train,X_test,y_test):

   svr_rbf = SVR(kernel="rbf", C=C, gamma=gamma)
   y_pred = svr_rbf.fit(X_train, y_train).predict(X_test)
   MSE=sklearn.metrics.mean_squared_error(y_test,y_pred)
   RMSE = math.sqrt(MSE)
   MAE=sklearn.metrics.mean_absolute_error(y_test, y_pred)
   R2=sklearn.metrics.r2_score(y_test, y_pred)
   return MSE,RMSE,MAE,R2

   
X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
#slicing          
#print(type(X_TRAIN))
#print(type(X_TEST))
#print(type(y_TRAIN))
#print(type(y_TEST))
K=5
L=int(X_TRAIN.shape[0]/K)
L_arr=list(np.arange(0,X_train.shape[0],L))
L_arr=np.append(L_arr,X_train.shape[0])
#print(L_arr)

X_Train=[]
y_Train=[]
left=X_train.shape[0]-(L*K)
for i in range(K):
     if left==0:
         X_Train.append(X_TRAIN[L_arr[i]:L_arr[i+1]])
         y_Train.append(y_TRAIN[L_arr[i]:L_arr[i+1]])
     else:
         if 0<=i<K-1:
             X_Train.append(X_TRAIN[L_arr[i]:L_arr[i+1]])
             y_Train.append(y_TRAIN[L_arr[i]:L_arr[i+1]])
         elif i==K-1:
             X_Train.append(X_TRAIN[L_arr[i]:L_arr[i+2]])
             y_Train.append(y_TRAIN[L_arr[i]:L_arr[i+2]])
                
X_Train=np.array(X_Train,dtype=object)   
y_Train=np.array(y_Train,dtype=object)    

A=np.linspace(-10,10,21)
C=[]
for i in A:
    C.append(pow(2,i))
gamma=C
e=[]
      
for l in range(len(gamma)):
    for k in range(len(C)):
        error_1=[]
        
        for i in range(K):
            Xk_t=[]
            Xk_test1=[]
            yk_t=[]
            yk_test1=[]
            for j in range(K):
                if j==i:
                    Xk_test1.append(X_Train[j])
                    yk_test1.append(y_Train[j])
                else:
                     Xk_t.append(X_Train[j])
                     yk_t.append(y_Train[j])
            Xk_train=np.concatenate(Xk_t)
          
            yk_train=np.concatenate(yk_t)
            Xk_test=np.concatenate(Xk_test1)
            yk_test=np.concatenate(yk_test1)
          
           #print(type(Xk_train))
           #print(type(yk_train))
           #print( type(Xk_test))
           #print(type(yk_test))
            MSE,RMSE,MAE,R2=rbf_kernel(gamma[l],C[k],Xk_train,yk_train,Xk_test,yk_test)
            error_1.append(MSE)
      
        #print(error_1)
        error=np.mean(np.array(error_1))
        #print(error)
        e.append((gamma[l],C[k],error))
        #print(len(e))#


len=len(C)*len(gamma)  

min=[]
for i in range(len):
    min.append(e[i][2])
#print(min)
min_error=np.min(np.array(min))
min_e_index=min.index(min_error)
#print(min_e_index)
C=e[min_e_index][1]
gamma=e[min_e_index][0]
print(f"optimum gamma:{gamma}")
print(f"optimum C:{C}")
    
    
MSE,RMSE,MAE,R2= rbf_kernel(gamma,C,X_TEST,y_TEST,X_test,y_test)
print(f"MSE:{MSE}")
print(f"RMSE:{RMSE}")
print(f"MAE:{MAE}")
print(f"R2:{R2}")
         