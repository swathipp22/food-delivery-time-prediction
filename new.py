#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import plotly.express as px

#Data Reading

fd1=pd.read_csv("deliverytime.csv")
fd=pd.DataFrame(fd1)
print(fd.head())
print('shape:',fd.shape)
print(fd.info())
print(fd.describe())
print(fd.isnull().sum())

#Data Visualization

sns.barplot(data=fd,y='Type_of_vehicle',x='Time_taken(min)')
plt.show()


fig=px.box(fd,x='Type_of_vehicle',y='Time_taken(min)',color='Type_of_order')
fig.show()

fig=plt.figure(figsize=(8,7))
fig.add_subplot(211)
sns.regplot(data=fd,x='Delivery_person_Age',y='Time_taken(min)',line_kws={'color':'red'})
fig.add_subplot(212)
sns.regplot(data=fd,x='Delivery_person_Ratings',y='Time_taken(min)',line_kws={'color':'red'})
plt.tight_layout()
plt.show()

#Data Cleaning

fd=fd.drop(['ID','Delivery_person_ID'],axis=1)
print(fd.head())

def dis(lat1, lon1, lat2, lon2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist
    return km
dis_cols = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude','Delivery_location_longitude']
fd['Distance'] = fd[dis_cols].apply(
    lambda x: dis(x[0], x[1], x[2], x[3]),
    axis=1)

print(fd.head())
print(fd.tail())

num_fd=fd.select_dtypes(exclude='object')
alp_fd=fd.select_dtypes(include='object')
print(num_fd.head())
print(alp_fd.head())

dummy_fd=pd.get_dummies(fd[['Type_of_order', 'Type_of_vehicle']], dtype='int64')
fd=pd.concat([fd,dummy_fd],axis=1)
fd=fd.drop(['Type_of_order', 'Type_of_vehicle'],axis=1)
print(fd.info())

#Data Visualization
  #Heatmap

Correlation_matrix=fd[['Delivery_person_Age','Delivery_person_Ratings','Time_taken(min)','Distance']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(Correlation_matrix,annot=True,cmap='coolwarm',fmt="0.2f",annot_kws={"size":12})
plt.title('Correlation matrix')
plt.show()

#Modelling

X=fd.drop(['Time_taken(min)'],axis=1)
y=fd['Time_taken(min)']

print(X)
print(y)

#Model Comparison

models=[]
models.append(('KNN',KNeighborsRegressor()))
models.append(('LNR',LinearRegression()))
models.append(('DT',DecisionTreeRegressor()))
models.append(('LS',Lasso()))
models.append(('RD',Ridge()))
results=[]
names=[]
scoring='neg_mean_squared_error'
kfold=KFold(n_splits=10)
for name,model in models:
    cv_results=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('cv results:',cv_results)
    print(f"mse of {name} is {cv_results.mean()}")

fig=plt.figure()
fig.suptitle("Algorithm Comparison")
ax=fig.add_subplot(111)
plt.boxplot(results)

ax.set_xticklabels(names)
plt.show()

#Train-test split

X=fd.drop(['Time_taken(min)'],axis=1)
y=fd['Time_taken(min)']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler= StandardScaler()
X_train1= scaler.fit_transform(X_train)

X_test1= scaler.transform(X_test)

print('X_train1:',X_train1.shape)
print('X_test1:',X_test1.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)
print('y:',y.shape)
print('X:',X.shape)

#Linear Regression

  #prediction

lr=LinearRegression()
lr.fit(X_train1,y_train)
plr=lr.predict(X_test1)

print('y=',y)
print('predicted values of lr=',plr)

print('MSE=',mean_squared_error(y_test,plr))
print('RMSE=',np.sqrt(mean_squared_error(y_test,plr)))
print('MAE=',mean_absolute_error(y_test,plr))
print('R2=',r2_score(y_test,plr))

  #Plotting

plt.figure(figsize = (16, 8))
plt.subplot(2,2,1)
plt.scatter(x=y_test, y=plr, color='black')
plt.title('Predicted Points VS. Actual Points for LR', fontdict={'fontsize': 15})
plt.xlabel('Actual Points (y_test)', fontdict={'fontsize': 10})
plt.ylabel('Predicted Points (plr)', fontdict={'fontsize': 10})
plt.show()
print('x=',y_test)
print('y=',plr)

plt.figure(figsize=(10,6))
plt.scatter(x=fd.iloc[:,7:8],y=fd.iloc[:,6:7],label='Real',color='blue')
plt.scatter(x=X_test.iloc[:,6:7],y=plr,label='Prediction',color='red')
plt.title('Real Data v/s Prediction for LR')
plt.xlabel('Distance(m)')
plt.ylabel('Time taken')
plt.legend()
plt.show()
print('x1=',fd.iloc[:,7:8])
print('y1=',fd.iloc[:,6:7])
print('x2=',X_test.iloc[:,6:7])
print('y2=',plr)


#Ridge

  #Prediction

rd=Ridge(alpha=1.0)
rd.fit(X_train1,y_train)
prd=rd.predict(X_test1)

print('y=',y)
print('predicted values of ridge=',prd)

print('MSE=',mean_squared_error(y_test,prd))
print('RMSE=',np.sqrt(mean_squared_error(y_test,prd)))
print('MAE=',mean_absolute_error(y_test,prd))
print('R2=',r2_score(y_test,prd))

  #Plotting

plt.figure(figsize = (16, 8))
plt.subplot(2,2,1)
plt.scatter(x=y_test, y=prd, color='black')
plt.title('Predicted Points VS. Actual Points for RIDGE', fontdict={'fontsize': 15})
plt.xlabel('Actual Points (y_test)', fontdict={'fontsize': 10})
plt.ylabel('Predicted Points (prd)', fontdict={'fontsize': 10})
plt.show()
print('x=',y_test)
print('y=',prd)

plt.figure(figsize=(10,6))
plt.scatter(x=fd.iloc[:,7:8],y=fd.iloc[:,6:7],label='Real',color='blue')
plt.scatter(x=X_test.iloc[:,6:7],y=prd,label='Prediction',color='red')
plt.title('Real Data v/s Prediction for RIDGE')
plt.xlabel('Distance(m)')
plt.ylabel('Time taken')
plt.legend()
plt.show()
print('x1=',fd.iloc[:,7:8])
print('y1=',fd.iloc[:,6:7])
print('x2=',X_test.iloc[:,6:7])
print('y2=',prd)


# Example data for prediction

new_data = {
    'Delivery_person_Age':25,
    'Delivery_person_Ratings':4.2,
    'Restaurant_latitude': 12.971598,
    'Restaurant_longitude': 77.594562,
    'Delivery_location_latitude': 12.914142,
    'Delivery_location_longitude': 77.634720,
    'Distance': dis(12.971598, 77.594562, 12.914142, 77.634720),
    'Type_of_order_Buffet':0,
    'Type_of_order_Drinks':0,
    'Type_of_order_Meal ':0,
    'Type_of_order_Snack':1,
    'Type_of_vehicle_bicycle':0,
    'Type_of_vehicle_electric_scooter':0,
    'Type_of_vehicle_motorcycle':1,
    'Type_of_vehicle_scooter':0
}

new_data_df = pd.DataFrame([new_data])

X_scaled = scaler.fit_transform(new_data_df)

predicted_time = lr.predict(X_scaled)
print("Predicted Delivery Time for LR:", predicted_time[0])

predicted_time = rd.predict(X_scaled)
print("Predicted Delivery Time for RIDGE:", predicted_time[0])


