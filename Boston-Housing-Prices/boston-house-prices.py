
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


from sklearn.datasets import load_boston
house_price=load_boston()

df=pd.DataFrame(house_price.data,columns=house_price.feature_names)

df['PRICE'] = house_price.target

# splitting the dataset into training and test sets with 7:3 ratio along with randomly shuffling
X_train, X_test, y_train, y_test = train_test_split(df, df['PRICE'], test_size=0.30, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)

# deleting the Price Column from the training part as it is the Output variable
del X_train['PRICE']
X_train.shape
del X_test['PRICE']
X_test.shape

# normal linear REgression
reg=LinearRegression().fit(X_train,y_train)

print(reg.coef_)
print("The intercept values",reg.intercept_)

# here predicting the residuals for the normal regression model.
predictionlr=reg.predict(X_train)
residuals=(y_train-predictionlr)

residuals_mean=np.mean(np.abs(residuals))
lst=list(range(0,len(residuals)))
# for the training data
print('for the training data')
print('R^2:',metrics.r2_score(y_train, predictionlr))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, predictionlr))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, predictionlr))
print('MSE:',metrics.mean_squared_error(y_train, predictionlr))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, predictionlr)))

# Predicting Test data with the model
y_test_pred = reg.predict(X_test)
# Model Evaluation
acc_linreg = metrics.r2_score(y_test, y_test_pred)
print('for the test data')
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))




