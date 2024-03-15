# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1.Import the required library and read the dataframe.
### 2.Write a function computeCost to generate the cost function.
### 3.Perform iterations og gradient steps with learning rate.
### 4.Plot the Cost function using Gradient Descent and generate the required graph. 
## Program:
```
Developed by: Mohamed Fareed F
RegisterNumber:  212222230082
```
```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
  
  X = np.c_[np.ones(len(X1)), X1]
 
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1, 1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate* (1 / len(X1)) * X.T.dot(errors)
  return theta
data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())

X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled, Y1_Scaled)

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
### data.head():
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121412904/e5a5d683-a5d7-4d7b-b282-97c25e78af16)
### X values: 
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121412904/b0b70430-402f-47dc-a8e1-a3523053da05)
### Y values:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121412904/be063aa6-513b-4e89-a09d-2f74355209d9)
### X scaled:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121412904/53e60ded-a0aa-4423-87a1-13e0fa457ebb)


### Y scaled:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121412904/2bf4f9a0-a01c-4821-9bca-e8efafd683a0)


### Predicted Value:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121412904/ddfaf6a3-f3b0-444e-8854-6fdc97da7480)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
