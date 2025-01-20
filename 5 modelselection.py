import pandas as pd
import numpy as np
df = pd.read_csv('/content/automobile.csv')
df.head()


#Linear Regression
from sklearn.linear_model import LinearRegression
df_copy = df.copy()
lm = LinearRegression()
x = df[['highway-mpg']]

y = df['Price']
lm.fit(x,y)
lm_for_copy = LinearRegression()
x1 = df_copy[['highway-mpg']]
y1 = df_copy['Price']
lm_for_copy.fit(x1,y1)
lm_for_copy.predict(x)
intercept = lm_for_copy.intercept_
slope = lm_for_copy.coef_
print(intercept)
print(slope)
desired_mpg = int(input('Enter a highway_mpg for predicting price: '))
y = slope * desired_mpg + intercept
print(y)


#Multiple Regression
multi_reg_model = LinearRegression()
x = df[['highway-mpg','horsepower']]
y = df['Price']
multi_reg_model.fit(x,y)
multi_reg_model.predict(x)
intercept = multi_reg_model.intercept_
slope = multi_reg_model.coef_
print(intercept)
print(slope)
desired_mpg = int(input('Enter a highway_mpg for predicting price: '))

horsepower = int(input('Enter a horsepower for predicting price: '))
y = (slope[0] * desired_mpg +intercept) + (slope[1] * horsepower + intercept)
print(y)


#Regressio Plot
import seaborn as sns
sns.regplot(x="Price",y="highway-mpg",data=df)