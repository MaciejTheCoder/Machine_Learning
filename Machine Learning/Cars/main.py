import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

auto = pd.read_csv(r"auto-mpg.csv")
auto.head()
auto.shape
X = auto.iloc[:,1:-1]
X = X.drop('horsepower', axis=1)
y = auto.iloc[:,0]
print (X)

lr =  LinearRegression()
lr.fit(X.to_numpy(),y)
lr.score(X.to_numpy(),y)

my_car1 = [4, 160, 1900, 12, 90, 1]
my_car2 = [4, 200, 2600, 15, 83, 1]
cars = [my_car1, my_car2]
 
mpg_predict = lr.predict(cars)
print(mpg_predict)

