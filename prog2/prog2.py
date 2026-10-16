import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
data.head()
x = data['']
y = data['Salary']
plt.scatter(x, y)
pl.title()
plt.show()

y = mx + b
m = 0
b = 0
Learning_rate = 0.01
epochs = 1000
n = len(x)
error = []

for i in range(epochs):
    y_hat = m * x + b
    mse = (1/n) * sum((y - y_hat) ** 2)
    error.append(mse)
    dm = (-2/n) * sum(x * (y - y_hat))
    db = (-2/n) * sum(y - y_hat)
    m = m - Learning_rate * dm
    b = b - Learning_rate * db
    

