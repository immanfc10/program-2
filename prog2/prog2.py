import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dev.csv')
data.head()
x = data['dev.x.txt']
y = data['dev.y.txt']
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('dev.x.txt')
plt.ylabel('dev.y.txt')
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
    if i % 100 == 0:
        print(f'Epoch {i}: MSE = {mse}, m = {m}, b = {b}')
        

    

