import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def linear_regression(x) :
    return 1 + 3 * x

data = pd.read_csv('./population.csv', names=['Population', 'Profit'])



data.insert(0,'One',1) 
print(data)
# separate x training data from y target variable

cols = data.shape[1] 

X = data.iloc[:, 0:cols-1] # will take 0 , 1 columns
y = data.iloc[:, cols-1:cols]  # easy


# convert from dataframes to numpy matrices

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0]))


def cost_function(X, y, theta) :
    
    z = np.power(((X * theta.T) - y), 2)
    
    return np.sum(z) / (2 * len(X))


print(f'J(th0, th1) = {cost_function(X, y, theta)}')


# GD function

def gradient_decsent(X, y, theta, alpha, iterations) :
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1]) 
    cost = np.zeros(iterations)
    
    for i in range(iterations):
        error = (X * theta.T) - y
        
        for j in range(parameters) :
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha/ len(X)) * np.sum(term))
    
        theta = temp
        cost[i] = cost_function(X, y, theta)    
    
    return theta, cost


# initialize variables for learning rate and iterations

alpha = 0.01
iterations = 1000

# preform drafient decsent to "Fit" the model parameters
g, cost = gradient_decsent(X, y, theta, alpha, iterations)


print(f'g : {g}')
print(f'cost : {cost[0:50]}')

print(f'cost_function(X, y, g) : {cost_function(X, y, g)}')



# get best fit line

x = np.linspace(data.Population.min(), data.Population.max(), 100)

# th0 + th1X 
f = g[0, 0] + (g[0, 1] * x)

# draw the line 

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs Population Size')


# draw error graph

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

















