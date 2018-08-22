#LINEAR REGRESSION WITH ONE VARIABLE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_cost(x, y, theta, m):
    cost = np.asscalar((1/(2*m)) * ((np.dot(x,theta)-y)**2).sum())
    return cost

def Gradient_Descent(x, y, theta):
    global cost_values
    m = x.shape[0]
    iterations = 1500
    alpha = 0.01
    cost_values = pd.DataFrame({'iteration' : [0], 'cost' : [calculate_cost(x,y,theta,m)]})
    contour_vals = pd.DataFrame({'theta_0' : [0], 'theta_1' : [0], 'cost' : [calculate_cost(x,y,theta,m)]})

    for iteration in range(1, iterations):
        theta_old = theta.copy()
        theta.ix[0] = theta.ix[0] - (alpha*(1/m)) * np.asscalar((np.dot(x, theta_old)-y).sum())
        for j in range(1,theta.shape[0]):
            theta.ix[j] = theta.ix[j] - (alpha*(1/m)) * np.asscalar(np.multiply((np.dot(x, theta_old)-y), pd.DataFrame(x.loc[:,x.columns[j]])).sum())
        cost = calculate_cost(x, y, theta, m)
        cost_values = cost_values.append({'iteration' : iteration, 'cost' : cost}, ignore_index=True)

    return theta

def hypothesis(x):
    global theta
    return np.asscalar(theta.ix[0]) + np.asscalar(theta.ix[1])*x

#load data
train_data = pd.read_csv(r"ex1data1.csv", header=None, names=["population", "profit"])

#visualize data
plt.scatter(train_data['population'], train_data['profit'], marker='.')
plt.xlabel('population')
plt.ylabel('price')
plt.show()

train_data.insert(0, "intercept", 1)
x = pd.DataFrame(train_data.loc[:,'intercept':'population'])
y = pd.DataFrame(train_data["profit"])
theta = pd.DataFrame({'theta' : [0,0]})
theta = Gradient_Descent(x,y,theta)

# View graph of #iteration vs cost 
plt.scatter(cost_values['iteration'], cost_values['cost'], marker='.')
plt.xlabel('iteration')
plt.ylabel('cost_J')
plt.show()

#Predict values
print(theta)
predict1 = np.asscalar(theta.ix[0] * 1 +  3.5 * theta.ix[1])
predict2 = np.asscalar(theta.ix[0] * 1 +  7 * theta.ix[1])
print(predict1)
print(predict2)

#show plotted prediction line
x = np.array(range(4,25))
y = hypothesis(x)
plt.scatter(train_data['population'], train_data['profit'], marker='.')
plt.plot(x,y)
plt.show()