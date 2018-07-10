#LINEAR REGRESSION WITH MULTIPLE VARIABLES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_cost(x, y, theta):
    m = x.shape[0]
    cost = np.asscalar((1/(2*m)) * ((np.dot(x,theta)-y)**2).sum())
    return cost


def normalize_features(data):
    for column in data.columns:
        means[column] = data[column].mean()
        stds[column] = data[column].std()
        data[column] = (data[column] - data[column].mean()) / data[column].std()

    return data

def Gradient_Descent(x,y,theta):
    global cost_values
    m = x.shape[0]
    iterations = 1500
    alpha = 0.1
    
    for iteration in range(1, iterations):
        theta_old = theta.copy()
        theta.ix[0] = theta.ix[0] - (alpha*(1/m)) * np.asscalar((np.dot(x, theta_old)-y).sum())
        for j in range(1,theta.shape[0]):
            theta.ix[j] = theta.ix[j] - (alpha*(1/m)) * np.asscalar(np.multiply((np.dot(x, theta_old)-y), pd.DataFrame(x.loc[:,x.columns[j]])).sum())
        cost = calculate_cost(x, y, theta)
        cost_values = cost_values.append({'iteration' : iteration, 'cost' : cost}, ignore_index=True)
    return theta

means = {}
stds = {}

#load data
train_data = pd.read_csv(r"ex1data2.csv", header=None, names=["sqft","bedrooms", "price"])

# visualize data
plt.scatter(train_data['bedrooms'], train_data['price'], marker='.')
plt.xlabel('bedrooms')
plt.ylabel('price')
plt.show()
plt.scatter(train_data['sqft'], train_data['price'], marker='.')
plt.xlabel('square feet')
plt.ylabel('price')
plt.show()

# create input
x = pd.DataFrame(train_data.loc[:,'sqft':'bedrooms'])
y = pd.DataFrame(train_data["price"])
x = normalize_features(x)
x.insert(0, "intercept", 1)
theta = pd.DataFrame({'theta' : [0,0,0]})

cost_values = pd.DataFrame({'iteration' : [0], 'cost' : [calculate_cost(x,y,theta)]})
theta = Gradient_Descent(x, y, theta)
print(theta)

# View graph of #iteration vs cost 
plt.scatter(cost_values['iteration'], cost_values['cost'], marker='.')
plt.xlabel('iteration')
plt.ylabel('cost_J')
plt.show()

#predict house with 3bedrooms and 1650 space
test_features = pd.DataFrame({'intercept' : [1], 'sqft' : [(1650-means['sqft'])/stds['sqft']], 'bedrooms' : [(3-means['bedrooms'])/stds['bedrooms']]})
print(np.asscalar(np.dot(test_features, theta)))

# Obtaining via Normal Equation
x_transpose = x.transpose()
inv = x_transpose.dot(x)
inv = pd.DataFrame(np.linalg.pinv(inv.values), inv.columns, inv.index)
theta = pd.DataFrame(np.dot(np.dot(inv, x_transpose), y))
print(theta)

# Predict house with 3bedrooms and 1650 using new theta
print(np.asscalar(np.dot(test_features, theta)))