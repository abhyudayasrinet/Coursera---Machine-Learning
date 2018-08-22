import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def map_features(x, degree):
    x_old = x.copy()
    x = pd.DataFrame({"intercept" : [1]*x.shape[0]})
    column_index = 1
    for i in range(1, degree+1):
        for j in range(0, i+1):
            x.insert(column_index, x_old.columns[1] + "^" + str(i-j) + x_old.columns[2] + "^" + str(j), np.multiply(x_old.iloc[:,1]**(i-j), x_old.iloc[:,2]**(j)))
            column_index+=1
    return x

def normalize_features(x):
    for column_name in x.columns[1:]:
        mean = x[column_name].mean()
        std = x[column_name].std()
        x[column_name] = (x[column_name] - mean) / std
    return x

def sigmoid(z):
    # print(z)
    return 1/(1+np.exp(-z))

def predict(x):
    global theta
    probability = np.asscalar(sigmoid(np.dot(x,theta)))
    if(probability >= 0.5):
        return 1
    else:
        return 0

def cost(x, y, theta):
    m = x.shape[0]
    h_theta = pd.DataFrame(sigmoid(np.dot(x,theta)))
    cost = 1/m * ((-np.multiply(y,h_theta.apply(np.log)) - np.multiply(1-y, (1-h_theta).apply(np.log))).sum())
    return cost

def gradient_descent(x, y, theta):
    global cost_values
    m = x.shape[0]
    iterations = 2500
    alpha = 0.03
    cost_values = pd.DataFrame({'iteration' : [0], 'cost' : [cost(x,y,theta)]})

    for iteration in range(0,iterations):
        theta_old = theta.copy()
        theta.iloc[0,0] = theta.iloc[0,0] - (alpha/m) * np.asscalar((sigmoid(np.dot(x,theta_old)) - y).sum())
        for i in range(1,theta.shape[0]):
            theta.iloc[i,0] = theta.iloc[i,0] - (alpha/m) * np.asscalar(np.multiply((sigmoid(np.dot(x,theta_old)) - y), pd.DataFrame(x.iloc[:,i])).sum())
        c = cost(x,y,theta)
        cost_values = cost_values.append({"iteration" : iteration, "cost" : c}, ignore_index=True)


### Read train data
train_data = pd.read_csv("ex2data1.csv", names = ["exam1", "exam2", "admit"])

### Add intercept column
train_data.insert(0, "intercept", 1)

### Create input data
x = train_data.loc[:,"intercept":"exam2"]
# print(x.head())
x = map_features(x, 2) #map polynomial features
# print(x.head())
x = normalize_features(x) #normalize features
# print(x.head())
y = pd.DataFrame(train_data.loc[:,"admit"])
theta = pd.DataFrame({"theta" : [0] * len(x.columns)})

### Test cost of initial theta
# print(x.shape)
# print(theta.shape)
# print(np.dot(x,theta))
# print(cost(x,y,theta))

### Perform Gradient Descent
gradient_descent(x, y, theta)
# print(theta)
# print(cost(x,y,theta))

### Plot iteration vs Cost
plt.scatter(cost_values["iteration"], cost_values["cost"])
plt.show()

### Calculate Accuracy
acc = 0
for i in range(0,x.shape[0]):
    p = predict(x.iloc[i,:])
    actual = y.iloc[i,0]
    if(p == actual):
        acc+=1
print((acc/x.shape[0]) * 100)

### Plot Decision Boundary
x = np.array(range(42,51))
y = hypothesis(x)
plt.scatter(train_data[train_data["admit"] == 0]["exam1"], train_data[train_data["admit"] == 0]["exam2"],marker="o")
plt.scatter(train_data[train_data["admit"] == 1]["exam1"], train_data[train_data["admit"] == 1]["exam2"],marker="x")
plt.plot(x,y)
plt.show()