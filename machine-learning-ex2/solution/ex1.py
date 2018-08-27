import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
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
    iterations = 1500
    alpha = 0.001
    cost_values = pd.DataFrame({'iteration' : [0], 'cost' : [cost(x,y,theta)]})

    for iteration in range(0,iterations):
        theta_old = theta.copy()
        theta.iloc[0,0] = theta.iloc[0,0] - (alpha/m) * np.asscalar((sigmoid(np.dot(x,theta_old)) - y).sum())
        for i in range(1,theta.shape[0]):
            theta.iloc[i,0] = theta.iloc[i,0] - (alpha/m) * np.asscalar(np.multiply((sigmoid(np.dot(x,theta_old)) - y), pd.DataFrame(x.iloc[:,i])).sum())
        c = cost(x,y,theta)
        cost_values = cost_values.append({"iteration" : iteration, "cost" : c}, ignore_index=True)

def hypothesis(x):
    global theta
    return (0.5 - theta.iloc[0,0] - theta.iloc[1,0]*x)/theta.iloc[2,0]

train_data = pd.read_csv("ex2data1.csv", names=["exam1","exam2","admit"])
train_data.insert(0, "intercept", 1)

### Visualize the data set
plt.scatter(train_data[train_data["admit"] == 0]["exam1"], train_data[train_data["admit"] == 0]["exam2"],marker="o")
plt.scatter(train_data[train_data["admit"] == 1]["exam1"], train_data[train_data["admit"] == 1]["exam2"],marker="x")
plt.show()

### Testing sigmoid
# print(sigmoid(0))
# print(sigmoid(1000))
# print(sigmoid(-1000))

x = pd.DataFrame(train_data.loc[:,"intercept":"exam2"])
y = pd.DataFrame(train_data.loc[:,"admit"])
m = train_data.shape[0]
theta = pd.DataFrame({"theta" : [0] * len(x.columns)})

### Test cost of initial theta
print(cost(x,y,theta))

### Perform Gradient Descent
gradient_descent(x, y, theta)
print(theta)

### Plot iteration vs Cost
plt.scatter(cost_values["iteration"], cost_values["cost"])
plt.show()

### Calculate Accuracy
prediction = pd.DataFrame({"intercept":[1], "exam1":[45], "exam2":[85]})
print(predict(prediction)) #sigmoid(np.dot(prediction, theta)))
acc = 0
for i in range(0,train_data.shape[0]):
    p = predict(train_data.loc[i,"intercept":"exam2"])
    actual = train_data.loc[i,"admit"]
    if(p == actual):
        acc+=1
print((acc/train_data.shape[0]) * 100)

### Plot Decision Boundary
x = np.array(range(42,51))
y = hypothesis(x)
plt.scatter(train_data[train_data["admit"] == 0]["exam1"], train_data[train_data["admit"] == 0]["exam2"],marker="o")
plt.scatter(train_data[train_data["admit"] == 1]["exam1"], train_data[train_data["admit"] == 1]["exam2"],marker="x")
plt.plot(x,y)
plt.show()