import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def map_features(x, degree):
    x_old = x.copy()
    x = pd.DataFrame({"intercept" : [1]*x.shape[0]})
    column_index = 1
    for i in range(1,degree+1):
        for j in range(0,i+1):
            x.insert(column_index, str(x_old.columns[1]) + "^" + str(i-j) + str(x_old.columns[2]) + "^" + str(j), np.multiply(x_old.iloc[:,1]**(i-j), x_old.iloc[:,2]**(j)))
            column_index += 1
    return x

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(x, y, theta, lmda):
    m = x.shape[0]
    h_theta = pd.DataFrame(sigmoid(np.dot(x,theta)))
    a = np.asscalar(((-np.multiply(y,h_theta.apply(np.log)) - np.multiply(1-y, (1-h_theta).apply(np.log))).sum()) / m)
    b = np.asscalar((lmda/(2*m)) * ((theta.iloc[1:,:]**2).sum()))
    return a+b

def gradient_descent(x, y, theta, lmda):
    global cost_values
    m = x.shape[0]
    iterations = 1500
    alpha = 0.03
    cost_values = pd.DataFrame({'iteration' : [0], 'cost' : [cost(x,y,theta, lmda)]})

    for iteration in range(0,iterations):
        theta_old = theta.copy()
        theta.iloc[0,0] = theta.iloc[0,0] - (alpha/m) * np.asscalar((sigmoid(np.dot(x,theta_old)) - y).sum())
        for i in range(1,theta.shape[0]):
            theta.iloc[i,0] = theta.iloc[i,0] - ((alpha/m) * np.asscalar(np.multiply((sigmoid(np.dot(x,theta_old)) - y), pd.DataFrame(x.iloc[:,i])).sum()) + ((lmda/m) * theta.iloc[i,0]))
        c = cost(x,y,theta, lmda)
        cost_values = cost_values.append({"iteration" : iteration, "cost" : c}, ignore_index=True)

def predict(x):
    global theta
    probability = np.asscalar(sigmoid(np.dot(x.T,theta)))
    return probability

def normalize_features(x):
    for column_name in x.columns[1:]:
        mean = x[column_name].mean()
        std = x[column_name].std()
        x[column_name] = (x[column_name] - mean) / std
    return x

train_data = pd.read_csv("ex2data2.csv", names=["test1","test2","y"])

### Visualize data set
# plt.scatter(train_data[train_data["y"] == 1]["test1"], train_data[train_data["y"] == 1]["test2"], marker="o")
# plt.scatter(train_data[train_data["y"] == 0]["test1"], train_data[train_data["y"] == 0]["test2"], marker="+")
# plt.show()

### Create input
x = train_data.loc[:, "test1":"test2"]
x.insert(0, "intercept", 1)
mapping_degree = 10
x = normalize_features(x)
x = map_features(x, mapping_degree)
y = pd.DataFrame(train_data.loc[:, "y"])
theta = pd.DataFrame({"theta" : [0] * len(x.columns)})
lmda = [0] #lamda values to test

### Test cost of initial theta
# print(cost(x,y,theta, lmda))

### Perform regularized gradient descent
for l in lmda:
    print("lambda: " + str(l))
    ### Perform Gradient Descent
    theta = pd.DataFrame({"theta" : [0] * len(x.columns)})
    gradient_descent(x, y, theta, l)

    ### Plot iteration vs Cost
    plt.scatter(cost_values["iteration"], cost_values["cost"])
    plt.show()

    ### Get Accuracy
    acc = 0
    for i in range(0,x.shape[0]):
        p = predict(pd.DataFrame(x.iloc[i,:]))
        if(p >= 0.5):
            p = 1
        else:
            p = 0
        actual = y.iloc[i,0]
        if(p == actual):
            acc+=1
    print((acc/x.shape[0]) * 100)

    ### Plot decision boundary
    x_min = train_data["test1"].min()
    x_max = train_data["test1"].max()
    y_min = train_data["test2"].min()
    y_max = train_data["test2"].max()
    diff = 0.1
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, diff), np.arange(y_min, y_max, diff))
    xx = pd.DataFrame(x_grid.ravel(), columns=["test1"])
    yy = pd.DataFrame(y_grid.ravel(), columns=["test2"])
    z = pd.DataFrame({"intercept" : [1]*xx.shape[0]})
    z["test1"] = xx
    z["test2"] = yy
    z = normalize_features(z)
    z = map_features(z,mapping_degree)
    p = z.apply(lambda row: predict(pd.DataFrame(row)), axis=1)
    p = np.array(p.values)
    p = p.reshape(x_grid.shape)
    plt.scatter(train_data[train_data["y"] == 0]["test1"], train_data[train_data["y"] == 0]["test2"],marker="o")
    plt.scatter(train_data[train_data["y"] == 1]["test1"], train_data[train_data["y"] == 1]["test2"],marker="x")
    plt.contour(x_grid, y_grid, p, levels = [0.5])
    # plt.contour(x_grid, y_grid, p, 50, cmap="RdBu")
    # plt.colorbar()
    plt.show()