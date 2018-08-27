import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.misc.pilutil import imread,imsave
import imageio

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(x, y, theta, lmda):
    m = x.shape[0]
    h_theta = pd.DataFrame(sigmoid(np.dot(x,theta)))
    cost = np.asscalar(1/m * ((-np.multiply(y,h_theta.apply(np.log)) - np.multiply(1-y, (1-h_theta).apply(np.log))).sum())) + np.asscalar((lmda/(2*m)) * (theta.iloc[1:,:]**2).sum())
    return cost

def gradient_descent(x, y, theta, lmda):
    global cost_values
    m = x.shape[0]
    iterations = 1500
    alpha = 0.03
    cost_values = pd.DataFrame({'iteration' : [0], 'cost' : [cost(x,y,theta,lmda)]})

    for iteration in range(0,iterations):
        theta_old = theta.copy()
        h_theta = sigmoid(np.dot(x,theta_old))
        beta_ = h_theta - y
        delta_theta = np.dot(x.T, beta_)
        theta_old.iloc[0,0] = 0
        theta = theta - ((alpha/m)*delta_theta) + ((lmda/m) * theta_old)
        c = cost(x,y,theta, lmda)
        cost_values = cost_values.append({"iteration" : iteration, "cost" : c}, ignore_index=True)
    return theta

def predict(x, theta, k):
    probability = sigmoid(np.dot(x.T,theta))
    p = (probability >= 0.5).astype(np.int)
    for i in range(k):
        if(p[i] == 1):
            if(i == 0):
                return 10
            else:
                return i
    return -1

### Load data
column_names = [str(i) for i in range(1,401)]
train_data = pd.read_csv("ex3data1_x.csv", names=column_names)
x = train_data.copy()
x.insert(0, "intercept", 1)
lmda = 3
y = pd.read_csv("ex3data1_y.csv", names=["y"]) 

### Create a sample image from dataset to view
# row = random.randint(0, train_data.shape[0])
# img = train_data.iloc[row,:].values
# img = img.reshape(20,20)
# imageio.imwrite("sample_image.jpg", img)

### Create initial theta
k = 10
theta = pd.DataFrame({"theta_0" : [0] * len(x.columns)})
j = 1
for i in range(1,k):
    theta.insert(j, "theta_" + str(i), 0)

### Print Initial cost
# print("Initial cost: " + str(cost(x,y,theta, lmda)))

### Perform Gradient Descent
for i in range(0,k):
    if(i==0):
        label = 10
    else:
        label = i
    print("Training for " + str(i))
    tmp_y = (y == label).astype(np.int)
    theta.iloc[:,i] = gradient_descent(x, tmp_y, pd.DataFrame(theta.iloc[:,i]), lmda)

    ### Plot iteration vs Cost
    # plt.scatter(cost_values["iteration"], cost_values["cost"])
    # plt.show()

### Write trained theta to csv
# theta = pd.read_csv("theta.csv") #Reuse a trained theta
theta.to_csv("theta.csv", index=False)

### Predict samples
# samples = [0, 536, 1378, 1873, 2303, 2784, 3361, 3812, 4374, 4871] #one sample for each class for testing
# print("predictions for :" + str(k))
# i = 0
# for sample in samples:
#     p = predict(x.iloc[sample,:], theta, k)
#     print("i:"+str(i))
#     print(p,y.iloc[sample,0])

acc = 0
for i in range(0,y.shape[0]):
    p = predict(x.iloc[i,:], theta, k)
    if(p == y.iloc[i,0]):
        acc+=1
print((acc/y.shape[0])*100)