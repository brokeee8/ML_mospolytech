sigmoid

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

def relu(x):
    return (x > 0) * x

def reluderif(x):
    return x > 0

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_size = len(x[0])
hidden_size = 4
output_size = len(y[0])

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 100000


for epoch in range(epochs):
    layer_hid = relu(np.dot(x, weight_hid)) #

    layer_out = sigmoid(np.dot(layer_hid, weight_out))
    error = (layer_out - y) ** 2
    
    layer_out_delta = (layer_out - y) * sigmoid_deriv(layer_out)
    layer_hidden_delta = layer_out_delta.dot(weight_out.T) * reluderif(layer_hid) #

    weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
    weight_hid -= learning_rate * x.T.dot(layer_hidden_delta)
    if epoch % 1000 == 0:
        error = np.mean(error)
        print(f"Epoch: {epoch}, Error: {error}")


new_input = np.array([[0,1]])
layer_hid = relu(new_input.dot(weight_hid))
layer_out = sigmoid(layer_hid.dot(weight_out))
print("Предсказание: ", layer_out)

tanh

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1-x**2

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_size = len(x[0])
hidden_size = 4
output_size = len(y[0])

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 100000

for epoch in range(epochs):
    layer_hid = tanh(np.dot(x, weight_hid))
    layer_out = tanh(np.dot(layer_hid, weight_out))
    error = (layer_out - y) ** 2

    layer_out_delta = (layer_out - y) * tanh_deriv(layer_out)
    layer_hidden_delta = layer_out_delta.dot(weight_out.T) * tanh_deriv(layer_hid)

    weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
    weight_hid -= learning_rate * x.T.dot(layer_hidden_delta)

    if epoch % 1000 == 0:
        error = np.mean(error)
        print(f"Epoch: {epoch}, Error: {error}")

new_input = np.array([[1, 0]])
layer_hid = tanh(new_input.dot(weight_hid))
layer_out = tanh(layer_hid.dot(weight_out))

print("Предсказание: ", layer_out)