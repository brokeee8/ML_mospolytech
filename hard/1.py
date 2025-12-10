import numpy as np

# Напишите по памяти код из урока "Скрытые слои и функция ReLu".
# Увеличьте количество нейронов в скрытом слое.
# Добавьте ещё один скрытый слой. Его значения надо также пропустить через функцию ReLU. Проанализируйте результат.

def neural_network(inputs, weights):
    return inputs.dot(weights)

def get_error(true_value, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))

def relu(x):
    return (x > 0) * x

inp = np.array([
[15, 10],
[15, 15],
[15, 20],
[25, 10]
])
# true_prediction = np.array([[10, 20, 15, 20]]).T
true_prediction = np.array([10, 20, 15, 20])

layer_hid_size = 3
layer_hid_size_2 = 3
layer_in_size = len(inp[0])
# layer_out_size = len(true_prediction[0])
layer_out_size = 1

weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_hid_2 = 2 * np.random.random((layer_hid_size, layer_hid_size_2)) - 1
weights_out = 2 * np.random.random((layer_hid_size_2, layer_out_size)) - 1
# print(weights_hid)
# print(weights_out)

prediction_hid = relu(np.dot(inp[0], weights_hid))
print(prediction_hid)
prediction_hid_2 = relu(np.dot(prediction_hid, weights_hid_2))
print(prediction_hid_2)
prediction = prediction_hid_2.dot(weights_out)
print(prediction)