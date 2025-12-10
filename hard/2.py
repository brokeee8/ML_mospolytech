import numpy as np

# Напишите по памяти код из урока "Обучение со скрытыми слоями".
# Увеличьте или уменьшите количество скрытых нейронов в первом скрытом слое (layer_hid_1_size). Как это влияет на скорость обучения и точность предсказания?
#  Какой размер слоя приводит к наилучшим результатам?
# Измените скорость обучения (learning_rate). Попробуйте разные значения, например, 0.001 или 0.0001. Как это влияет на сходимость обучения и
#  качество предсказания?
# Измените количество эпох обучения (num_epochs). Увеличьте или уменьшите это число. Как это влияет на ошибку обучения и точность предсказания?

def relu(x):
    # print(x)
    # for i in range(len(x)):
    #     if (x[i] < 0): x[i] = 0
    # return x 
    return (x > 0) * x

def reluderif(x):
    return x > 0


inp = np.array([
                [15, 10],
                [15, 18],
                [15, 20],
                [25, 10]
                ])

true_prediction = np.array([[15, 18, 20, 25]]).T

layer_hid_size = 3
layer_in_size = len(inp[0])
layer_out_size = len(true_prediction[0])


np.random.seed(100)
weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_out = np.random.random((layer_hid_size, layer_out_size))

prediction_hid = relu(np.dot(inp[0], weights_hid))
print(prediction_hid)
prediction = prediction_hid.dot(weights_out)
print(prediction)

learning_rate = 0.0001

num_epochs = 500

for i in range(num_epochs):
    layer_out_error = 0
    for i in range(len(inp)):
        layer_in = inp[i:i+1]
        layer_hid = relu(layer_in.dot(weights_hid))
        layer_out = layer_hid.dot(weights_out)
        layer_out_error += np.sum(layer_out - true_prediction[i:i+1]) ** 2
        layer_out_delta = true_prediction[i:i+1] - layer_out
        layer_hid_delta = layer_out_delta.dot(weights_out.T) * reluderif(layer_hid)
        weights_out += learning_rate * layer_hid.T.dot(layer_out_delta)
        weights_hid += learning_rate * layer_in.T.dot(layer_hid_delta)
        print("Predictions: %s, True predictions: %s" %(layer_out, true_prediction[i:i+1]))
    print("Errors: %.4f" % layer_out_error)
    print('-------------------------------')