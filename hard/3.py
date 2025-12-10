import numpy as np
from keras.src.datasets import mnist

def relu(x):
    return (x > 0) * x

def reluderiv(x):
    return x > 0


train_images_count = 1000
test_images_count = 10000
pixels_per_image = 28 * 28
digits_num = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = x_train[0:train_images_count].reshape(train_images_count, pixels_per_image) / 255
train_labels = y_train[0:train_images_count]


test_images = x_test[0:test_images_count].reshape(test_images_count, pixels_per_image) / 255
test_labels = y_test[0:test_images_count]

one_hot_labels = np.zeros((len(train_labels), digits_num))
for j in range(len(train_labels)):
    one_hot_labels[j][train_labels[j]] = 1
train_labels = one_hot_labels

one_hot_labels = np.zeros((len(test_labels), digits_num))
for i,j in enumerate(test_labels):
    one_hot_labels[i][j] = 1
test_labels = one_hot_labels

hidden_size = 130
weight_hid = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weight_out = 0.2 * np.random.random((hidden_size, digits_num)) - 0.1


learning_rate = 0.001

for i in range(100):
    correct_answers = 0
    for j in range(len(train_images)):
        layer_in = train_images[j:j+1]
        layer_hid = relu(layer_in.dot(weight_hid))
        layer_out = layer_hid.dot(weight_out)
        correct_answers += int(np.argmax(layer_out) == np.argmax(train_labels[j:j+1]))
        layer_out_delta = layer_out - train_labels[j:j+1]   
        layer_hid_delta = layer_out_delta.dot(weight_out.T) * reluderiv(layer_hid)
        weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
        weight_hid -= learning_rate * layer_in.T.dot(layer_hid_delta)
    print("Epoch: ", i)
    print("Accuracy: % .2f" % (correct_answers*100/len(train_images)))

correct_answers = 0
for j in range(len(test_images)):
     layer_in = test_images[j:j + 1]
     layer_hid = relu(np.dot(layer_in, weight_hid))
     layer_out = np.dot(layer_hid, weight_out)
     correct_answers += int(np.argmax(layer_out) == np.argmax(test_labels[j:j + 1]))

print("Accuracy: %.2f" % (correct_answers * 100 / len(test_images)))