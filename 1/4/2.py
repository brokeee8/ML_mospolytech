import numpy as np


# задаем нейронную сеть с несколькими входами
def neural_networks(inp, weights):
    return inp.dot(weights)


# реализуем функцию, возвращающую ошибку прогнозирования
def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


# Объявим два входа и один выход, а также ожидаемый прогноз и скорость обучения
inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_prediction = 1
learning_rate = 0.00001  # подбираем необходимую скорость обучения

# Приступаем к обучению
for i in range(1000):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weights, error), 'iteration', i+1)
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta
    