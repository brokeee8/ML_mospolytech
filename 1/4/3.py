import numpy as np

#задаем нейронную сеть с одним входом и несколькими выходами
def neural_networks(inp, weights):
    return inp * weights
#реализуем функцию, возвращающую ошибку прогнозирования
def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2
#Объявим один вход и два выхода, а также ожидаемый прогноз и скорость обучения
inp = 50
weights = np.array([0.2,0.3])
true_predictions = np.array([50,120])
learning_rate = 0.0001

#Приступаем к обучению
for i in range(30):
    prediction = neural_networks(inp, weights)
    error = get_error(true_predictions, prediction)
    print("Prediction: %s, Weights: %s, Error: %s" %(prediction, weights, error), 'iteration', i+1)
    delta = (prediction - true_predictions) * inp * learning_rate
    weights = weights - delta