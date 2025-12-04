import numpy as np

#решение проблемы с расхождением

def neuralNetwork(inps, weights):
    prediction = inps * weights
    return prediction


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


inp = 0.8
weight = 0.2
true_prediction = 0.2
learning_rate = 0.5

for i in range(100):
    prediction = neuralNetwork(inp, weight)
    error = get_error(true_prediction, prediction)
    print(
        "Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error),
        "iteration",
        i + 1,
    )
    delta = (prediction - true_prediction) * inp * learning_rate  # скорость обучения
    weight -= delta
