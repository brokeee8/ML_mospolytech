import numpy as np


# реализуем простейшую нейронную сеть с одним входным и выходным нейронами
def neural_networks(inp, weight):
    return inp * weight


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


inp = 0.9  # 0.9509 близкое чтобы было 0.0001
weight = 0.2

true_prediction = 0.8
print(get_error(true_prediction, neural_networks(inp, weight)))
print(neural_networks(inp, weight))


for i in range(10):
    prediction = neural_networks(inp, weight)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error))

    delta = (prediction - true_prediction) * inp
    print(delta)
    weight = weight - delta
    if error == 0.00000000000000000000:
        break

# #2 номер: при изменении входных данных меняется то, столько итераций должны быть, чтобы error стало равно нулю
# например при inp = 0.5 и weight = 0.2 нужно 80 итераций
# #3 номер: при изменении true_prediction меняется error И delta, а впоследствии weight
# #4 номер: если говорить про заданные значение, то 10 итераций является оптимальным значением
