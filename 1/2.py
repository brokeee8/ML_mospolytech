import numpy as np

array_1d = np.array([1, 2, 3, 4, 5])
print(array_1d)

array_multiplied = array_1d * 2
print(array_multiplied)

array_2d_1 = np.random.rand(3, 3)
array_2d_2 = np.random.rand(3, 3)
print(array_2d_1)
print(array_2d_2)

#элементное умножение
element_wise_mult = array_2d_1 * array_2d_2
print(element_wise_mult)

#матричное умножение
matrix_mult = np.dot(array_2d_1, array_2d_2)
print(matrix_mult)

random_array = np.random.randint(1, 100, 10)
print(random_array)

even_elements = random_array[random_array % 2 == 0]
print(even_elements)

even_indices = np.where(random_array % 2 == 0)[0]
even_elements_2 = random_array[even_indices]
print(even_indices)
print(even_elements_2)

mean_value = np.mean(random_array)
std_value = np.std(random_array)
max_value = np.max(random_array)
min_value = np.min(random_array)

print(f"массив: {random_array}")
print(f"ср знач: {mean_value:.2f}")
print(f"стандартное отклонение: {std_value:.2f}")
print(f"макс знач: {max_value}")
print(f"мин знач: {min_value}")


# упрощение кода с помощью numpy

def neuralNetwork2(inps, weights):
    prediction_h1 = inps.dot(weights[0])
    prediction_h2 = prediction_h1.dot(weights[1])
    prediction_h3 = prediction_h2.dot(weights[2])
    prediction_out = prediction_h3.dot(weights[3])
    return prediction_out

inp = np.array([23, 45])

np.random.seed(42)

weight_h_1 = np.random.rand(2)
weight_h_2 = np.random.rand(2)

weight_out_1 = np.random.rand(2)
weight_out_2 = np.random.rand(2)

weight_h_3 = np.random.rand(2)
weight_out_3 = np.random.rand(2)

weight_h = np.array([weight_h_1, weight_h_2]).T
weight_out = np.array([weight_out_1, weight_out_2]).T
weight_h3 = np.array([weight_h_3, weight_out_3]).T

weights = [weight_h, weight_out, weight_h3, np.array([[1.0], [1.0]])]

print(neuralNetwork2(inp, weights))

