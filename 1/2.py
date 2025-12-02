import numpy as np


# def neuralNetwork(inp, weights):
#     prediction_h = inp.dot(weights[0])
#     prediction_out = prediction_h.dot(weights[1])
#     return prediction_out


# inp = np.array([23, 45])

# weight_h_1 = [0.4, 0.1]
# weight_h_2 = [0.3, 0.2]

# weight_out_1 = [0.4, 0.1]
# weight_out_2 = [0.3, 0.1]

# weights_h = np.array([weight_h_1, weight_h_2]).T
# weights_out = np.array([weight_out_1, weight_out_2]).T

# weights = [weights_h, weights_out]

# # print(neuralNnetwork(inp, weights))

# # ------- задание 1
# mass = np.array([1, 2, 3, 4, 5])
# # print(mass * 2)

# arr1 = np.random.rand(3,3)
# arr2 = np.random.rand(3,3)
# # print(arr1)
# # print(arr2)
# # print(arr1.dot(arr2))

# arr3 = np.random.randint(0, 101, size=10)
# print(arr3)
# print(arr3[::2])
# print(np.mean(arr3))
# print(np.std(arr3))
# print(np.max(arr3))
# print(np.min(arr3))

# Упрощение кода с помощью NumPy

# import numpy as np

# def neuralNetwork(inp, weights):
#     prediction_h_1 = inp.dot(weights[0])
#     prediction_h_2 = prediction_h_1.dot(weights[1])
#     prediction_out = prediction_h_2.dot(weights[2])  
#     return prediction_out

# inp = np.array([23, 45])

# # первый скрытый слой
# weight_h_1 = [0.4, 0.1]
# weight_h_2 = [0.3, 0.2]
# weight_h_3 = [0.6, 0.2]

# # второй скрытый слой
# weight_mid_1 = [0.4, 0.1, 0.5]
# weight_mid_2 = [0.3, 0.1, 0.6]
# weight_mid_3 = [0.7, 0.4, 0.3]

# # выходной слой
# weight_out_1 = [0.4, 0.1, 0.5]
# weight_out_2 = [0.3, 0.1, 0.6]

# weights_h = np.array([weight_h_1, weight_h_2, weight_h_3]).T
# weights_mid = np.array([weight_mid_1, weight_mid_2, weight_mid_3]).T
# weights_out = np.array([weight_out_1, weight_out_2]).T

# weights = [weights_h, weights_mid, weights_out] 
# print(neuralNetwork(inp, weights))

import numpy as np

def neuralNetwork(inp, weights):
    prediction_h_1 = inp.dot(weights[0])  
    prediction_h_2 = prediction_h_1.dot(weights[1])  
    prediction_out = prediction_h_2.dot(weights[2]) 
    return prediction_out

# Входной слой: 2 нейрона
inp = np.array([23, 45])

# первый скрытый слой
weight_h_1 = np.random.rand(2)  
weight_h_2 = np.random.rand(2)  

# второй скрытый слой
weight_mid_1 = np.random.rand(2) 
weight_mid_2 = np.random.rand(2) 

# выходной слой
weight_out_1 = np.random.rand(2) 
weight_out_2 = np.random.rand(2) 

# Формируем матрицы весов:
weights_h = np.array([weight_h_1, weight_h_2]).T          
weights_mid = np.array([weight_mid_1, weight_mid_2]).T    
weights_out = np.array([weight_out_1, weight_out_2]).T    

weights = [weights_h, weights_mid, weights_out]

print(neuralNetwork(inp, weights))