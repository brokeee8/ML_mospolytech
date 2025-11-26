# def neural_network(inp, weight):
#     prediction = inp * weight
#     return prediction

# inputs = [150, 160, 170, 180, 190])
# for i in range(len(inputs)):
#     print(neural_network(inputs[i]))

# print(out_1)

# #На базе кода из урока "Создание простейшей нейросети".
# #1.1
# #Измените входные данные и вес нейросети в коде. Запустите программу с новыми значениями и опишите, 
# #как это повлияло на выходные данные. 
# #Объясните, почему это произошло с точки зрения работы нейронной сети
 
# def neural_network(inp, weight):
#     prediction = inp * weight
#     return prediction

# weight = 0.3  
# inp = 200     

# pred = neural_network(inp, weight)
# print(inp, weight, pred)

# #1.2
# #Создайте список входных данных (например, inputs = [150, 160, 170, 180, 190]) 
# #и используйте цикл for для вычисления выходных данных нейросети для каждого значения в списке. 
# #Распечатайте выходные данные для каждого входного значения.

# def neural_network(inp, weight):
#     prediction = inp * weight
#     return prediction

# inputs = [150, 160, 170, 180, 190]
# weight = 0.1

# for i in range(len(inputs)):
#     result = neural_network(inputs[i], weight)
#     print(inputs[i], result)

# #1.3
# #Модифицируйте функцию neural_network так, чтобы она принимала два входных параметра: inp и bias. 
# # Результат будет задан как inp * weight + bias. 
# # Запустите функцию с новыми значениями inp, weight и bias. Как изменится выходная переменная? Почему?

# def neural_network(inp, weight, bias):
#     prediction = inp * weight + bias
#     return prediction

# weight = 0.1
# bias = 5.0
# inp = 150

# result = neural_network(inp, weight, bias)
# print(inp, weight, bias, result)

# result_old = inp * weight
# print(inp, weight, result_old)


# #На базе кода из урока "Создание нейросети с несколькими входами".

# def neuralNetwork(inps, weights):
#     prediction = 0
#     for i in range(len(weights)):
#         prediction += inps[i]*weights[i]
#     return prediction

# out_1 = neuralNetwork([150, 40], [0.3, 0.4])
# out_2 = neuralNetwork([80, 60], [0.2, 0.4])

# print(out_1)
# print(out_2)

# #2.1
# # Измените функцию neural_network так, чтобы она возвращала не только предсказание (prediction), 
# # но и весь список промежуточных значений (произведение каждого элемента входных данных на соответствующий вес). 
# # Выведите обе переменные (prediction и список промежуточных значений) на экран.

# def neuralNetwork(inps, weights):
#     prediction = 0
#     intermediate_values = []  
    
#     for i in range(len(weights)):
#         intermediate = inps[i] * weights[i] 
#         intermediate_values.append(intermediate)
#         prediction += intermediate
    
#     return prediction, intermediate_values

# out_1, intermediates_1 = neuralNetwork([150, 40], [0.3, 0.4])
# out_2, intermediates_2 = neuralNetwork([80, 60], [0.2, 0.4])

# print(out_1,intermediates_1)
# print(f"Промежуточные значения: {intermediates_1}")
# print(f"Проверка: {intermediates_1[0]} + {intermediates_1[1]} = {intermediates_1[0] + intermediates_1[1]}")

# print(out_1,intermediates_2)
# print(f"Промежуточные значения: {intermediates_2}")
# print(f"Проверка: {intermediates_2[0]} + {intermediates_2[1]} = {intermediates_2[0] + intermediates_2[1]}")



# #На базе кода из урока "Создание нейросети с несколькими выходами".

# # 3.1 Измените веса нейросети в коде и определите, при каких значениях весов выходные данные для каждого элемента 
# # становятся больше 0.5. 
# # Решите это методом проб и ошибок, меняя веса с небольшими шагами.

# def neuralNetwork(inp, weights):
#     prediction = [0, 0]
#     for i in range(len(weights)):
#         prediction[i] = inp * weights[i]
#     return prediction

# weights = [0.2, 0.5]
# result = neuralNetwork(4, weights)
# print(f"Исходные веса: {weights}")
# print(f"Выход: {result}")

# test_weights = [
#     [0.13, 0.13],  # 4*0.13=0.52, 4*0.13=0.52 - оба > 0.5
#     [0.2, 0.2],    # 4*0.2=0.8, 4*0.2=0.8 - оба > 0.5
#     [0.1, 0.2],    # 4*0.1=0.4, 4*0.2=0.8 - только второй > 0.5
#     [0.15, 0.15]   # 4*0.15=0.6, 4*0.15=0.6 - оба > 0.5
# ]

# print("\nМетод проб и ошибок:")
# for w in test_weights:
#     result = neuralNetwork(4, w)
#     status = "ОБА > 0.5" if result[0] > 0.5 and result[1] > 0.5 else "не все > 0.5"
#     print(f"Веса: {w}, Выход: {result}, {status}")


# #3.2 Напишите код с циклом, где значение веса будет увеличиваться до тех пор, пока выходное значение меньше 0.5. 
# # Как только один выход стал больше 0.5, то изменение его веса останавливается. 
# # Как только второй выход стал больше 0.5, то изменение его веса также останавливается, а цикл завершается. Выведите получившиеся веса.

# def neuralNetwork(inp, weights):
#     prediction = [0, 0]
#     for i in range(len(weights)):
#         prediction[i] = inp * weights[i]
#     return prediction

# weights = [0.01, 0.01]
# inp = 4
# step = 0.01  # Шаг увеличения веса
# max_iterations = 1000

# target_reached = [False, False]

# print("Автоматический подбор весов:")
# print(f"Начальные веса: {weights}")

# for iteration in range(max_iterations):
#     result = neuralNetwork(inp, weights)
#     for i in range(len(weights)):
#         if not target_reached[i] and result[i] <= 0.5:
#             weights[i] += step
#         elif result[i] > 0.5 and not target_reached[i]:
#             target_reached[i] = True
#             print(f"Выход {i+1} достиг цели при весе {weights[i]:.3f}")
    
#     if all(target_reached):
#         final_result = neuralNetwork(inp, weights)
#         print(f"\nЦикл завершен на итерации {iteration + 1}")
#         print(f"Финальные веса: [{weights[0]:.3f}, {weights[1]:.3f}]")
#         print(f"Финальный выход: {[final_result[0]:.3f}, {final_result[1]:.3f}]")
#         break
# else:
#     print("Достигнуто максимальное количество итераций")

# print("\nПроверка:")
# print(f"Вход: {inp}")
# print(f"Выход 1: {inp} * {weights[0]:.3f} = {inp * weights[0]:.3f}")
# print(f"Выход 2: {inp} * {weights[1]:.3f} = {inp * weights[1]:.3f}")


# #На базе кода из урока "Создание нейросети с несколькими входами и выходами".

# #4.1 Добавьте еще один набор весов (weights_4 = [0.4, 0.2, 0.1]) и добавьте его в список weights. 
# #  Запустите функцию с этим новым набором весов. Как это повлияло на предсказанные значения? Объясните, почему.
# def neuralNetwork(inp, weights):
#     prediction = [0, 0]
#     for i in range(len(weights)):
#         ws = 0  
#         for j in range(len(inp)):
#             ws += inp[j] * weights[i][j]
#         prediction[i] = ws
#     return prediction

# inp = [50, 165]

# weights_1 = [0.2, 0.1]
# weights_2 = [0.3, 0.1]
# weights_3 = [0.4, 0.2]

# weights = [weights_1, weights_2, weights_3]

# print("Исходные веса:")
# print(f"weights_1: {weights_1}")
# print(f"weights_2: {weights_2}")
# print(f"weights_3: {weights_3}")

# result = neuralNetwork(inp, weights)
# print(f"\nВходные данные: {inp}")
# print(f"Выходные данные: {result}")

# print("\nАнализ влияния нового набора весов:")
# print(f"Выход 1 (weights_1): {inp[0]}*{weights_1[0]} + {inp[1]}*{weights_1[1]} = {result[0]}")
# print(f"Выход 2 (weights_2): {inp[0]}*{weights_2[0]} + {inp[1]}*{weights_2[1]} = {result[1]}")
# print(f"Выход 3 (weights_3): {inp[0]}*{weights_3[0]} + {inp[1]}*{weights_3[1]} = {result[2]}")

# #4.2 Измените веса нейросети таким образом, чтобы выходные данные для первого и второго нейрона стали равными. 
# # Используйте метод проб и ошибок. Входные значения менять нельзя.

# def neuralNetwork(inp, weights):
#     prediction = [0, 0]
#     for i in range(len(weights)):
#         ws = 0
#         for j in range(len(inp)):
#             ws += inp[j] * weights[i][j]
#         prediction[i] = ws
#     return prediction

# inp = [50, 165]

# test_weights = [
#     [[0.25, 0.1], [0.25, 0.1]], 
#     [[0.3, 0.08], [0.2, 0.12]],  
#     [[0.28, 0.09], [0.22, 0.11]] 
# ]

# print("Метод проб и ошибок для равных выходов:")
# for i, w in enumerate(test_weights):
#     result = neuralNetwork(inp, w)
#     print(f"Тест {i+1}: Веса {w} -> Выход: {result}, Равны: {result[0] == result[1]}")
    
#     calc1 = inp[0]*w[0][0] + inp[1]*w[0][1]
#     calc2 = inp[0]*w[1][0] + inp[1]*w[1][1]
#     print(f"  Нейрон 1: {inp[0]}*{w[0][0]} + {inp[1]}*{w[0][1]} = {calc1}")
#     print(f"  Нейрон 2: {inp[0]}*{w[1][0]} + {inp[1]}*{w[1][1]} = {calc2}")
#     print()

# #4.3 Выполните предыдущее задание, но с помощью цикла. После цикла выведите получившиеся веса.

# def neuralNetwork(inp, weights):
#     prediction = [0, 0]
#     for i in range(len(weights)):
#         ws = 0
#         for j in range(len(inp)):
#             ws += inp[j] * weights[i][j]
#         prediction[i] = ws
#     return prediction

# inp = [50, 165]

# weights = [[0.2, 0.1], [0.3, 0.1]]
# step = 0.001  # Малый шаг для точности
# max_iterations = 10000

# print("Автоматический подбор весов для равных выходов:")
# print(f"Начальные веса: {weights}")
# print(f"Начальный выход: {neuralNetwork(inp, weights)}")

# for iteration in range(max_iterations):
#     result = neuralNetwork(inp, weights)
    
#     diff = result[0] - result[1]
    
#     if abs(diff) < 0.001:
#         print(f"\nНайдено решение на итерации {iteration}:")
#         print(f"Финальные веса: {[[round(weights[0][0], 3), round(weights[0][1], 3)], [round(weights[1][0], 3), round(weights[1][1], 3)]]}")
#         print(f"Финальный выход: {[round(result[0], 3), round(result[1], 3)]}")
#         break
    
#     if diff > 0:  # Первый выход больше второго
#         weights[0][0] -= step  # Уменьшаем вес первого входа для первого нейрона
#         weights[0][1] -= step
#         weights[1][0] += step  # Увеличиваем вес первого входа для второго нейрона
#         weights[1][1] += step
#     else:  # Второй выход больше первого
#         weights[0][0] += step  # Увеличиваем вес первого входа для первого нейрона
#         weights[0][1] += step
#         weights[1][0] -= step  # Уменьшаем вес первого входа для второго нейрона
#         weights[1][1] -= step

# else:
#     print("Достигнуто максимальное количество итераций")

# final_result = neuralNetwork(inp, weights)
# print(f"\nПроверка:")
# print(f"Нейрон 1: {inp[0]}*{weights[0][0]:.3f} + {inp[1]}*{weights[0][1]:.3f} = {final_result[0]:.3f}")
# print(f"Нейрон 2: {inp[0]}*{weights[1][0]:.3f} + {inp[1]}*{weights[1][1]:.3f} = {final_result[1]:.3f}")
# print(f"Разница: {abs(final_result[0] - final_result[1]):.6f}")

# #На базе кода из урока "Добавление скрытого слоя".

#5.1 Измените веса нейросети так, чтобы предсказанные значения для второго слоя (prediction_h) стали больше 5. 
# Напишите код, который это сделает. Выведите получившиеся веса. Само собой, входные данные менять нельзя.

def neuralNetwork(inps, weights):
    prediction_h = [0] * len(weights[0])
    for i in range(len(weights[0])):
        ws = 0
        for j in range(len(inps)):
            ws += inps[j] * weights[0][i][j]
        prediction_h[i] = ws
    return prediction_h

inp = [23, 45]
weight_h_1 = [0.03, 0.1]
weight_h_2 = [0.2, 0.04]

weight_out_1 = [0.4, 0.1]
weight_out_2 = [0.3, 0.1]

weight_h = [weight_h_1, weight_h_2]
weight_out = [weight_out_1, weight_out_2]

weights = [weight_h, weight_out]

step = 0.001
max_iter = 1000

for iteration in range(max_iter):
    prediction_h = neuralNetwork(inp, weights)
    if all(x > 5 for x in prediction_h):
        break
    for i in range(len(weights[0])):
        for j in range(len(weights[0][i])):
            weights[0][i][j] += step
print([round(w,3) for w in weights[0][0]])
print([round(w,3) for w in weights[0][1]])

final_prediction_h = neuralNetwork(inp,weights)
print([x for x in final_prediction_h])