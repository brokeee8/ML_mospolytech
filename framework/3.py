import numpy as np

#
# Tensor
#
class Tensor(object):
    
    id_count = 0

    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.init(data, creators, operation_on_creation, autograd, id)

    def init(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None
        self.autograd = autograd
        self.children = {}
        
        if id is None:
            self.__class__.id_count += 1
            self.id = self.__class__.id_count
            
        if self.creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:    
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1
                            
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
                if self.operation_on_creation == "+":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                elif self.operation_on_creation == "-1":
                    self.creators[0].backward(self.grad.__neg__(), self)
                    
                elif self.operation_on_creation == "-":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)
                    
                elif self.operation_on_creation == "*":
                    new_grad = self.grad * self.creators[1]
                    self.creators[0].backward(new_grad, self)
                    new_grad = self.grad * self.creators[0]
                    self.creators[1].backward(new_grad, self)
                    
                elif "sum" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
                    
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.sum(axis), self)
                    
                elif self.operation_on_creation == "transpose":
                    self.creators[0].backward(self.grad.transpose(), self)
                    
                elif self.operation_on_creation == "dot":
                    temp = self.grad.dot(self.creators[1].transpose())
                    self.creators[0].backward(temp, self)
                    temp = self.creators[0].transpose().dot(self.grad)
                    self.creators[1].backward(temp, self)

                elif self.operation_on_creation == "sigmoid":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self*(temp - self)), self)

                elif self.operation_on_creation == "tanh":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (temp - self * self), self)

                elif self.operation_on_creation == "relu":
                    temp = self.grad * Tensor((self.creators[0].data > 0) * 1.0)
                    self.creators[0].backward(temp, self)
                                    
    def check_grads_from_children(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True
            
    def add(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __add__(self, other):
        return self.add(other)
    
    def __str__(self):
        return str(self.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self,other], "-", True)
        return Tensor(self.data - other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * (-1), [self], "-1", True)
        return Tensor(self.data * (-1))

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))
    
    def expand(self, axis, count_copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        expand_shape = list(self.data.shape) + [count_copies]
        expand_data = (self.data.repeat(count_copies).reshape(expand_shape))
        expand_data = expand_data.transpose(transpose)
        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)
        return Tensor(expand_data)

    def dot(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        return Tensor(self.data.transpose())

    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)), [self], "sigmoid",True)
        return Tensor(1/(1+np.exp(-self.data)))
    
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh",True)
        return Tensor(np.tanh(self.data))

    def relu(self):  
        if self.autograd:
            return Tensor((self.data > 0) * self.data, [self], "relu",True)
        return (self.data > 0) * self.data
    
    def __repr__(self):
        return str(self.data.__repr__())



class SGD(object):
    def __init__(self, weights, learning_rate = 0.01):
        self.weights = weights
        self.learning_rate = learning_rate

    def step(self):
        for weight in self.weights:
            weight.data -= self.learning_rate * weight.grad.data
            weight.grad.data *=0


# a_1 = Tensor([[1,2,3], [4,5,6]], autograd=True)
# a_2 = Tensor([[2,3],[2,3],[2,3]], autograd=True)
# print(a_2)
# print(a_1)

# print(a_1.transpose())
# print(a_2.transpose())

# a_3 = a_1.dot(a_2)
# print(a_3)

# a_3.backward(Tensor([1,4]))
# print(a_1.grad)

# print("---------------")

# a_1 = Tensor([[1,2,3], [4,5,6]], autograd=True)
# a_3 = a_1.sigmoind()
# a_3.backward(Tensor([4,5,10]))
# print(a_3)
# print(a_1.grad)

# print("---------------")

# a_2 = Tensor([[2,3,4],[2,3,5]], autograd=True)
# a_4 = a_2.tanh()
# a_4.backward(Tensor([4,5,10]))

# print(a_3)
# print(a_2.grad)

# np.random.seed(0)
# inp = Tensor([[2,3],[5,10]], autograd=True)
# true_predictions = Tensor([[5],[15]], autograd=True)

# weights = [
# Tensor(np.random.rand(2,2), autograd=True),
# Tensor(np.random.rand(2,1), autograd=True),
# Tensor(np.random.rand(1,1), autograd=True)
# ]
# sgd = SGD(weights, 0.001)
# num_epochs = 10

# for i in range(num_epochs):
#     prediction = inp.dot(weights[0]).dot(weights[1]).dot(weights[2])
#     error = (prediction - true_predictions) * (prediction - true_predictions)

#     error.backward(Tensor(np.ones_like(error.data)))
#     sgd.step()

#     print("Error: ", error)

# print(weights)




# relu

a = Tensor([[2,3,4],[2,3,5]], autograd=True)
a2 = a.relu()
a2.backward(Tensor([4,5,10]))
print(a2.grad)

np.random.seed(0)

weights = [
    Tensor(np.random.randn(3, 3), autograd=True),
    Tensor(np.random.randn(3, 3), autograd=True),
    Tensor(np.random.randn(3, 1), autograd=True)
]

sgd = SGD(weights, 0.01)

train_data = [
    ([1, 4, 5], 20),
    ([1, 5, 5], 25),
    ([1, 3, 10], 30),
    ([1, 4, 8], 32),
    ([1, 5, 8], 40),
    ([1, 5, 9], 45),
    ([1, 6, 8], 48),
    ([1, 7, 7], 49),
    ([1, 5, 10], 50),
    ([1, 6, 9], 54),
    ([1, 7, 8], 56),
    ([1, 8, 8], 64),
    ([1, 7, 10], 70),
    ([1, 8, 9], 72),
    ([1, 9, 9], 81),
    ([2, 5, 5], 50),
    ([2, 4, 7], 56),
    ([2, 5, 7], 70),
    ([2, 6, 7], 84),
    ([2, 4, 11], 88),
    ([2, 5, 9], 90),
    ([2, 6, 8], 96),
    ([2, 5, 10], 100),
    ([3, 3, 3], 27),
    ([3, 3, 4], 36),
    ([3, 3, 5], 45),
    ([3, 4, 7], 84),
    ([3, 5, 4], 60)
]

for epoch in range(1000):
    for inputs, target in train_data:

        inp = Tensor([inputs], autograd=True)
        true_predictions = Tensor([[target]], autograd=True)

        prediction = inp.dot(weights[0]).sigmoid().dot(weights[1]).sigmoid().dot(weights[2])
        
        error = (prediction - true_predictions) * (prediction - true_predictions)
        
        error.backward()
        sgd.step()

    if epoch % 20 == 0:
        print("Epoch", epoch, "error =", error.data)


test_inp = Tensor([[3,5,4]], autograd=False)
test_pred = test_inp.dot(weights[0]).sigmoid().dot(weights[1]).sigmoid().dot(weights[2])
print("Prediction for (3,5,4):", test_pred.data)