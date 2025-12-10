import numpy as np


class Tensor(object):
    count_id = 0  
    
    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None
        self.autograd = autograd
        self.children = {}
        
        if id is None:
            Tensor.count_id += 1  
            self.id = Tensor.count_id
        else:
            self.id = id
            
        if self.creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1
                
                print(f"сreator {self.id}")
    
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:    
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if grad_origin.id in self.children and self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
        if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
            if self.operation_on_creation == "+":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)
    
    def check_grads_from_children(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True
    
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)
    
    def __str__(self):
        return str(self.data)


a_1 = Tensor([1, 2, 3], autograd=True)
a_2 = Tensor([1, 2, 3], autograd=True)
a_3 = Tensor([1, 2, 3], autograd=True)

a_add_1 = a_1 + a_2
a_add_2 = a_2 + a_3
a_add_3 = a_add_1 + a_add_2

a_add_3.backward(Tensor([4, 5, 3]))

print(f"Градиент a_1: {a_1.grad}")
print(f"Градиент a_2: {a_2.grad}")
print(f"Градиент a_3: {a_3.grad}")