import numpy as np

def sgd_optimizer(params, grads, learning_rate=0.01):
    updated_params = []
    for param, grad in zip(params, grads):
        param -= learning_rate * grad
        updated_params.append(param)
    return updated_params



params = [np.random.randn(3, 3), np.random.randn(3, 1)] 
grads = [np.random.randn(3, 3), np.random.randn(3, 1)]  

updated_params = sgd_optimizer(params, grads)

print("Updated Parameters:")
for param in updated_params:
    print(param)
