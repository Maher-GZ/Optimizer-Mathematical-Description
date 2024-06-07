import numpy as np

def adam_optimizer(params, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for i, (param, grad) in enumerate(zip(params, grads)):
        t += 1
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)
        param -= learning_rate * m_hat / (np.sqrt(v_hat + epsilon))

    return params

# Example usage:

params = [np.random.randn(3, 3), np.random.randn(3, 1)]   
grads = [np.random.randn(3, 3), np.random.randn(3, 1)]  

updated_params = adam_optimizer(params, grads)

print("Updated Parameters:")
for param in updated_params:
    print(param)
