import numpy as np

class RMSPropOptimizer:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.E_g2 = None

    def initialize(self, params):
        self.E_g2 = {k: np.zeros_like(v) for k, v in params.items()}

    def update(self, params, grads):
        if self.E_g2 is None:
            self.initialize(params)

        for k in params.keys():
            self.E_g2[k] = self.beta * self.E_g2[k] + (1 - self.beta) * grads[k]**2
            params[k] -= self.lr * grads[k] / (np.sqrt(self.E_g2[k]) + self.epsilon)

        return params

# Example usage
if __name__ == "__main__":
    # Sample parameters and gradients
    params = {'w': np.array([1.0, 2.0]), 'b': np.array([0.5])}
    grads = {'w': np.array([0.1, -0.2]), 'b': np.array([0.05])}

    # Create RMSProp optimizer
    optimizer = RMSPropOptimizer(lr=0.01)

    # Update parameters
    for i in range(100):
        params = optimizer.update(params, grads)
    print(f"Iteration {100}: {params}")