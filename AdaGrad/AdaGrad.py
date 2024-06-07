import numpy as np

def adagrad_update(params, grads, lr=0.01, epsilon=1e-8, historical_gradients=None):
    if historical_gradients is None:
        # Initialize historical gradients
        historical_gradients = {key: np.zeros_like(value) for key, value in grads.items()}
    
    updated_params = {}
    for key in params:
        # Accumulate the square of gradients
        historical_gradients[key] += grads[key] ** 2

        # Update parameters
        adjusted_lr = lr / (np.sqrt(historical_gradients[key]) + epsilon)
        updated_params[key] = params[key] - adjusted_lr * grads[key]
    
    return updated_params, historical_gradients

# Example usage
if __name__ == "__main__":
    # Suppose we have a simple quadratic function f(x, y) = x^2 + y^2
    def compute_gradients(x, y):
        return {'x': 2 * x, 'y': 2 * y}

    # Initial parameters
    params = {'x': 3.0, 'y': 4.0}
    historical_gradients = None
    lr = 0.1

    # Perform a few iterations of optimization
    for i in range(10):
        grads = compute_gradients(params['x'], params['y'])
        params, historical_gradients = adagrad_update(params, grads, lr, historical_gradients=historical_gradients)
        print(f"Iteration {i+1}: x = {params['x']:.4f}, y = {params['y']:.4f}")
