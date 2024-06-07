import numpy as np

def sgd_momentum(params, grads, learning_rate, momentum, velocity):
    """
    Perform one update step using SGD with momentum.
    
    Arguments:
    params -- dictionary containing the parameters (weights and biases)
    grads -- dictionary containing the gradients of the parameters
    learning_rate -- learning rate for the update
    momentum -- momentum factor (usually between 0.9 and 0.99)
    velocity -- dictionary containing the current velocity of the parameters
    
    Returns:
    updated_params -- dictionary containing the updated parameters
    updated_velocity -- dictionary containing the updated velocity
    """
    updated_params = {}
    updated_velocity = {}

    for key in params.keys():
        # Update the velocity
        updated_velocity[key] = momentum * velocity[key] - learning_rate * grads[key]
        
        # Update the parameters
        updated_params[key] = params[key] + updated_velocity[key]

    return updated_params, updated_velocity

# Example usage:
# Initialize parameters, gradients, velocity, learning rate, and momentum
params = {'W': np.array([1.0, 2.0]), 'b': np.array([0.5])}
grads = {'W': np.array([0.1, 0.2]), 'b': np.array([0.05])}
velocity = {'W': np.zeros_like(params['W']), 'b': np.zeros_like(params['b'])}
learning_rate = 0.01
momentum = 0.9

# Perform an update
updated_params, updated_velocity = sgd_momentum(params, grads, learning_rate, momentum, velocity)

print("Updated parameters:", updated_params)
print("Updated velocity:", updated_velocity)
