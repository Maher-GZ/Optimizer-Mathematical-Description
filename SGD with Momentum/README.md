
# SGD with Momentum

This repository contains a simple implementation of Stochastic Gradient Descent (SGD) with Momentum in Python. SGD with Momentum is an optimization algorithm that helps accelerate the convergence of gradient descent by taking into account the past gradients.

## Function Description

The `sgd_momentum` function performs one update step using SGD with Momentum.

### Parameters

- `params` (dict): Dictionary containing the parameters (e.g., weights `W` and biases `b`).
- `grads` (dict): Dictionary containing the gradients of the parameters.
- `learning_rate` (float): Learning rate for the update step.
- `momentum` (float): Momentum factor (typically between 0.9 and 0.99).
- `velocity` (dict): Dictionary containing the current velocity of the parameters.

### Returns

- `updated_params` (dict): Dictionary containing the updated parameters.
- `updated_velocity` (dict): Dictionary containing the updated velocity.

## Equation 

![SGD with Momentum](SGD with Momentum/Momentum.png)
