# AdaGrad Optimizer

This repository contains a simple implementation of the AdaGrad optimizer in Python. AdaGrad is an optimization algorithm that adapts the learning rate based on the history of gradients, which helps in improving the convergence of the algorithm.

## Overview

The AdaGrad optimizer is designed to handle sparse data and varying learning rates. It adjusts the learning rate for each parameter based on the sum of the squares of the gradients previously computed for that parameter. This ensures that parameters with larger gradients get smaller updates, and parameters with smaller gradients get larger updates.

## Equation
 
adjusted_lr = lr / (np.sqrt(historical_gradients[key]) + epsilon)
updated_params[key] = params[key] - adjusted_lr * grads[key]

## Features

- **Adaptive Learning Rate**: Adjusts the learning rate based on the historical gradients, which helps in better convergence.
- **Sparse Data Handling**: Efficiently handles sparse data scenarios.
- **Simple Implementation**: Easy to understand and integrate into existing projects.

## How It Works

1. **Initialization**: Initialize the historical gradient accumulators.
2. **Gradient Accumulation**: Accumulate the square of the gradients for each parameter.
3. **Parameter Update**: Adjust the learning rate for each parameter using the accumulated gradients and update the parameters accordingly.

## Usage

To use the AdaGrad optimizer, follow these steps:

1. **Define the Gradients**: Implement a function to compute the gradients of your objective function with respect to the parameters.
2. **Initialize Parameters**: Set initial values for the parameters you want to optimize.
3. **Optimization Loop**: In each iteration, compute the gradients, update the parameters using AdaGrad, and repeat until convergence.

## Example

Suppose you have a simple quadratic function \( f(x, y) = x^2 + y^2 \). You can minimize this function using the AdaGrad optimizer by iteratively updating the parameters \( x \) and \( y \) based on their gradients.

### Steps:

1. **Compute Gradients**: Calculate the gradients of the function with respect to \( x \) and \( y \).
2. **Update Parameters**: Use the AdaGrad optimizer to update \( x \) and \( y \) based on the computed gradients.
3. **Repeat**: Continue the process for a specified number of iterations or until the parameters converge.

## Benefits

- **Stability**: Prevents the learning rate from becoming too small or too large by dynamically adjusting it.
- **Efficiency**: Suitable for high-dimensional and sparse data problems.
- **Ease of Use**: Can be easily integrated into any optimization routine with minimal changes to existing code.

## Conclusion

The AdaGrad optimizer is a powerful tool for adaptive learning rate optimization, especially in scenarios with sparse data and varying gradient magnitudes. By using this optimizer, you can achieve better and more stable convergence in your machine learning models.
