
# RMSProp Optimizer

RMSProp (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm designed to address the diminishing learning rates issue. It helps to maintain a moving average of the squared gradients, which helps to normalize the gradient and maintain a consistent learning rate.

## How RMSProp Works


The RMSProp optimizer works as follows:

theta = theta - (alpha * grad) / (np.sqrt(E_g2) + epsilon)

### Key Steps

1. **Gradient Calculation**: Compute the gradient \( g_t \) of the loss function with respect to the parameter at the current time step.
2. **Exponential Moving Average**: Update the moving average of the squared gradients \( E[g^2]_t \) using a decay rate \( \beta \) and the current gradient.
3. **Parameter Update**: Adjust the parameter \( \theta_t \) using the computed gradient, the learning rate \( \alpha \), and the updated moving average of the squared gradients.

This approach helps to maintain a consistent learning rate by adapting to the magnitude of the gradients, ensuring that the optimizer converges smoothly and efficiently.


## Description

RMSProp adjusts the learning rate for each parameter by dividing the learning rate by an exponentially decaying average of the squared gradients. This helps to prevent large updates and provides a more controlled and stable convergence. The algorithm keeps track of a moving average of the squared gradients and uses it to scale the gradient updates.
