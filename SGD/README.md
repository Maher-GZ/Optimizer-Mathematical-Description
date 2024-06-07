# Stochastic Gradient Descent (SGD) Optimizer

## Overview
Stochastic Gradient Descent (SGD) is an optimization algorithm commonly used for training machine learning models. It is a variant of the gradient descent algorithm that updates the model parameters using the gradient of the loss function with respect to the parameters.

## Equation 

param =param - learning_rate * grad


## Features
- Simplicity: SGD is straightforward and easy to implement.
- Efficiency: It updates the parameters using a single example or a mini-batch of examples at each iteration, making it computationally efficient.
- Convergence: SGD can converge to a local minimum of the loss function, although it may oscillate around the minimum due to the stochastic nature of the updates.

## Usage
SGD can be used in various machine learning and deep learning frameworks, including TensorFlow, PyTorch, and scikit-learn. Here's a general guideline for using SGD optimizer:

1. Define your model architecture.
2. Choose a suitable learning rate for SGD.
3. Compile or instantiate your model with the SGD optimizer, specifying the chosen learning rate.
4. Train your model using the compiled model and your training data.

## Hyperparameters
- Learning rate: The step size used for parameter updates. It controls the magnitude of parameter updates during optimization.
- Batch size: The number of examples used in each iteration of optimization. SGD can be applied with a single example (stochastic gradient descent) or a mini-batch of examples (mini-batch gradient descent).

## References
- Sebastian Ruder, "An overview of gradient descent optimization algorithms", arXiv:1609.04747
- TensorFlow SGD optimizer documentation: [tf.keras.optimizers.SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)
- PyTorch SGD optimizer documentation: [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- scikit-learn SGD optimizer documentation: [sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)

## License
This implementation of SGD optimizer is provided under the MIT license. See the [LICENSE](LICENSE) file for details.
