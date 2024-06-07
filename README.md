# Optimizer-Mathematical-Description

This repository contains implementations of several popular optimization algorithms used in machine learning and deep learning. Each optimizer is presented in its own subfolder, including a mathematical description and a simple implementation function. The following optimizers are included:

- AdaGrad
- Adam
- SGD (Stochastic Gradient Descent)
- SGD with Momentum
- RMSprop

## Repository Structure

The repository is structured as follows:
 
### AdaGrad

**Description:** The AdaGrad algorithm adjusts the learning rate for each parameter individually, scaling it inversely proportional to the square root of the sum of all historical squared values of the gradient.

- Mathematical Description: See `AdaGrad/description.md`
- Implementation: See `AdaGrad/adagrad.py`

### Adam

**Description:** The Adam optimizer combines the advantages of two other extensions of stochastic gradient descent. Specifically, it keeps an exponentially decaying average of past gradients (like momentum) and past squared gradients (like RMSprop).

- Mathematical Description: See `Adam/description.md`
- Implementation: See `Adam/adam.py`

### SGD (Stochastic Gradient Descent)

**Description:** SGD updates the parameters by following the negative gradient of the loss function with respect to the parameter. It is a basic but powerful optimizer.

- Mathematical Description: See `SGD/description.md`
- Implementation: See `SGD/sgd.py`

### SGD with Momentum

**Description:** SGD with Momentum helps accelerate SGD in the relevant direction and dampens oscillations. It adds a fraction of the update vector of the past time step to the current update vector.

- Mathematical Description: See `SGD_Momentum/description.md`
- Implementation: See `SGD_Momentum/sgd_momentum.py`

### RMSprop

**Description:** RMSprop is an unpublished, adaptive learning rate method. It keeps a moving average of the squared gradient for each parameter and divides the gradient by the root of this moving average.

- Mathematical Description: See `RMSprop/description.md`
- Implementation: See `RMSprop/rmsprop.py`

## How to Use

Each subfolder contains a `README.md` file with specific instructions on how to run the implementation code and understand the mathematical description provided. To use a particular optimizer, navigate to its respective folder and follow the instructions provided.

## Contributing

Contributions are welcome! If you would like to contribute, please fork the repository and submit a pull request. Ensure that your contributions are well-documented and tested.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

If you have any questions or suggestions, please feel free to contact the repository maintainer.

---

Happy optimizing!


