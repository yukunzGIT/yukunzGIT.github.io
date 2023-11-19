"""
Filename: multi_class_ann.py
Author: Yukun (Edward) Zhang
Email: ykzhang1211@g.ucla.edu
Date Created: 2017-11-15
Last Updated: 2023-08-01
Description: Implementation of multi-hidden-layer neural network from scratch using ONLY the basis NumPy and SciPy for multi-class classification and regression. 
"""

# Import the basic libraries scipy and numpy for our task.
from scipy.optimize import minimize
import numpy as np 

# Helper functions to transform between one big vector of weights,
# and a list of layer parameters of the form (W,b).
def flatten_weights(weights):
    """
    Flatten a list of weight matrices and biases into a single 1D array.
    
    Parameters
    ----------
    weights: list of tuples
        Each tuple consists of weight matrix and bias.
    
    Returns
    -------
    numpy.ndarray
        A 1D array containing all the flattened weights and biases concatenated.

    Example
    -------
    >>> W1 = np.array([[0.6, 0.9]])
        b1 = np.array([0.4])
    >>> flatten_weights([(W1, b1)])
    array([0.6, 0.9, 0.4])
    """

    # `sum(weights, ())` is a trick to flatten the outer list. 
    #  summing an empty tuple `()`, ensures the function doesn't attempt to sum the matrices and biases, but just concatenate the lists.
    return np.concatenate([w.flatten() for w in sum(weights, ())])  # list comprehension

def unflatten_weights(weights_flat, layer_sizes):
    """
    Transform a flattened array of neural network parameters back to its structured form of weight matrices and bias vectors

    Parameters
    ----------
    weights_flat : numpy.ndarray
        A 1D array containing all the flattened weights and biases.
    
    layer_sizes : list of int
        A list representing the number of nodes in each layer, 
        starting from the input layer and ending with the output layer.

    Returns
    -------
    list of tuple
        Each tuple contains a weight matrix and a bias vector for a layer. 

    Example
    -------
    If you want to unflatten weights for a neural network with an input layer of 2 nodes, 
    a hidden layer of 2 nodes, and an output layer of 1 nodes:
    >>> flat_weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> layer_sizes = [2, 2, 1]
    >>> unflatten_weights(flat_weights, layer_sizes)
    [(array([[1., 2.], [3., 4.]]), array([[5., 6.]])), (array([[7.], [8.]]), array([[9.]]))]
    """

    # Initialize an empty list to store the reshaped weights and biases for each layer.
    weights = list()
    
    # Initialize a counter to keep track of our position in the flattened weights array.
    counter = 0

    # Iterate through each layer. (excluding the last one because we don't need 
    # weights going out of the output layer).
    for i in range(len(layer_sizes) - 1):
        # Calculate the size (number of elements) of the weight matrix for the current layer.
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        # Calculate the size (number of elements) of the bias matrix for the current layer.
        b_size = layer_sizes[i + 1]

        # Reshape a slice of the flattened array to form the weight matrix for the current layer.
        W = np.reshape(
            weights_flat[counter : counter + W_size],
            (layer_sizes[i + 1], layer_sizes[i]),
        )
        # Update the counter by adding the size of the weight matrix.
        counter += W_size

        # Extract the bias vector for the current layer from the flattened array.
        b = weights_flat[counter : counter + b_size][None]
        # Update the counter by adding the size of the bias vector.
        counter += b_size

        # Append the weight matrix and bias vector as a tuple to the structured weights list.
        weights.append((W, b))
    return weights

def log_sum_exp(Z):
    """
    Compute the logarithm of the sum of exponentials for each row of Z to avoid numerical overflow or underflow issues.
    This function avoids numerical instability issues by using a trick involving the max value of Z.

    Parameters
    ----------
    Z : numpy.ndarray
        A 2D numpy array where each row corresponds to a data point 
        and each column corresponds to a class.

    Returns
    -------
    numpy.ndarray
        A 1D array where each entry is the log-sum-exp of the 
        corresponding row from Z.

    Example
    -------
    >>> Z = np.array([[1, 2, 3], [10, 20, 30]])
    >>> log_sum_exp(Z)
    array([ 3.40760596, 30.0001234 ])
    """

    # Find the maximum value in each row of Z for numerical stability.
    Z_max = np.max(Z, axis=1)
    # The subtraction of Z_max ensures that the largest number in each row becomes zero 
    # in the exponentiated sequence, thus avoiding overflow.
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:, None]), axis=1))  # per-colmumn max across rows

def vector2matrix(Y, num_classes=3):
    """
    Convert a vector of integer class labels into a matrix of one-hot encoded vectors.

    Parameters
    ----------
    Y : numpy.ndarray
        A 1D numpy array containing integer class labels, where each integer is in the range [0, num_classes-1].
    
    num_classes : int
        The total number of unique classes. Defaults to 3.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array where each row is the one-hot encoded vector representation of the 
        corresponding class label in `Y`.

    Examples
    --------
    >>> Y = np.array([0, 2, 1, 0])
    >>> vector2matrix(Y)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.]])
    """
    y = np.eye(num_classes)[Y]
    return y

class NeuralNetwork:
    """
    A neural network class that provides methods to train and predict with one hidden layer and sigmoid activation.
    
    Attributes
    ----------
    hidden_layer_sizes : list
        List of integers indicating the size of each hidden layer.
    λ : float
        L2 regularization parameter. Default is 1.
    max_iter : int
        Maximum number of iterations for optimization. Default is 100.
    classification : bool
        Determines if the task is classification (True) or regression (False).
    weights : list of tuples
        Weights of the neural network where each tuple contains (W, b) for each layer.
    layer_sizes : list
        Sizes of each layer in the neural network, including input and output layers.

    Methods
    -------
    funObj(weights_flat, X, y)
        Compute loss and gradient for a given set of weights.
    fit(X, y)
        Train the neural network using L-BFGS optimization.
    fit_SGD(X, y, batch_size=500, alpha=1e-3)
        Train the neural network using Stochastic Gradient Descent.
    predict(X)
        Predict the outputs for a given set of inputs.
    """

    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, λ=1, max_iter=100):
        """Initialize neural network with hyperparameters."""
        self.hidden_layer_sizes = hidden_layer_sizes
        self.λ = λ
        self.max_iter = max_iter

    def funObj(self, weights_flat, X, y):
        """
        Compute the objective function value and its gradient for the given weights.
        
        Parameters
        ----------
        weights_flat : ndarray
            1-D array containing the flattened weights for the neural network.
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape (n_samples, n_outputs)
            True labels/target values.

        Returns
        -------
        f : float
            Objective value (loss).
        g : ndarray
            Gradient of the objective value with respect to the weights.
        """

        # Ensure that y is a 2D array for consistency.
        if y.ndim == 1:
            y = y[:, None]

        # Determine the sizes of entire layers.
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]

        # Determine problem type. If y has more than one column, assume classification. 
        self.classification = (
            y.shape[1] > 1    # assume it's classification if y has more than 1 column.
        )  
        
        # Convert flat weights to a list of (weight, bias) tuples for each layer.
        weights = unflatten_weights(weights_flat, self.layer_sizes)

        # Initialize activations list. The first activation is simply the input data.
        activations = [X]
        for W, b in weights:
            # Compute the pre-activation (linear combination of weights and data plus bias).
            Z = X @ W.T + b
            # Apply the sigmoid activation function element-wise.
            X = 1 / (1 + np.exp(-Z))
            # Store post-activation values for backpropagation later.
            activations.append(X)

        # yhat holds the final layer's pre-activation values
        yhat = Z

        if self.classification:  # softmax
            # Compute the normalizing factor for the softmax function in our classification problem.
            tmp = np.sum(np.exp(yhat), axis=1)
            # Compute the negative log likelihood loss using a stable log-sum-exp trick.
            f = -np.sum(yhat[y.astype(bool)] - log_sum_exp(yhat))
            # Gradient computation for softmax activation.
            grad = np.exp(yhat) / tmp[:, None] - y
        else:  
            # L2 (Mean Squared Error) loss for regression problems.
            f = 0.5 * np.sum((yhat - y) ** 2)
            # Simple gradient computation for L2 loss.
            grad = yhat - y  

        # Compute the gradients for weights and biases of the last layer using backpropagation formula.
        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        # Initialize the gradient list for all layers.
        g = [(grad_W, grad_b)]

        # Backpropagate the gradient through each layer.
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W, b = weights[i]
            # Propagate gradient backwards through weights.
            grad = grad @ W
            # Element-wise multiply by derivative of sigmoid function for the activations. 
            grad = grad * (
                activations[i] * (1 - activations[i])
            )  # gradient of logistic loss
            # Compute gradients for weights and biases at this layer.
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)

            # Insert gradients at the beginning of the gradient list
            g = [(grad_W, grad_b)] + g  # insert to start of list

        # Convert list of gradients for each layer back into a flat array for optimization.
        g = flatten_weights(g)

        # Add L2 regularization term to the objective function value and its gradient.
        f += 0.5 * self.λ * np.sum(weights_flat**2)
        g += self.λ * weights_flat

        return f, g

    def fit(self, X, y):
        """
        Fit the neural network to training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape (n_samples, n_outputs)
            True labels/target data.
        
        Returns
        -------
        None
        """

        # Ensure that y is a 2D array for consistency.
        if y.ndim == 1:
            y = y[:, None]

        # Determine the sizes of entire layers.
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]

        # Determine problem type. If y has more than one column, assume classification. 
        self.classification = (
            y.shape[1] > 1    # assume it's classification if y has more than 1 column.
        )  

        # Initialize neural network weights with small random values.
        scale = 0.01
        weights = list()

        # Create weights and biases for each layer, using the defined sizes.
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            b = scale * np.random.randn(1, self.layer_sizes[i + 1])
            weights.append((W, b))

        # Flatten the structured list of weights and biases into a single 1D array for optimization algorithms. 
        weights_flat = flatten_weights(weights)
        
        # Commented out gradient check. This can be useful for debugging the backpropagation implementation.
        # check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)

        # Optimize the flattened weights using an optimization algorithm.
        # The objective function and its gradient are defined in the 'funObj' method.
        (weights_flat_new,) = minimize(
            lambda w: self.funObj(w, X, y)[0],  # Return the loss value from funObj
            jac=lambda w: self.funObj(w, X, y)[1],  # Return the gradient value from funObj
            x0=weights_flat,    # Starting point for optimization is the initialized weights
            options={"maxiter": self.max_iter}, # Limit the maximum number of iterations
        ).x

        # Convert the optimized flat array of weights back into the original structured format.
        # This will be used for making predictions and further processing.
        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)

    def fit_SGD(self, X, y, batch_size=500, alpha=1e-3):
        """
        Fit the neural network model using Stochastic Gradient Descent (SGD) on the provided training data.

        Parameters:
        ----------
        X : ndarray
            Training data.
        y : ndarray
            The target data. Can either be a 1D array for binary classification/regression 
            or a 2D array for multi-class classification.
        batch_size : int, optional
            The number of samples per batch for the SGD. Default is 500.
        alpha : float, optional
            The learning rate for SGD. Default is 1e-3.

        Returns:
        -------
        None
        """

        # Ensure that y is a 2D array for consistency.
        if y.ndim == 1:
            y = y[:, None]

        # Determine the sizes of entire layers.
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]

        # Determine problem type. If y has more than one column, assume classification. 
        self.classification = (
            y.shape[1] > 1    # assume it's classification if y has more than 1 column.
        )  

        # Initialize neural network weights with small random values.
        scale = 0.01
        weights = list()

        # Create weights and biases for each layer, using the defined sizes.
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            b = scale * np.random.randn(1, self.layer_sizes[i + 1])
            weights.append((W, b))
        
        # Flatten the structured list of weights and biases into a single 1D array for optimization algorithms. 
        weights_flat = flatten_weights(weights)

        # Begin the Stochastic Gradient Descent (SGD) training loop.
        n = X.shape[0]
        for epoch in range(self.max_iter):

            # Periodically (every 25 epochs) compute and print the overall loss to monitor progress.
            if epoch % 25 == 0:
                f, g = self.funObj(weights_flat, X, y)
                print("Epoch %d, Loss = %f" % (epoch, f))

            # Split the dataset into mini-batches and perform an update for each batch.
            for t in range(n // batch_size):

                # Randomly select a subset (mini-batch) of the data without replacement.
                batch = np.random.choice(n, size=batch_size, replace=False)

                # Calculate loss and gradient for the current mini-batch.
                f, g = self.funObj(weights_flat, X[batch], y[batch])
                # Update the flattened weights using the computed gradient and the learning rate.
                weights_flat = weights_flat - alpha * g

        # Convert the optimized flat array of weights back into the original structured format.
        # This will be used for making predictions and further processing.
        self.weights = unflatten_weights(weights_flat, self.layer_sizes)

    def predict(self, X):
        """
        Predict the output for the given input data using the trained neural network.

        Parameters:
        ----------
        X : ndarray
            Input data for prediction; shape = (number of samples, number of features).

        Returns:
        -------
        ndarray
            Predicted labels for classification tasks or predicted values for regression tasks.
            Can either be a 1D array for binary classification/regression 
            or a 2D array for multi-class classification.

        Notes:
        -----
        This method processes the input through each layer of the neural network, applying 
        weights, biases, and activation functions. For classification tasks, the function 
        returns the class with the highest score from the output layer. For regression tasks, 
        it returns the raw output values.
        """
        # Iterate over the layers of the neural network, applying weights, biases, and activation function.
        # The loop processes the input 'X' through each layer, updating 'X' at each step.
        for W, b in self.weights:
            # Compute the weighted sum of inputs for the current layer.
            Z = X @ W.T + b
            # Apply the non-linear sigmoid activation function. This squashes the output between 0 and 1.
            X = 1 / (1 + np.exp(-Z))

        if self.classification:
            # For classification tasks
            return np.argmax(Z, axis=1)
        else:
            # For regression tasks
            return Z