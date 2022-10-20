class AdalineSGD:
    """Adaptive Linear Neuron Classiier.

    Parameters:
    eta: float
        Learning rate between 0 and 1.
    n_iter: int
        Passes over the trainig dataset.
    shuffle: bool (deafult true)
        If true shuffles training data every epoch to prevent cycles.
    random_state: int
        Random number generator seed from the random weight initialization.
    
    Attributes
    w_: id-array
        Weights after fitting.
    b_: Scalar
        Bias unit after fitting.
    losses_: list
        Mean squared error loss function value averaged over all training examples
        in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        pass

    def fit(self, X, y):
        """
        Fit the training data.

        Parameters
        X: {array-like}, shape = [n_examples, n_features]
        y: array-like, shape = [n_examples]

        Returns
        self: object
        """

        pass

    def partial_fit(self, X, y):
        """Fit training data without reinitialializing the weight.
        """

        pass

    def _shuffle(self, X, y):
        """Shuffle training data."""
        pass

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers."""
        pass

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update weights."""
        pass
    
    def net_input(self, X):
        """Calculate net input."""
        pass

    def activation(self, X):
        """Compute linear activation."""
        pass

    def predict(self, X):
        """Return class label after unit step."""
        pass