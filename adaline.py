class AdalineGD:
    """Adaptive Linear Neuron Classifier.

    Parameters
    eta: float
        Learning rate between 0 and 1.
    n_iter: int
        Passes over the training dataset.
    random_state: int
        Random number generator to seed weight initialization.
    
    Attributes
    w_: 1d array
        Weights after fitting.
    b_: Scalar
        Bias units after fitting.
    lossess_: list
        Mean squared error loss function values in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

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

    def net_input(self, X):
        """Calculate net input."""
        pass

    def activation(self, X):
        """Compute linear activation."""
        pass

    def predict(self, X):
        """Return class label after unit step."""
        pass