# %%

from sklearn.neural_network import MLPClassifier
import numpy as np


def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model."""
    params = [
        model.coefs_,
        model.intercepts_,
    ]
    return params


def set_model_params(model, params):
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coefs_, model.intercepts_ = params
    return model


def set_initial_params(model):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    params = model.get_params()

    model.coefs_ = [np.zeros(n_features)] + [np.zeros(i)
                                             for i in params["hidden_layer_sizes"]]
    model.intercepts_ = [
        np.zeros(i) for i in params["hidden_layer_sizes"]] + [np.zeros(n_classes)]


# %%


def get_sklearn_model(client=False):
    params = {}
    if client:
        params.update(
            {"max_iter": 1, "warm_start": True})
    model = MLPClassifier(**params)
    return model
