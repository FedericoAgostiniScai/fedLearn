# %%

from argparse import ArgumentParser
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import flwr as fl

import utils

# %%

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class myClient(fl.client.NumPyClient):
    def __init__(self, model, cid, random_state=0):
        self.model = model
        data = np.load(f"../data/train_{cid}.npz")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data["X"], data["y"], test_size=0.15, random_state=random_state)
        self.model = utils.make_nn(client=True)

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception(
            "Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


# %%

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--id", type=int, help="Client ID")

    return (vars(parser.parse_args()))

# %%


if __name__ == "__main__":

    args = parse_args()

    with open("params.json", "r") as f:
        params = json.load(f)

    model = utils.make_nn(**params)
    client = myClient(model, cid=args["id"])
