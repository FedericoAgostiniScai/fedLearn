# %%

from argparse import ArgumentParser
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import flwr as fl

import utils

# %%

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class myClient(fl.client.NumPyClient):
    def __init__(self, model, cid, random_state=0):
        data = pd.read_csv(f"../data/train_{cid}.csv.gz")
        X, y = data.drop(columns="target").to_numpy(
        ), data["target"].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state)
        self.model = model

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

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.X_train,
            self.y_train,
            # validation_split=0.1,
            **config  # Get hyperparameters for this round
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.X_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            # "val_loss": history.history["val_loss"][0],
            # "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(
            self.X_test,
            self.y_test,
            **config
        )
        num_examples_test = len(self.X_test)
        return loss, num_examples_test, {"accuracy": accuracy}


# %%

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--id", type=int, default=0, help="Client ID")

    return (vars(parser.parse_args()))

# %%


if __name__ == "__main__":

    args = parse_args()

    with open("params.json", "r") as f:
        params = json.load(f)

    model = utils.make_nn(**params)
    client = myClient(model, cid=args["id"])

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080", client=client)
