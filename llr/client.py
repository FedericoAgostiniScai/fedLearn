import warnings
from argparse import ArgumentParser
import flwr as fl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import utils


class myClient(fl.client.NumPyClient):
    def __init__(self, cid, random_state=0):
        data = pd.read_csv(f"../data/train_{cid}.csv.gz")
        X, y = data.drop(columns="target").to_numpy(
        ), data["target"].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state)
        self.model = utils.get_sklearn_model(client=True)
        utils.set_initial_params(self.model)

    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"Loss = {loss:.4} | acc = {accuracy:.4}")
        return loss, len(self.X_test), {"accuracy": accuracy}
# %%


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--id", type=int, default=0, help="Client ID")

    return (vars(parser.parse_args()))

# %%


if __name__ == "__main__":

    args = parse_args()

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080", client=myClient(cid=args["id"]))
