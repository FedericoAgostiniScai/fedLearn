import json
import pandas as pd
import flwr as fl
import utils
from sklearn.metrics import log_loss
from typing import Dict


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    data = pd.read_csv(f"../data/test.csv.gz")
    X_test, y_test = data.drop(
        columns="target").to_numpy(), data["target"].to_numpy()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Loss = {loss:.4} | acc = {accuracy:.4}")
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"steps": val_steps}


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    with open("params.json", "r") as f:
        params = json.load(f)

    model = utils.make_nn(**params)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(
            model.get_weights()),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=15),
    )
