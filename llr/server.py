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
        utils.set_model_params(model, parameters)
        y_pred = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred)
        accuracy = model.score(X_test, y_test)
        print(f"Loss = {loss:.4} | acc = {accuracy:.4}")
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = utils.get_sklearn_model()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=500),
    )
