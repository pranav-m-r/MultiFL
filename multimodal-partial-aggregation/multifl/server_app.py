from flwr.common import Context, ndarrays_to_parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from multifl.models import build_multimodal_model
from multifl.strategy import FedSim
from transformers import AutoTokenizer
from typing import Dict, List, Tuple


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate fit metrics (e.g., train loss)."""
    valid_losses = [m[1]["train_loss"] for m in metrics if "train_loss" in m[1] and isinstance(m[1]["train_loss"], (int, float))]
    if not valid_losses:
        return {"train_loss": 0.0}
    avg_loss = sum(valid_losses) / len(valid_losses)
    return {"train_loss": avg_loss}


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics (e.g., validation loss and accuracy)."""
    # Filter out potential errors reported by clients and non-numeric values
    valid_losses = [m[1]["val_loss"] for m in metrics if "val_loss" in m[1] and isinstance(m[1]["val_loss"], (int, float))]
    valid_accuracies = [m[1]["val_accuracy"] for m in metrics if "val_accuracy" in m[1] and isinstance(m[1]["val_accuracy"], (int, float))]

    avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0.0
    avg_accuracy = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else 0.0

    return {"val_loss": avg_loss, "val_accuracy": avg_accuracy}


def server_fn(context: Context):
    # Read configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    min_fit_clients = context.run_config.get("min-fit-clients", 2)
    min_avail_clients = context.run_config.get("min-available-clients", 2)

    # Initialize model parameters and get keys
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = build_multimodal_model(text_vocab_size=tokenizer.vocab_size)
    parameter_keys = list(model.state_dict().keys())
    initial_ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(initial_ndarrays)

    # Define strategy - USE FedSim and pass keys
    strategy = FedSim(
        parameter_keys=parameter_keys,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_avail_clients,
        min_available_clients=min_avail_clients,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)
