"""multifl: A Flower / PyTorch app for multimodal learning."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from multifl.models import build_multimodal_model
from multifl.task import get_weights
from transformers import AutoTokenizer


def server_fn(context: Context):
    # Read configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = build_multimodal_model(text_vocab_size=tokenizer.vocab_size)
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)