"""multifl: A Flower / PyTorch app for multimodal learning."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from multifl.models import build_multimodal_model
from multifl.task import train_model, test_model, set_weights, get_weights, load_data
from transformers import AutoTokenizer


class MultimodalFlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        train_loss = train_model(self.model, self.trainloader, self.local_epochs, self.device)
        return get_weights(self.model), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        val_loss, val_accuracy = test_model(self.model, self.valloader, self.device)
        return val_loss, len(self.valloader.dataset), {"val_loss": val_loss, "val_accuracy": val_accuracy}


def client_fn(context: Context):
    # Load dataset and model
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Build multimodal model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = build_multimodal_model(text_vocab_size=tokenizer.vocab_size)
    local_epochs = context.run_config["local-epochs"]

    # Return Flower client
    return MultimodalFlowerClient(model, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)