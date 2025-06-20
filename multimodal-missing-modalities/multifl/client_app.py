import torch
from flwr.client import ClientApp, NumPyClient, Client
from flwr.common import Context
from multifl.models import build_multimodal_model
from multifl.task import train_model, test_model, set_weights, get_weights, load_data
from multifl.utils import set_modality, get_modality
import random
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple
import json



class MultimodalFlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs, partition_id):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # using 0 by default, should change when fit is called
        self.modality = 0 # 0: both, 1: text, 2: image
        self.partition_id = partition_id
        # print(f"Client initialized with modality: {self.modality}")

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train or report modality based on server_round."""
        modality = config.get("modality", 0)
        self.modality = modality
        try:
            server_round = config.get("server_round")
            if server_round is None:
                raise ValueError("server_round missing from config.")

            # print(f"Client (modality {self.modality}): Round {server_round}, starting fit.")

            # Deserialize param_keys from JSON string
            param_keys_json = config.get("param_keys_json")
            if param_keys_json is None or not isinstance(param_keys_json, str):
                raise ValueError(f"param_keys_json missing or invalid type in fit config for round {server_round}.")

            param_keys = json.loads(param_keys_json)
            if not isinstance(param_keys, list):
                    raise TypeError(f"Deserialized param_keys is not a list, type: {type(param_keys)}")

            # Check for empty parameters, which might happen if server logic fails
            if not parameters or not param_keys:
                 print(f"Warning: Client (modality {self.modality}) received empty parameters or keys for fit round {server_round}. Skipping training.")
                 return [], 0, {"train_loss": 0.0, "modality": self.modality}

            # print(f"Client (modality {self.modality}): Received {len(param_keys)} keys for training.")
            set_weights(self.model, (parameters, param_keys))

            # Ensure trainloader has data
            if not self.trainloader or len(self.trainloader.dataset) == 0:
                 print(f"Warning: Client (modality {self.modality}) has no training data. Skipping training.")
                 return [], 0, {"train_loss": 0.0, "modality": self.modality}

            train_loss = train_model(self.model, self.trainloader, self.local_epochs, self.device, self.modality)

            updated_weights, updated_keys = get_weights(self.model, self.modality)
            # print(f"Client (modality {self.modality}): Sending back {len(updated_keys)} updated keys.")

            metrics = {"train_loss": float(train_loss), "modality": self.modality, "partition_id": self.partition_id} # Ensure loss is float
            num_examples_train = len(self.trainloader.dataset)
            return updated_weights, num_examples_train, metrics

        except Exception as e:
            # Log the error and return a failure indicator if possible (depends on Flower version/API)
            # For NumPyClient, we return empty weights and indicate error in metrics
            print(f"Error during fit for client (modality {self.modality}): {e}")
            import traceback
            traceback.print_exc()
            # Returning empty list, 0 samples, and error metric
            return [], 0, {"error": str(e), "modality": self.modality}


    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, Dict]:
        """Evaluate the model simulating the client's modality."""
        try:
            # print(f"Client (modality {self.modality}): Starting evaluation.")
            # Deserialize param_keys from JSON string (server should send all keys for eval)
            param_keys_json = config.get("param_keys_json")
            if param_keys_json is None or not isinstance(param_keys_json, str):
                raise ValueError("param_keys_json missing or invalid type in evaluate config.")

            param_keys = json.loads(param_keys_json)
            if not isinstance(param_keys, list):
                    raise TypeError(f"Deserialized param_keys is not a list, type: {type(param_keys)}")

            # Check for empty parameters
            if not parameters or not param_keys:
                 print(f"Warning: Client (modality {self.modality}) received empty parameters or keys for evaluation. Skipping.")
                 return 0.0, 0, {"val_loss": 0.0, "val_accuracy": 0.0}

            # print(f"Client (modality {self.modality}): Received {len(param_keys)} keys for evaluation.")
            set_weights(self.model, (parameters, param_keys)) # Load the full global model state

            # Ensure valloader has data
            if not self.valloader or len(self.valloader.dataset) == 0:
                 print(f"Warning: Client (modality {self.modality}) has no validation data. Skipping evaluation.")
                 return 0.0, 0, {"val_loss": 0.0, "val_accuracy": 0.0}

            val_loss, val_accuracy = test_model(self.model, self.valloader, self.device, self.modality)

            num_examples_val = len(self.valloader.dataset)
            metrics = {"val_loss": float(val_loss), "val_accuracy": float(val_accuracy)} # Ensure floats
            # print(f"Client (modality {self.modality}): Evaluation complete. Loss={val_loss:.4f}, Acc={val_accuracy:.2f}%")

            return float(val_loss), num_examples_val, metrics

        except Exception as e:
            print(f"Error during evaluate for client (modality {self.modality}): {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), 0, {"error": str(e)}


def client_fn(context: Context) -> Client:
    """Create a Flower client."""
    # print(f"context.node_config: {context.node_config}")
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)
    local_epochs = context.run_config.get("local-epochs", 1) # Default to 1 epoch if not set

    # print(f"Creating client for partition {partition_id}/{num_partitions}")

    # Load data for the partition
    try:
        trainloader, valloader = load_data(partition_id, num_partitions)
    except Exception as e:
        print(f"Failed to load data for partition {partition_id}: {e}")
        raise RuntimeError(f"Client {partition_id} could not load data.") from e

    # Load tokenizer and build model
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = build_multimodal_model(text_vocab_size=tokenizer.vocab_size)
    except Exception as e:
        print(f"Failed to load tokenizer or build model for partition {partition_id}: {e}")
        raise RuntimeError(f"Client {partition_id} could not initialize model.") from e

    client = MultimodalFlowerClient(model, trainloader, valloader, local_epochs, partition_id)
    return client.to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
