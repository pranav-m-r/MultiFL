from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from multifl.dataloader import HatefulMemesDataset
from typing import List, Tuple, Dict, Optional # Added Dict, Optional


fds = None  # Cache FederatedDataset

# Image transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Constants for layer prefixes
IMAGE_PREFIX = "encoders.image"
TEXT_PREFIX = "encoders.text"
FUSION_PREFIX = "fusion_module."
HEAD_PREFIX = "head_module."


def load_data(partition_id: int, num_partitions: int):
    """Load partition Hateful Memes data."""
    global fds
    if fds is None:
        try:
            # partitioner = IidPartitioner(num_partitions=num_partitions)
            partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.5, partition_by="label")
            fds = FederatedDataset(
                dataset="neuralcatcher/hateful_memes",
                partitioners={"train": partitioner},
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    partition = fds.load_partition(partition_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise
    multimodal_dataset = HatefulMemesDataset(partition, tokenizer, transform=image_transform)

    train_size = int(0.8 * len(multimodal_dataset))
    test_size = len(multimodal_dataset) - train_size
    generator = torch.Generator().manual_seed(partition_id)
    train_dataset, test_dataset = torch.utils.data.random_split(multimodal_dataset, [train_size, test_size], generator=generator)

    num_workers = 0
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=32, num_workers=num_workers)
    return trainloader, testloader


# modality = 2 is image, 1 is text, 0 is both
def train_model(model: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device, modality: int) -> float:
    """Train the model, freezing unused encoders for missing modalities."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Freeze layers based on modality
    # print(f"Training with modality {modality}. Freezing layers accordingly.")
    for name, param in model.named_parameters():
        param.requires_grad = True # Ensure all are trainable by default

        if modality == 1: # Text only, freeze image encoder
            if name.startswith(IMAGE_PREFIX):
                # print(f"  Freezing {name}")
                param.requires_grad = False
        elif modality == 2: # Image only, freeze text encoder
            if name.startswith(TEXT_PREFIX):
                # print(f"  Freezing {name}")
                param.requires_grad = False

    # Filter parameters for the optimizer after freezing
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=0.001)

    model.train()
    total_loss = 0.0
    num_batches = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in trainloader:
            # Ensure batch is a dictionary and keys exist
            if not isinstance(batch, dict) or "image" not in batch or "text" not in batch or "label" not in batch:
                print(f"Warning: Skipping malformed batch: {type(batch)}")
                continue

            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            labels = batch["label"].to(device)

            model_input: Dict[str, Optional[torch.Tensor]] = {} # Use Optional for clarity

            # Prepare input based on the training modality
            if modality == 0: # Both modalities
                model_input["image"] = images
                model_input["text"] = texts
            elif modality == 1: # Text only
                model_input["text"] = texts
                model_input["image"] = torch.zeros_like(images)
            elif modality == 2: # Image only
                model_input["image"] = images
                model_input["text"] = torch.zeros_like(texts)
            else:
                raise ValueError(f"Unknown training modality: {modality}")

            optimizer.zero_grad()
            outputs = model(model_input) # Forward pass still happens on all parts
            loss = criterion(outputs, labels)

            # Check for NaN/inf loss
            if not torch.isfinite(loss):
                print(f"Warning: Encountered non-finite loss ({loss.item()}) in epoch {epoch}, batch {batch_count}. Skipping batch.")
                continue

            loss.backward()
            optimizer.step() # Updates only requires_grad=True params

            epoch_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        # print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        total_loss += avg_epoch_loss
        num_batches += 1

    # Return average loss over epochs
    return total_loss / num_batches if num_batches > 0 else 0.0


def test_model(model: nn.Module, valloader: DataLoader, device: torch.device, modality: int) -> Tuple[float, float]:
    """Evaluate the model, simulating missing modalities by providing zero inputs."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # print(f"Evaluating with simulated modality {modality}.")

    with torch.no_grad():
        for batch in valloader:
            if not isinstance(batch, dict) or "image" not in batch or "text" not in batch or "label" not in batch:
                print(f"Warning: Skipping malformed batch during evaluation: {type(batch)}")
                continue

            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            labels = batch["label"].to(device)

            # Prepare input based on the evaluation modality
            model_input: Dict[str, Optional[torch.Tensor]] = {}
            if modality == 0: # Evaluate using both modalities
                model_input["image"] = images
                model_input["text"] = texts
            elif modality == 1: # Evaluate using only text (provide zeros for image)
                model_input["text"] = texts
                model_input["image"] = torch.zeros_like(images)
                # print("  Testing with text only (zeros for image)")
            elif modality == 2: # Evaluate using only image (provide zeros for text)
                model_input["image"] = images
                model_input["text"] = torch.zeros_like(texts)
                # print("  Testing with image only (zeros for text)")
            else:
                 raise ValueError(f"Unknown evaluation modality: {modality}")

            outputs = model(model_input)
            loss = criterion(outputs, labels)

            # Check for NaN/inf loss during evaluation as well
            if not torch.isfinite(loss):
                print(f"Warning: Encountered non-finite loss ({loss.item()}) during evaluation. Skipping batch.")
                continue

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Handle division by zero if valloader is empty or all batches were skipped
    num_val_batches = len(valloader)
    if total == 0:
        print("Warning: No samples processed during evaluation.")
        return 0.0, 0.0

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / num_val_batches if num_val_batches > 0 else 0.0

    # print(f"Evaluation Results (Modality {modality}): Avg Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    return avg_loss, accuracy


def get_weights(model: torch.nn.Module, modality: int) -> Tuple[List[np.ndarray], List[str]]:
    """Get model weights and keys relevant to the specified modality for sending updates."""
    state_dict = model.state_dict()
    relevant_weights = []
    relevant_keys = []

    for name, param in state_dict.items():
        include_param = False
        is_image = name.startswith(IMAGE_PREFIX)
        is_text = name.startswith(TEXT_PREFIX)
        is_fusion = name.startswith(FUSION_PREFIX)
        is_head = name.startswith(HEAD_PREFIX)

        # Determine if this parameter should have been trained by this client
        if modality == 0: # Both modality client trains everything
            if is_image or is_text or is_fusion or is_head: include_param = True
        elif modality == 1: # Text only client trains text, fusion, head
            if is_text or is_fusion or is_head: include_param = True
        elif modality == 2: # Image only client trains image, fusion, head
            if is_image or is_fusion or is_head: include_param = True
        else:
             raise ValueError(f"Unknown modality {modality} in get_weights")

        if include_param:
            relevant_weights.append(param.detach().cpu().numpy())
            relevant_keys.append(name)

    if not relevant_keys:
        print(f"Warning: get_weights for modality {modality} resulted in no relevant keys.")

    return relevant_weights, relevant_keys


def set_weights(model: torch.nn.Module, params_with_keys: Tuple[List[np.ndarray], List[str]]) -> None:
    """Set model weights using provided parameters and their keys."""
    parameters, keys = params_with_keys
    if not parameters or not keys:
        print("Warning: set_weights received empty parameters or keys. No weights set.")
        return
    if len(parameters) != len(keys):
        raise ValueError(f"Mismatch in set_weights: {len(parameters)} parameters, {len(keys)} keys. Keys: {keys}")

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        print("Warning: Model has no parameters. Cannot set weights.")
        return

    # Create updates dictionary, ensuring tensors are on the correct device
    updates = {key: torch.tensor(param, device=model_device) for key, param in zip(keys, parameters)}

    current_state_dict = model.state_dict()
    updated_state_dict = OrderedDict()
    loaded_keys = set()
    missing_in_model = []
    shape_mismatches = []

    # Iterate through the model's state dict to ensure order and handle missing updates
    for name, current_param in current_state_dict.items():
        if name in updates:
            update_tensor = updates[name]
            if current_param.shape == update_tensor.shape:
                updated_state_dict[name] = update_tensor.to(current_param.dtype)
                loaded_keys.add(name)
            else:
                shape_mismatches.append(f"Layer {name}: model={current_param.shape}, received={update_tensor.shape}")
                updated_state_dict[name] = current_param # Keep original if shape mismatch
        else:
            # Keep the existing parameter if no update was provided for it
            updated_state_dict[name] = current_param

    provided_keys = set(keys)
    keys_not_in_model = provided_keys - set(current_state_dict.keys())
    keys_not_loaded = provided_keys - loaded_keys - keys_not_in_model

    if keys_not_in_model:
        print(f"Warning: Keys provided in set_weights but not found in model state_dict: {keys_not_in_model}")
    if shape_mismatches:
        print(f"Warning: Shape mismatches encountered in set_weights: {shape_mismatches}")

    try:
        model.load_state_dict(updated_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Error during model.load_state_dict: {e}")
        print("Updated state dict keys:", list(updated_state_dict.keys()))
        print("Model state dict keys:", list(model.state_dict().keys()))
        raise
