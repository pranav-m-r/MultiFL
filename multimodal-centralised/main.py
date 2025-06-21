import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from models import build_multimodal_model
from dataloader import HatefulMemesDataset
from torch.utils.data import random_split

# Image transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epoch_val_accuracies = []
    current_run_best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader):
            image = batch['image'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model({"image": image, "text": text})
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                text = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model({"image": image, "text": text})
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {total_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        val_acc_percent = 100 * correct / total
        print(f'Val Accuracy: {val_acc_percent:.2f}%')
        epoch_val_accuracies.append(val_acc_percent)
        if val_acc_percent > current_run_best_val_acc:
            current_run_best_val_acc = val_acc_percent
    
    return current_run_best_val_acc, epoch_val_accuracies


def main():
    # Hyperparameter configurations to test
    hyperparameter_configs = [
        {"lr": 0.001, "batch_size": 32, "lstm_hidden_size": 100, "mlp_hidden_dims": [128], "mlp_dropout": 0.3, "num_epochs": 50},
        {"lr": 0.001, "batch_size": 32, "lstm_hidden_size": 100, "mlp_hidden_dims": [256, 128], "mlp_dropout": 0.5, "num_epochs": 50},
        {"lr": 0.001, "batch_size": 32, "lstm_hidden_size": 150, "mlp_hidden_dims": [256, 128], "mlp_dropout": 0.5, "num_epochs": 50},
        {"lr": 0.001, "batch_size": 32, "lstm_hidden_size": 150, "mlp_hidden_dims": [256, 128], "mlp_dropout": 0.3, "num_epochs": 50},
    ]

    best_overall_val_accuracy = 0.0
    best_config = None
    results_log = []

    # Load Hateful Memes dataset
    # Use the official train and validation splits
    train_hf_dataset = load_dataset('neuralcatcher/hateful_memes', split='train')
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create HatefulMemesDataset instances
    dataset = HatefulMemesDataset(train_hf_dataset, tokenizer, transform=image_transform)
    # Split the dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset_processed, val_dataset_processed = random_split(dataset, [train_size, val_size])

    for config_idx, config in enumerate(hyperparameter_configs):
        print(f"\n--- Running Configuration {config_idx+1}/{len(hyperparameter_configs)} ---")
        print(config)

        # DataLoaders
        train_loader = DataLoader(train_dataset_processed, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset_processed, batch_size=config["batch_size"])
        
        # Build model
        model = build_multimodal_model(
            text_vocab_size=tokenizer.vocab_size,
            lstm_hidden_size=config["lstm_hidden_size"],
            mlp_hidden_dims=config["mlp_hidden_dims"],
            mlp_dropout=config["mlp_dropout"]
        )
        
        # Train model
        current_best_val_acc, all_epoch_accuracies = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=config["num_epochs"],
            learning_rate=config["lr"]
        )
        
        results_log.append({"config": config, "best_val_accuracy_this_run": current_best_val_acc, "epoch_accuracies": all_epoch_accuracies})

        if current_best_val_acc > best_overall_val_accuracy:
            best_overall_val_accuracy = current_best_val_acc
            best_config = config
            # torch.save(model.state_dict(), f"best_model_acc_{current_best_val_acc:.2f}.pth") # Optionally save best model
    
    print("\n--- Hyperparameter Tuning Finished ---")
    print(f"Overall Best Validation Accuracy: {best_overall_val_accuracy:.2f}%")
    print(f"Best Hyperparameter Configuration: {best_config}")
    print("\nFull Log:")
    for entry in results_log:
        print(entry)
    
    # Plotting validation accuracies
    plt.figure(figsize=(20, 10))
    for idx, entry in enumerate(results_log):
        config_label = f"Config {idx+1}: LR={entry['config']['lr']}, LSTM_H={entry['config']['lstm_hidden_size']}, MLP_D={entry['config']['mlp_hidden_dims']}, MLP_Drop={entry['config']['mlp_dropout']}"
        plt.plot(range(1, entry['config']['num_epochs'] + 1), entry['epoch_accuracies'], marker='o', label=config_label)
    plt.title('Validation Accuracy vs. Epochs for Different Configurations')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.ylim(30, 100)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_accuracies_plot.png") # Save the plot

if __name__ == '__main__':
    main()