import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.model import ClassifierNetwork
from src.data_loader import DataProcessor
import numpy as np
import pandas as pd
from datetime import datetime
from src.early_stopping import EarlyStopping

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    loss_list = []
    accuracy_list = []

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Sigmoid to get probas. Loss function automatically applies sigmoid to its computation. We need to add it to the network output.
        predicted_labels = (torch.sigmoid(y_pred) > 0.5).float()  # 0.5 thresh for convenience
        accuracy = torch.mean((predicted_labels == batch_y).float())

        loss_list.append(loss.item())
        accuracy_list.append(accuracy.item())

    return np.mean(loss_list), np.mean(accuracy_list)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on test set and return average loss."""
    model.eval()
    loss_list = []
    accuracy_list = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            predicted_labels = (torch.sigmoid(y_pred) > 0.5).float()  # Sigmoid to get probas. Loss function automatically applies sigmoid to its computation. We need to add it to the network output.
            accuracy = torch.mean((predicted_labels == batch_y).float())

            loss_list.append(loss.item())
            accuracy_list.append(accuracy.item())

    return np.mean(loss_list), np.mean(accuracy_list)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load data
    print("\nLoading data...")
    dp = DataProcessor()
    dataset = dp()

    # Train-test split (80-20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {train_size}, Test size: {test_size}")

    # Create dataloaders
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize model with parameters from notebook
    input_size = dataset.tensors[0].shape[1]
    output_size = 1

    net = ClassifierNetwork(
        n_hidden_layers=2,
        input_size=input_size,
        hidden_size=16,
        output_size=output_size
    )
    net = net.to(device)

    print("\nModel architecture:")
    print(net)
    print(f"\nInput size: {input_size}, Output size: {output_size}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)

    data_dict = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    early_stopping = EarlyStopping(patience=5, delta=0.001)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(
            net, train_loader, criterion, optimizer, device
        )
        test_loss, test_accuracy = evaluate(net, test_loader, criterion, device)

        # Print epoch results
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Accuracy: {train_accuracy:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Accuracy: {test_accuracy:.4f}")

        data_dict["epoch"].append(epoch + 1)
        data_dict["train_loss"].append(train_loss)
        data_dict["train_accuracy"].append(train_accuracy)
        data_dict["test_loss"].append(test_loss)
        data_dict["test_accuracy"].append(test_accuracy)

        early_stopping(test_loss, net)  # Simple Early Stopping implementation (manual)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Create DataFrame from rows
    metrics_df = pd.DataFrame(data_dict)

    # Save dataframe to the data/ folder
    metrics_df.to_csv(f"data/training_metrics_{timestamp}.csv", index=False)

    print("-" * 60)
    print("Training complete!")

    # Save the trained model
    model_path = f"models/trained_model_{timestamp}.pth"
    torch.save(net.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
