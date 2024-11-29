import os
import argparse
import torch.amp
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.dataset import get_data_loaders

from src.model import UNETR_2D
from src.utils import dice_loss, save_checkpoint, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train a 2D U-Net Segmentation model.')
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs', required=True)
    parser.add_argument('-p', '--checkpoint', type=str, default=None, help='Model checkpoint', required=False)
    args = parser.parse_args()
   
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    model_results_dir = os.path.join(results_dir, f"train_{len(os.listdir(results_dir)) + 1}")
    os.makedirs(model_results_dir, exist_ok=True)

    last_model_path = os.path.join(model_results_dir, f"last.pth")
    best_model_path = os.path.join(model_results_dir, f"best.pth")
    loss_plot_path = os.path.join(model_results_dir, "loss.png")

    # if n_classes == 1:
    #     loss_fn = nn.BCEWithLogitsLoss()
    #     activ_func = "Sigmoid"
    # else:
    criterion = nn.CrossEntropyLoss()

    # scaler = torch.GradScaler()
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    train_loader, val_loader = get_data_loaders(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNETR_2D()
    model.to(device)
    scaler = torch.amp.GradScaler("cuda")

    if args.checkpoint is not None:
        load_checkpoint(torch.load(args.checkpoint), model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_loss = 0  # Initialize best validation Dice score

    train_losses = []
    val_losses = []


    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):        
        # Training phase
        model.train()
        running_train_loss = 0.0
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            optimizer.zero_grad()
            # print(images.shape)
        
            with torch.amp.autocast("cuda"):
                # print("image shape: ", images.shape)
                # Forward pass
                preds = model(images)
                # print("preds shape: ", preds.shape)
                # print("preds_classes: ", preds.unique())
                #   loss = criterion(pred, masks)
                # print("mask_shape: ", masks.shape)
                # print("mask_classes: ", masks.unique())  # Should show class indices (e.g., 0, 1, 2, 3)
                # print("preds shape: ", preds.shape, preds.min(), preds.max())
        
            loss = dice_loss(preds, masks)
            # loss = criterion(preds, masks)
            running_train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                    
        train_losses.append(running_train_loss / len(train_loader))

        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device).squeeze(1)

                preds = model(images)
                # print("mask_shape: ", masks.unique())  # Should show class indices (e.g., 0, 1, 2, 3)
                # print("preds shape: ", preds.shape, preds.min(), preds.max())
                # loss = criterion(preds, masks)
                loss = dice_loss(preds, masks)
                running_val_loss += loss.item()

        val_losses.append(running_val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_losses[-1]:.2f}, - "
            f"Val Loss: {val_losses[-1]:.2f}")
        
        
        # Plot train and validation losses
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Train and Validation Loss Over Epochs")     

        plt.savefig(loss_plot_path)
        plt.close()

        # Save the latest model after each epoch
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=last_model_path)
        
        # Save the best model
        if val_losses[-1] < best_loss:
            save_checkpoint(checkpoint, filename=best_model_path)
            best_loss = val_losses[-1]


if __name__ == "__main__":
    main()