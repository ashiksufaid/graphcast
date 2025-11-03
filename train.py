import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np

def train(model, num_epochs, criterion, optimizer, scheduler, train_loader, val_loader, device, checkpoint_dir):
    model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    # --- Create checkpoint directory ONCE ---
    # Also fixed the nesting bug (e.g., "checkpoints/checkpoints/...")
    os.makedirs(checkpoint_dir, exist_ok=True) 
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            batch.to(device)
            optimizer.zero_grad()
            predicted_delta = model(batch)
            true_delta = batch['grid'].y
            loss = criterion(predicted_delta, true_delta)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            num_batches += 1
            running_loss += loss.item()*batch.batch_size
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss/len(train_loader.dataset)
        train_losses.append(epoch_loss)

        with torch.no_grad():
            model.eval()
            running_val_loss = 0
            for val_batch in val_loader:
                val_batch.to(device)
                val_predicted_delta = model(val_batch)
                val_true_delta = val_batch['grid'].y
                val_loss = criterion(val_predicted_delta, val_true_delta)
                running_val_loss += val_loss.item()*val_batch.batch_size
            epoch_val_loss = running_val_loss/len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f}")
    # --- Save the best model based on validation loss ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")

    # --- [THIS ENTIRE BLOCK WAS MOVED] ---
    # This code now runs *AFTER* all epochs are complete.
    
    print("Training Complete.")
    
    # Save the final model
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return train_losses, val_losses

        
                