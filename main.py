import torch
import numpy as np
import json
import torch.nn
import torch.optim as optim
from torch.nn import MSELoss
from WeatherGraphDataset import WeatherDataset
from torch_geometric.data import DataLoader
from train import train
from GraphCast import GraphCast


BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = "static_graph.pt"
VARIABLES = ['T2', 'z_500', 'u_250']
NUM_OUTPUT_VARS = len(VARIABLES)
LATENT_DIM = 128
MLP_HIDDEN_DIM = 128
PROCESSOR_LAYERS = 6
LEARNING_RATE= 1e-4

train_path = "/storage/ashik/wbcoarse_train.nc"
val_path = "/storage/ashik/wbcoarse_val.nc"

train_mean_std = "/storage/ashik/train_precomp.nc"
val_mean_std = "/storage/ashik/val_precomp.nc"

temp_graph = torch.load(GRAPH_PATH, weights_only=False)
grid_node_in_feats = len(VARIABLES) * 2 # t-1 and t concatenated
mesh_node_in_feats = temp_graph['mesh'].x.shape[1]
edge_in_feats_g2m = temp_graph['grid', 'to', 'mesh'].edge_attr.shape[1]
mesh_edge_in_feats = temp_graph['mesh', 'to', 'mesh'].edge_attr.shape[1]
edge_in_feats_m2g = temp_graph['mesh', 'to', 'grid'].edge_attr.shape[1]

model = GraphCast(
    grid_node_in_feats=grid_node_in_feats,
    mesh_node_in_feats=mesh_node_in_feats,
    mesh_edge_in_feats=mesh_edge_in_feats,
    edge_in_feats_g2m=edge_in_feats_g2m,
    edge_in_feats_m2g=edge_in_feats_m2g,
    num_output_vars=NUM_OUTPUT_VARS,
    latent_dim=LATENT_DIM,
    mlp_hidden_dim=MLP_HIDDEN_DIM,
    processor_layers=PROCESSOR_LAYERS
)

train_dataset = WeatherDataset(train_path, GRAPH_PATH, train_mean_std)
val_dataset = WeatherDataset(val_path, GRAPH_PATH, val_mean_std)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False)

criterion = MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
num_train_batches = len(train_loader)
TOTAL_TRAINING_STEPS = num_train_batches * EPOCHS
WARMUP_STEPS = 1000
warmup_fraction = WARMUP_STEPS / TOTAL_TRAINING_STEPS
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr = LEARNING_RATE,
    total_steps=TOTAL_TRAINING_STEPS,
    pct_start=warmup_fraction,
    anneal_strategy='cos'
)

train_losses, val_losses = train(model=model, num_epochs=EPOCHS, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, device=DEVICE, checkpoint_dir="/storage/ashik")

losses_data = {
    'train_losses': train_losses,
    'val_losses': val_losses
}

with open('training_losses2.json', 'w') as f:
    json.dump(losses_data, f, indent=4)

print("Losses saved to training_losses2.json")

# --- To load them back later ---
# with open('training_losses.json', 'r') as f:
#     loaded_data = json.load(f)
# loaded_train_losses = loaded_data['train_losses']
# loaded_val_losses = loaded_data['val_losses']


# ##
# # --- Setup Model --
# model.to(DEVICE) # Make sure to move model to device *before* this

# # --- Count Parameters ---
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")

# # --- Get the number of batches ---
# num_train_batches = len(train_loader)
# num_val_batches = len(val_loader)

# print(f"Number of training batches: {num_train_batches}")
# print(f"Number of validation batches: {num_val_batches}")
