import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

class WeatherDataset(Dataset):
    """
    Custom PyTorch Dataset for loading weather data for GraphCast.
    
    This class loads the time-varying features and attaches them
    to a static HeteroData graph object.
    """
    def __init__(self, xr_dataset_path, static_graph_path, mean_std_path, variables =['T2', 'z_500', 'u_250']):
        """
        Args:
            xr_dataset_path (str): Path to the preprocessed .nc file.
            static_graph_path (str): Path to the saved static .pt file 
                                     containing the HeteroData graph.
            mean_std_path (str): Path to the .pt file containing the 
                                 {'mean': ..., 'std': ...} dictionary.
            variables (list): List of variable names to load.
        """
        super().__init__()
        self.data = xr.open_dataset(xr_dataset_path)
        self.static_graph = torch.load(static_graph_path, weights_only=False)
        stats = torch.load(mean_std_path, weights_only=False)
        self.mean = stats['mean']
        self.std = stats['std']

        feature_list = [self.data[v].values.reshape(self.data.dims['time'], -1) for v in variables]
        features_np = np.stack(feature_list, axis=-1)
        self.features = torch.from_numpy(features_np).float()
        self.num_timesteps = self.features.shape[0]

    def __len__(self):
        return self.num_timesteps - 2

    def __getitem__(self, idx):
        raw_sequence = self.features[idx:idx+3]
        normalized_sequence = (raw_sequence - self.mean) / (self.std + 1e-08)
        x_t_minus_1 = normalized_sequence[0] # Shape: [nodes, variables]
        x_t = normalized_sequence[1]         # Shape: [nodes, variables]
        x_t_plus_1 = normalized_sequence[2]  # Shape: [nodes, variables]
        
        input_features = torch.cat([x_t_minus_1, x_t], dim=-1)
        target_delta = x_t_plus_1 - x_t
        
        data = self.static_graph.clone()
        data['grid'].x = input_features
        data['grid'].y = target_delta

        return data