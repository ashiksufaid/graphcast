import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import HeteroConv

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU() # SiLU is the same as swish
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x

class InteractionNetworkLayer(MessagePassing):
    """
    Implements a single layer of an Interaction Network GNN.
    Follows the structure described in the GraphCast paper.
    """
    def __init__(self, node_features, edge_features, hidden_features=64):
        super().__init__(aggr='add')
        # MLP for updating edge features (Eq. 7 / Eq. 11 / Eq. 14)
        # Input: edge_feat + sender_node_feat + receiver_node_fea
        self.edge_mlp = MLP(in_features=edge_features + 2 * node_features,
                            hidden_features=hidden_features,
                            out_features=edge_features)
        # MLP for updating node features (Eq. 8 / Eq. 12 / Eq. 15)
        # Input: node_feat + aggregated_edge_feat
        self.node_mlp = MLP(in_features=node_features + edge_features,
                            hidden_features=hidden_features,
                            out_features=node_features)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (Tensor): Node features [num_nodes, node_features]
            edge_index (LongTensor): Edge connectivity [2, num_edges]
            edge_attr (Tensor): Edge features [num_edges, edge_features]

        Returns:
            Tuple[Tensor, Tensor]: Updated node features, Updated edge features
        """
        # store initial features for residual connection
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        x_initial_dst = x_dst
        edge_attr_initial = edge_attr
        # --- 1. Edge Update ---
        # Select sender (j) and receiver (i) node features based on edge_index
        sender_nodes = x_src[edge_index[0]]
        receiver_nodes = x_dst[edge_index[1]]
        # Concatenate features for edge MLP input [cite: 224, 244, 261]
        edge_mlp_input = torch.cat([sender_nodes, receiver_nodes, edge_attr], dim=-1)
        # but often done here in implementations. Check paper carefully Eq 10 vs 13)
        updated_edge_attr = self.edge_mlp(edge_mlp_input)

        # --- 2. Node Update ---
        # Propagate messages (aggregates updated_edge_attr at receiver nodes)
        # This calls message(), aggregate(), and update() internally.
        # We pass the `updated_edge_attr` to be used in message().
        aggregated_edges = self.propagate(edge_index, x=x, edge_attr=updated_edge_attr)
    
        # Concatenate node features with aggregated edge features 
        node_mlp_input = torch.cat([x_dst, aggregated_edges], dim=-1)
        # Apply node MLP and add residual 
        updated_x_dst = self.node_mlp(node_mlp_input)
        updated_x_dst = x_initial_dst + updated_x_dst # Residual for nodes
        updated_edge_attr = edge_attr_initial + updated_edge_attr # Residual for edges
        return updated_x_dst, updated_edge_attr
        
    def message(self, edge_attr):
        # In the simplest interaction network, the message IS the updated edge feature.
        # This is what gets aggregated in propagate().
        return edge_attr

class GraphCastEncoder(nn.Module):
    """Encodes grid features onto the mesh graph."""
    def __init__(self, 
                 grid_node_feats, mesh_node_feats, mesh_edge_feats, edge_feats_g2m, 
                 latent_dim=64, hidden_layers=1): # Match paper's MLP depth
        super().__init__()
        
        mlp_hidden_dim = latent_dim # Paper uses same hidden/output dim usually

        # Embedding MLPs (Eq. 6)
        self.grid_embed = MLP(grid_node_feats, mlp_hidden_dim, latent_dim)
        self.mesh_embed = MLP(mesh_node_feats, mlp_hidden_dim, latent_dim)
        self.mesh_edge_embed = MLP(mesh_edge_feats, mlp_hidden_dim, latent_dim)
        self.edge_embed_g2m = MLP(edge_feats_g2m, mlp_hidden_dim, latent_dim)
        # self.edge_embed_m2g = MLP(edge_feats_g2m, mlp_hidden_dim, latent_dim)
        self.grid_update_mlp = MLP(latent_dim, mlp_hidden_dim, latent_dim)
        # Interaction Network specifically for grid-to-mesh edges
        # We assume node/edge features are already embedded to latent_dim

        self.conv = HeteroConv({
            ('grid', 'to', 'mesh'): InteractionNetworkLayer(
                node_features=latent_dim,
                edge_features=latent_dim,
                hidden_features=mlp_hidden_dim
            )
        }, aggr="sum")

    def forward(self, data):
        """
        embed everything
        """
        data['grid'].x = self.grid_embed(data['grid'].x)
        data['mesh'].x = self.mesh_embed(data['mesh'].x)
        data['mesh', 'to', 'mesh'].edge_attr = self.mesh_edge_embed(data['mesh', 'to', 'mesh'].edge_attr)
        data['grid', 'to', 'mesh'].edge_attr = self.edge_embed_g2m(
            data['grid', 'to', 'mesh'].edge_attr
        )

        # Grid2Mesh message passing
        edge_index_g2m = data['grid', 'to', 'mesh'].edge_index
        # --- 2️⃣ Grid → Mesh message passing ---
        out_dict = self.conv(data.x_dict,
                             data.edge_index_dict,
                             data.edge_attr_dict)
        # Update mesh node states (residual connection)
        data['mesh', 'to', 'mesh'].x, data['grid', 'to', 'mesh'].edge_attr = out_dict['mesh']
        # --- 3️⃣ Grid node self-update (Eq. 9) ---
        grid_update_delta = self.grid_update_mlp(data['grid'].x) 
        data['grid'].x = data['grid'].x + grid_update_delta #

        return data

class GraphCastProcessor(nn.Module):
    def __init__(self, num_layers=4, latent_dim=64, mlp_hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Each layer has independent (unshared) weights 
            self.layers.append(
                InteractionNetworkLayer(node_features=latent_dim,
                                        edge_features=latent_dim, # Assuming edges are also embedded
                                        hidden_features=mlp_hidden_dim)
            )

    def forward(self, data):
        mesh_node_features = data['mesh', 'to', 'mesh'].x
        mesh_edge_features = data['mesh', 'to', 'mesh'].edge_attr
        mesh_edge_index = data['mesh', 'to', 'mesh'].edge_index

        #sequentially apply each interaction layer
        # mesh_batch = data["mesh"].batch if 'batch' in data["mesh"] else None
        for layer in self.layers:
            mesh_node_features, mesh_edge_features = layer(
                x=mesh_node_features,
                edge_index= mesh_edge_index,
                edge_attr = mesh_edge_features,
            )
        
        data['mesh'].x = mesh_node_features
        data['mesh'].edge_attr = mesh_edge_features

        return data

class GraphCastDecoder(nn.Module):
    """Decodes mesh features back onto the grid graph."""
    def __init__(self, 
                 latent_dim=64, # Input features are latent dim
                 mlp_hidden_dim=64, 
                 edge_feats_m2g=4, num_output_vars=4): # Dimension of raw M2G edge attributes
        super().__init__()

        # Optional: Embedder for Mesh-to-Grid edges if not embedded elsewhere
        self.edge_embeded_m2g = MLP(edge_feats_m2g, mlp_hidden_dim, latent_dim)

        # HeteroConv for the Mesh-to-Grid interaction
        self.conv = HeteroConv({
            ('mesh', 'to', 'grid'): InteractionNetworkLayer(
                                        node_features=latent_dim,
                                        edge_features=latent_dim, # Assumes embedded edges
                                        hidden_features=mlp_hidden_dim)
        }, aggr="sum")

        
        self.out_mlp = MLP(latent_dim, mlp_hidden_dim, num_output_vars)
    def forward(self, data):
        data['mesh', 'to', 'grid'].edge_attr = self.edge_embeded_m2g(data['mesh','to','grid'].edge_attr)
        out_dict = self.conv(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        data['grid'].x, data['mesh', 'to', 'grid'].edge_attr = out_dict['grid']
        return data

class GraphCast(nn.Module):
    """The main GraphCast model combining Encoder, Processor, and Decoder."""
    def __init__(self,
                 grid_node_in_feats: int, 
                 mesh_node_in_feats: int, 
                 mesh_edge_in_feats: int, 
                 edge_in_feats_g2m: int, 
                 edge_in_feats_m2g: int,
                 num_output_vars: int,    # Number of variables in the target delta
                 latent_dim: int = 64,
                 mlp_hidden_dim: int = 64,
                 processor_layers: int = 4): # Use your desired number of layers
        super().__init__()
        
        self.encoder = GraphCastEncoder(
            grid_node_feats=grid_node_in_feats,
            mesh_node_feats=mesh_node_in_feats,
            mesh_edge_feats=mesh_edge_in_feats, 
            edge_feats_g2m=edge_in_feats_g2m,
            latent_dim=latent_dim,
            hidden_layers=1 # MLP hidden layers = 1
        )
        
        self.processor = GraphCastProcessor(
            num_layers=processor_layers,
            latent_dim=latent_dim,
            mlp_hidden_dim=mlp_hidden_dim
        )
        
        self.decoder = GraphCastDecoder(
            latent_dim=latent_dim, 
            mlp_hidden_dim=mlp_hidden_dim,
            edge_feats_m2g=edge_in_feats_m2g 
        )
        

        self.output_mlp = MLP(latent_dim, mlp_hidden_dim, num_output_vars)

    def forward(self, data):
        data = self.encoder(data)
        data = self.processor(data)
        data = self.decoder(data)
        output_delta = self.output_mlp(data['grid'].x)

        return output_delta