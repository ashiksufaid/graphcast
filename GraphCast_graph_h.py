import torch
import numpy as np
import xarray as xr
from sklearn.neighbors import KDTree
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from scipy.spatial.transform import Rotation

# --- Mesh Data Structure ---
class TriangularMesh:
    """Represents a mesh with vertices and triangular faces."""
    def __init__(self, vertices, faces):
        self.vertices = vertices  # shape [num_vertices, 3]
        self.faces = faces        # shape [num_faces, 3]

# --- Mesh Generation - Level 0 (Icosahedron) ---
def get_icosahedron():
    """Returns a regular icosahedron mesh (base mesh)."""
    phi = (1 + np.sqrt(5))/2
    vertices = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ], dtype=np.float32)

    vertices /= np.linalg.norm(vertices, axis=1)[:, None] # normalize to unit sphere

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    
    # Apply rotation to align with poles (from Graphcast_graph.py)
    z_up = np.array([0, 0, 1])
    north = vertices[0] / np.linalg.norm(vertices[0])
    v = np.cross(north, z_up)
    if np.linalg.norm(v) > 1e-8:
        v /= np.linalg.norm(v)
        c = np.dot(north, z_up)
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + k + k @ k * (1 / (1 + c))
        vertices = vertices @ R.T

    return TriangularMesh(vertices=vertices.astype(np.float32), faces=faces)


# --- Mesh Refinement Logic (from your prompt) ---

class _ChildVerticesBuilder:
    def __init__(self, parent_vertices):
        # Corrected initialization variable name
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
        self._all_vertices_list = list(parent_vertices)

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))

    def _create_child_vertex(self, parent_vertex_indices):
        # Middle point projected to unit sphere
        child_vertex_position = self._parent_vertices[
            list(parent_vertex_indices)].mean(0)
        child_vertex_position /= np.linalg.norm(child_vertex_position)
        
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(
            self._all_vertices_list)
        self._all_vertices_list.append(child_vertex_position)

    def get_new_child_vertex_index(self, parent_vertex_indices):
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]

    def get_all_vertices(self):
        return np.array(self._all_vertices_list)

def _two_split_unit_sphere_triangle_faces(triangular_mesh: TriangularMesh) -> TriangularMesh:
    """Splits each triangular face into 4 triangles preserving orientation."""

    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

    new_faces = []
    for ind1, ind2, ind3 in triangular_mesh.faces:
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))

        new_faces.extend([
            [ind1, ind12, ind31],  # 1
            [ind12, ind2, ind23],  # 2
            [ind31, ind23, ind3],  # 3
            [ind12, ind23, ind31],  # 4
        ])

    return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(),
                          faces=np.array(new_faces, dtype=np.int32))

def get_hierarchy_of_triangular_meshes(splits: int) -> list[TriangularMesh]:
    """Returns a sequence of meshes, starting with the base icosahedron."""
    current_mesh = get_icosahedron()
    output_meshes = [current_mesh]
    for _ in range(splits):
        current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
        output_meshes.append(current_mesh)
    return output_meshes

def merge_meshes(mesh_list: list[TriangularMesh]) -> TriangularMesh:
    """Merges all meshes to form the multi-mesh (vertices of finest, all faces)."""
    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0)
    )

def create_edge_index(lat_dim, lon_dim):
    """Creates the edge_index for a regular lat-lon grid."""
    nodes = []
    # Flatten the grid to a list of nodes
    for i in range(lat_dim):
        for j in range(lon_dim):
            nodes.append((i,j))

    edge_index = [[],[]]
    num_nodes = lat_dim * lon_dim

    for node_idx in range(num_nodes):
        lat_i, lon_j = nodes[node_idx] #taking lat and lon for each node

        #connect to 8 neighbours
        for delta_lat in [-1, 0, 1]:
            for delta_lon in [-1, 0, 1]:
                if delta_lat == 0 and delta_lon == 0:
                    continue #don't connect to self

                neighbor_lat = lat_i + delta_lat
                # Handle longitude wraparound
                neighbor_lon = (lon_j + delta_lon) % lon_dim

                # Check if neighbor is within latitude bounds
                if 0 <= neighbor_lat < lat_dim:
                    neighbor_idx = neighbor_lat * lon_dim + neighbor_lon
                    edge_index[0].append(node_idx)
                    edge_index[1].append(neighbor_idx)

    return torch.tensor(edge_index, dtype=torch.long)

def lat_lon_to_cartesian(grid_lat_lon, r=1):
    lat, lon = torch.tensor(grid_lat_lon[:,0], dtype = torch.float32), torch.tensor(grid_lat_lon[:,1], dtype = torch.float32)
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    x = r * torch.cos(lat_rad) * torch.cos(lon_rad)
    y = r * torch.cos(lat_rad) * torch.sin(lon_rad)
    z = r * torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=1)


def get_mesh_edge_features(pos, edge_index):
    """
    Computes the 4D edge features for a mesh graph.
    
    Args:
        pos (torch.Tensor): Node positions in 3D Cartesian coordinates, shape [N, 3].
        edge_index (torch.Tensor): Graph connectivity, shape [2, E].
        
    Returns:
        torch.Tensor: Edge features, shape [E, 4].
    """
    senders, receivers = edge_index[0], edge_index[1]
    pos_senders, pos_receivers = pos[senders], pos[receivers]

    # --- 1. Feature 1: Edge Length ---
    edge_lengths = torch.linalg.norm(pos_senders - pos_receivers, dim=-1, keepdim=True)

    # --- 2. Features 2-4: Local Vector Difference ---
    
    # --- START OF FIX ---
    
    # Convert receiver positions to numpy for scipy
    receivers_np = pos_receivers.cpu().numpy()
    
    # Create a target vector array that has the same shape as the receivers array
    target_vectors_np = np.broadcast_to([1.0, 0.0, 0.0], receivers_np.shape)
    
    # Find the rotation that aligns each receiver vector with its corresponding target vector
    rotations, _ = Rotation.align_vectors(a=receivers_np, b=target_vectors_np)
    
    # --- END OF FIX ---
    
    # Apply these rotations to the corresponding sender vectors'
    pos_senders_rotated = torch.from_numpy(rotations.apply(pos_senders.cpu().numpy())).to(pos.device)
    
    # The rotated receiver is now at (1,0,0)
    ref_vector = torch.tensor([1.0, 0.0, 0.0], device=pos.device)
    local_vector_diff = pos_senders_rotated - ref_vector

    # --- 3. Combine all features ---
    edge_features = torch.cat([edge_lengths, local_vector_diff], dim=-1)
    
    return edge_features

def get_hetero_edge_features(sender_pos, receiver_pos):
    """
    Computes the 4D edge features for heterogeneous graph connections.
    
    This logic is based on the GraphCast paper (page 18)[cite: 195, 196, 197]. It calculates:
    1. The Euclidean distance (length) of the edge.
    2. A 3D vector representing the sender's position relative to the receiver,
       in a coordinate system where the receiver is at a reference point ([1,0,0]).

    Args:
        sender_pos (torch.Tensor): Sender node positions in 3D, shape [E, 3].
        receiver_pos (torch.Tensor): Receiver node positions in 3D, shape [E, 3].
        
    Returns:
        torch.Tensor: Edge features, shape [E, 4].
    """
    # --- Feature 1: Edge Length ---
    edge_lengths = torch.linalg.norm(sender_pos - receiver_pos, dim=-1, keepdim=True)

    # --- Features 2-4: Local Vector Difference ---
    
    # Use SciPy's Rotation.align_vectors to find the rotation that moves
    # each receiver vector to the reference vector [1, 0, 0].
    # This is a highly efficient way to perform this operation in batch.
    rotations, _ = Rotation.align_vectors(
        a=receiver_pos.cpu().numpy(), 
        b=np.broadcast_to([1.0, 0.0, 0.0], receiver_pos.shape)
    )
    
    # Apply these rotations to the corresponding sender vectors
    sender_pos_rotated = torch.from_numpy(
        rotations.apply(sender_pos.cpu().numpy())
    ).to(sender_pos.device, dtype=torch.float32)
    
    # The reference vector that all receivers were rotated to align with
    ref_vector = torch.tensor([1.0, 0.0, 0.0], device=sender_pos.device, dtype=torch.float32)
    
    # The difference is now in the receiver's local coordinate system
    local_vector_diff = sender_pos_rotated - ref_vector

    # --- Combine all features ---
    edge_features = torch.cat([edge_lengths, local_vector_diff], dim=-1)
    
    return edge_features

def GetGraph(data, mesh_split, connection_radius_factor=0.6):
    """
    Creates a heterogeneous graph object containing grid and mesh nodes,
    and the connections between them.

    Args:
        data (xarray.Dataset or similar): 
            An object containing 'latitude' and 'longitude' coordinates 
            for the grid.
        connection_radius_factor (float, optional): 
            A factor to determine the connection radius, based on the 
            longest mesh edge. Defaults to 0.6.

    Returns:
        torch_geometric.data.HeteroData: The static graph object.
    """
    
    static_graph = HeteroData()

    # --- 1. Grid Nodes ---
    lat_coords = data.coords['latitude'].values
    lon_coords = data.coords['longitude'].values
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    grid_lat_lon = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1)
    
    # Assume lat_lon_to_cartesian returns a numpy array
    grid_pos_3d_np = lat_lon_to_cartesian(grid_lat_lon)
    # Store position as a tensor in the graph
    static_graph['grid'].pos = torch.tensor(grid_pos_3d_np, dtype=torch.float32)

    # (Optional: Grid-to-Grid edges, using the commented line as a template)
    # grid_edge_index = create_edge_index(len(lat_coords), len(lon_coords))
    # static_graph['grid', 'links', 'grid'].edge_index = grid_edge_index

    
    # --- 2. Mesh Nodes --- 
    mesh_hierarchy = get_hierarchy_of_triangular_meshes(splits=mesh_split)
    mesh = merge_meshes(mesh_hierarchy)
    mesh_vertices = mesh.vertices  # Assumed to be numpy
    faces = mesh.faces              # Assumed to be numpy

    # Mesh node features
    # ----------------- REPLACEMENT ----------------- as suggested by gemini
    mesh_pos_3d = torch.tensor(mesh_vertices, dtype=torch.float32)
    static_graph['mesh'].pos = mesh_pos_3d
    
    x_t, y_t, z_t = mesh_pos_3d.T
    r = torch.linalg.norm(mesh_pos_3d, dim=1) # Note: corrected here. changed to dim=1 from dim=0
    
    # 1. Polar angle (theta or co-latitude): angle from North Pole (z-axis)
    theta = torch.acos(z_t / r)
    # 2. Azimuthal angle (phi or longitude)
    phi = torch.atan2(y_t, x_t) 
    
    # Structural Node Features: cos(theta), cos(phi), sin(phi)
    node_features = torch.stack([
        torch.cos(theta), # Latitude feature (cos(polar angle))
        torch.cos(phi),   # Longitude feature (cos)
        torch.sin(phi)    # Longitude feature (sin)
    ], dim=1)
    
    static_graph['mesh'].x = node_features.float()
    # Mesh node position
    mesh_pos_3d = torch.tensor(mesh_vertices, dtype=torch.float32)
    static_graph['mesh'].pos = mesh_pos_3d


    # --- 3. Mesh-to-Mesh Edges --- -> changes made: added self loops
    edges = []
    for tri in faces:
        a, b, c = tri
        edges.extend([(a, b), (b, c), (c, a)])
    num_mesh_nodes = mesh_pos_3d.shape[0]
    self_loops = [(i, i) for i in range(num_mesh_nodes)]
    edges.extend(self_loops)
    edges += [(j, i) for (i, j) in edges]  # Add reverse edges
    edges = list(set(edges))  # Remove duplicates
    edge_index_mesh = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Store as a 'mesh' node type edge
    static_graph['mesh'].edge_index = edge_index_mesh
    
    # Mesh edge features
    edge_features = get_mesh_edge_features(mesh_pos_3d, edge_index_mesh)
    static_graph['mesh'].edge_attr = edge_features.float()
    
    # Also store as a ('mesh', 'to', 'mesh') edge type for message passing
    static_graph['mesh', 'to', 'mesh'].edge_index = static_graph["mesh"].edge_index
    static_graph['mesh', 'to', 'mesh'].edge_attr = static_graph["mesh"].edge_attr
    # Note: 'static_graph['mesh', 'to', 'mesh'].x' is not a valid assignment.
    # Node features (x) are stored on the node type ('mesh'), not the edge type.
    
    
    # --- 4. Grid-to-Mesh Edges ---
    mesh_senders = edge_index_mesh[0]
    mesh_receivers = edge_index_mesh[1]
    all_mesh_edge_lengths = torch.linalg.norm(
        mesh_pos_3d[mesh_senders] - mesh_pos_3d[mesh_receivers], dim=-1
    )
    max_mesh_edge_length = all_mesh_edge_lengths.max()
    connection_radius = connection_radius_factor * max_mesh_edge_length
    
    # KDTree requires numpy arrays
    tree = KDTree(mesh_vertices)
    indices_list = tree.query_radius(grid_pos_3d_np, r=connection_radius)
    
    counts = [len(indices) for indices in indices_list]
    grid_nodes = np.repeat(np.arange(len(indices_list)), counts)
    mesh_nodes = np.concatenate(indices_list)
    
    g2m_edge_index = torch.tensor(np.array([grid_nodes, mesh_nodes]), dtype=torch.long)
    static_graph['grid', 'to', 'mesh'].edge_index = g2m_edge_index
    
    
    # --- 5. Mesh-to-Grid Edges ---
    grid_tree = KDTree(grid_pos_3d_np)
    indices_list = grid_tree.query_radius(mesh_vertices, r=connection_radius)
    
    counts = [len(indices) for indices in indices_list]
    m2g_senders = np.repeat(np.arange(len(indices_list)), counts)
    m2g_receivers = np.concatenate(indices_list)
    
    m2g_edge_index = torch.tensor(np.array([m2g_senders, m2g_receivers]), dtype=torch.long)
    static_graph['mesh', 'to', 'grid'].edge_index = m2g_edge_index
    
    
    # --- 6. Heterogeneous Edge Features ---
    
    # Grid-to-Mesh edge features
    g2m_sender_pos = static_graph['grid'].pos[g2m_edge_index[0]]
    g2m_receiver_pos = static_graph['mesh'].pos[g2m_edge_index[1]]
    g2m_edge_attr = get_hetero_edge_features(g2m_sender_pos, g2m_receiver_pos)
    static_graph['grid', 'to', 'mesh'].edge_attr = g2m_edge_attr
    
    # Mesh-to-Grid edge features
    m2g_sender_pos = static_graph['mesh'].pos[m2g_edge_index[0]]
    m2g_receiver_pos = static_graph['grid'].pos[m2g_edge_index[1]] # Corrected to use graph tensor
    m2g_edge_attr = get_hetero_edge_features(m2g_sender_pos, m2g_receiver_pos)
    static_graph['mesh', 'to', 'grid'].edge_attr = m2g_edge_attr
    
    return static_graph

# data = xr.open_dataset("/storage/ashik/wbcoarse.nc")
# static_graph = GetGraph(data, mesh_split=3, connection_radius_factor=0.6)

# print(static_graph)