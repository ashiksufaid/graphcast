# README.md

***

## ⚠️ Project Status

**This repository is an ongoing project and is still under construction.** The current code provides a foundational implementation of the GraphCast model for weather forecasting.

***

## 🌪️ GraphCast Implementation

This repository contains a PyTorch Geometric (PyG) implementation of **GraphCast**, a Graph Neural Network (GNN) model designed for weather forecasting.

***

## 📁 Repository Structure

The core functionality is split into the following modular Python files:

| File Name | Description |
| :--- | :--- |
| **`main.py`** | The primary script to set up the environment, define hyperparameters, initialize the model, data loaders, optimizer, and scheduler, and start the training process. |
| **`GraphCast.py`** | Defines the complete $\text{GraphCast}$ model architecture. It includes the $\text{MLP}$ utility, $\text{InteractionNetworkLayer}$, $\text{Encoder}$, $\text{Processor}$, and $\text{Decoder}$ modules. |
| **`Graphcast_graph.py`** | Utility file for creating the $\text{static heterogeneous graph}$ structure (grid and icosahedron mesh). It handles coordinate conversions and the computation of the $\text{4D edge features}$. |
| **`WeatherGraphDataset.py`** | Defines the custom PyTorch $\text{WeatherDataset}$ class. It loads time-series weather data, applies $\text{normalization}$ based on pre-computed statistics, and prepares the $\text{input}$ and $\text{target delta}$ features for the graph. |
| **`train.py`** | Contains the core $\text{train}$ function. It implements the $\text{training and validation loop}$, handles $\text{forward/backward}$ passes, and manages saving the $\text{best}$ and $\text{final model checkpoints}$. |
