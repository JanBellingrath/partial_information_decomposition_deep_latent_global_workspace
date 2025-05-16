import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb
from pathlib import Path
import numpy as np
import traceback
import argparse
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as SklearnGMM
from typing import Optional, Any, Dict, List, Union, Tuple
import re
import pickle
import hashlib
import torch.cuda.amp as amp
from datetime import datetime
import umap
from torch.utils.data._utils.collate import default_collate
from queue import Queue
from threading import Thread, Event
import time
from dataclasses import dataclass
import queue
import warnings
from tqdm import tqdm
import numpy.core.multiarray as multiarray

# Import gmm-torch
from gmm import GaussianMixture

# Add numpy's multiarray._reconstruct to safe globals for cluster cache loading
torch.serialization.add_safe_globals([
    multiarray._reconstruct,
    np.ndarray,  # Add numpy.ndarray to safe globals
    np.dtype     # Add numpy.dtype to safe globals
])

# At the module level, simply declare the variable
# Module-level storage for wandb workers
global_args = None

# Add at module level, before any functions
# Global cache for features
CACHED_FEATURES = None

# Define a wrapper class for gmm-torch to maintain compatibility
class TorchGMM:
    """Wrapper class for sklearn's GMM with PyTorch tensor interface"""
    def __init__(self, n_components=10, random_state=None, n_init=10, max_iter=300, covariance_type='full'):
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.model = None
        
    def fit(self, X):
        """Fit GMM to data"""
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
            
        # Create and fit sklearn GMM
        self.model = SklearnGMM(
            n_components=self.n_components,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            covariance_type=self.covariance_type
        )
        self.model.fit(X)
        return self
    
    def predict_proba(self, X):
        """Get cluster probabilities"""
        if self.model is None:
            raise RuntimeError("Model must be fit before predicting")
            
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
            
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Get cluster assignments"""
        if self.model is None:
            raise RuntimeError("Model must be fit before predicting")
            
        # Convert to numpy if tensor
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
            
        return self.model.predict(X)
    
    def state_dict(self):
        """Get model state for saving"""
        if self.model is None:
            raise RuntimeError("Model must be fit before getting state")
            
        return {
            'means': torch.from_numpy(self.model.means_).float(),
            'covariances': torch.from_numpy(self.model.covariances_).float(),
            'weights': torch.from_numpy(self.model.weights_).float(),
            'precisions_cholesky': torch.from_numpy(self.model.precisions_cholesky_).float(),
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'random_state': self.random_state
        }
    
    def load_state_dict(self, state_dict):
        """Load model state"""
        self.model = SklearnGMM(
            n_components=state_dict['n_components'],
            covariance_type=state_dict['covariance_type'],
            random_state=state_dict['random_state']
        )
        
        # Convert tensors back to numpy arrays
        self.model.means_ = state_dict['means'].numpy()
        self.model.covariances_ = state_dict['covariances'].numpy()
        self.model.weights_ = state_dict['weights'].numpy()
        self.model.precisions_cholesky_ = state_dict['precisions_cholesky'].numpy()
        
        # Set additional attributes needed by sklearn GMM
        self.model.n_features_in_ = self.model.means_.shape[1]
        self.model.converged_ = True
        self.model.n_iter_ = 0
        
        return self

def _compute_precision_cholesky(covariances, covariance_type):
    """Helper function to compute precision cholesky factor"""
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    return _compute_precision_cholesky(covariances, covariance_type)

def move_features_to_device(features: Dict[str, Dict[str, torch.Tensor]], 
                            device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    """Move all feature tensors to specified device."""
    gpu_features = {}
    for split, split_data in features.items():
        gpu_features[split] = {}
        for key, tensor in split_data.items():
            if isinstance(tensor, torch.Tensor):
                gpu_features[split][key] = tensor.to(device)
            else:
                gpu_features[split][key] = tensor
    return gpu_features

# Custom KMeans class that just implements predict with numpy
class CustomKMeans:
    def __init__(self, centers):
        self.cluster_centers_ = centers
    
    def predict(self, X):
        # Convert input to numpy if it's a torch tensor
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        # Calculate distances to all centers
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        # Return index of closest center for each point
        return np.argmin(distances, axis=1)

def set_global_args(args):
    """Set the global arguments used by the training function.
    
    Args:
        args: ArgumentParser args containing domain checkpoints, dataset paths, etc.
    """
    global global_args  # This is the correct place for the global statement
    global_args = args

class LatentDataset(Dataset):
    """Dataset for loading cached latent features."""
    def __init__(self, v_features, t_features, labels=None):
        self.v_features = v_features
        self.t_features = t_features
        self._labels = labels  # Use private attribute

    @property
    def labels(self):
        """Get the labels."""
        return self._labels

    def __len__(self):
        return len(self.v_features)

    def __getitem__(self, idx):
        v = self.v_features[idx]
        t = self.t_features[idx]
        if self._labels is not None:
            return v, t, self._labels[idx]
        return v, t

class SoftLatentDataset(LatentDataset):
    """Returns (v, t, soft_probs) instead of hard labels."""
    def __init__(self, v_features, t_features, probs):
        super().__init__(v_features, t_features, labels=None)
        self._probs = probs  # Tensor[N, K]

    @property
    def labels(self):
        """Get the soft labels (probabilities)."""
        return self._probs

    def __getitem__(self, idx):
        v = self.v_features[idx]
        t = self.t_features[idx]
        return v, t, self._probs[idx]  # FloatTensor[K]

class GPUBatchedDataset(Dataset):
    """Dataset that keeps all data on GPU and handles shuffling efficiently."""
    def __init__(self, v_features, t_features, labels):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.v_features = v_features.to(device)
        self.t_features = t_features.to(device)
        self.labels = labels.to(device)
        self.length = len(labels)
        self.indices = torch.arange(self.length, device=device)
        self.epoch = 0
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self.indices.device)
        return self.v_features[idx], self.t_features[idx], self.labels[idx]
    
    def set_epoch(self, epoch):
        """Update epoch counter to trigger shuffling."""
        if epoch != self.epoch:
            self.epoch = epoch
            # Shuffle indices on device
            perm = torch.randperm(self.length, device=self.indices.device)
            self.indices = self.indices[perm]
            # Shuffle data on device
            self.v_features = self.v_features[perm]
            self.t_features = self.t_features[perm]
            self.labels = self.labels[perm]

# Import from analyze_pid_new
from analyze_pid_new import (
    MultimodalDataset,
    prepare_pid_data,
    load_checkpoint,
    generate_samples_from_model,
    generate_samples_from_dataset_fixed,
    Discrim,
    load_domain_modules,
    process_batch,
)

def process_domain_value(value: torch.Tensor, domain_name: str, device: torch.device) -> torch.Tensor:
    """Process a single domain value by moving it to device and handling special cases.
    
    Args:
        value: Input tensor to process
        domain_name: Name of the domain being processed
        device: Target device for the tensor
        
    Returns:
        Processed tensor on the specified device
    """
    # Move to device and handle v_latents special case
    value = value.to(device)
    if domain_name == "v_latents" and value.dim() > 2:
        return value[:, 0, :]
    return value

def process_domain_data_vectorized(domain_data: Any, domain_name: str) -> torch.Tensor:
    """Vectorized version of process_domain_data."""
    # Handle dictionary format
    if isinstance(domain_data, dict):
        data = domain_data.get(domain_name, None)
        if data is None:
            # Try frozenset keys
            for k, v in domain_data.items():
                if isinstance(k, frozenset) and domain_name in k:
                    data = v
                    break
            if data is None:
                data = domain_data
    else:
        data = domain_data

    # Handle tensor data directly first - this is important for already processed data
    if isinstance(data, torch.Tensor):
        return data

    # Handle BERT embeddings
    if hasattr(data, 'bert'):
        return data.bert
    
    # Handle dictionary with domain_name key
    if isinstance(data, dict) and domain_name in data:
        value = data[domain_name]
        if hasattr(value, 'bert'):
            return value.bert
        elif isinstance(value, torch.Tensor):
            return value
        
    # Handle Attribute objects - vectorized version
    if hasattr(data, 'category'):
        # Pre-allocate tensor for attributes
        attrs = torch.empty(8, dtype=torch.float32)
        # Fill attributes in one go
        attrs[0] = float(data.category if not isinstance(data.category, torch.Tensor) else data.category)
        attrs[1] = float(data.x if not isinstance(data.x, torch.Tensor) else data.x)
        attrs[2] = float(data.y if not isinstance(data.y, torch.Tensor) else data.y)
        attrs[3] = float(data.size if not isinstance(data.size, torch.Tensor) else data.size)
        attrs[4] = float(data.rotation if not isinstance(data.rotation, torch.Tensor) else data.rotation)
        attrs[5] = float(data.color_r if not isinstance(data.color_r, torch.Tensor) else data.color_r)
        attrs[6] = float(data.color_g if not isinstance(data.color_g, torch.Tensor) else data.color_g)
        attrs[7] = float(data.color_b if not isinstance(data.color_b, torch.Tensor) else data.color_b)
        return attrs

    # Handle list data - vectorized version
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty list for {domain_name}")
        
        # Pre-allocate list for tensors
        tensors = []
        
        # Process items in batch
        for item in data:
            if isinstance(item, dict):
                if domain_name in item:
                    value = item[domain_name]
                    tensors.append(value.bert if hasattr(value, 'bert') else value)
                else:
                    # Find first tensor or bert attribute
                    for v in item.values():
                        if hasattr(v, 'bert'):
                            tensors.append(v.bert)
                            break
                        elif isinstance(v, torch.Tensor):
                            tensors.append(v)
                            break
            elif hasattr(item, 'bert'):
                tensors.append(item.bert)
            elif isinstance(item, torch.Tensor):
                tensors.append(item)
            elif hasattr(item, 'category'):
                # Vectorized attribute processing
                attrs = torch.tensor([
                    float(item.category if not isinstance(item.category, torch.Tensor) else item.category),
                    float(item.x if not isinstance(item.x, torch.Tensor) else item.x),
                    float(item.y if not isinstance(item.y, torch.Tensor) else item.y),
                    float(item.size if not isinstance(item.size, torch.Tensor) else item.size),
                    float(item.rotation if not isinstance(item.rotation, torch.Tensor) else item.rotation),
                    float(item.color_r if not isinstance(item.color_r, torch.Tensor) else item.color_r),
                    float(item.color_g if not isinstance(item.color_g, torch.Tensor) else item.color_g),
                    float(item.color_b if not isinstance(item.color_b, torch.Tensor) else item.color_b)
                ])
                tensors.append(attrs)
        
        if not tensors:
            raise ValueError(f"Could not extract any valid tensors for {domain_name}")
        
        # Stack all tensors at once
        return torch.stack(tensors)

    raise ValueError(f"Unsupported data type for {domain_name}: {type(data)}")

def process_batch_vectorized(batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """Vectorized version of process_batch."""
    processed: Dict[str, torch.Tensor] = {}
    
    def process_dict(d: Dict[Any, Any]) -> None:
        # Process all items in the dictionary at once
        for k, v in d.items():
            domain_name = next(iter(k)) if isinstance(k, frozenset) else k
            
            # Handle BERT embeddings
            if hasattr(v, 'bert'):
                processed[domain_name] = process_domain_value(v.bert, domain_name, device)
            # Handle nested dictionary
            elif isinstance(v, dict) and domain_name in v:
                value = v[domain_name]
                if hasattr(value, 'bert'):
                    processed[domain_name] = process_domain_value(value.bert, domain_name, device)
                elif isinstance(value, torch.Tensor):
                    processed[domain_name] = process_domain_value(value, domain_name, device)
                else:
                    processed[domain_name] = process_domain_value(
                        process_domain_data_vectorized(value, domain_name), 
                        domain_name, 
                        device
                    )
            # Handle tensor directly
            elif isinstance(v, torch.Tensor):
                processed[domain_name] = process_domain_value(v, domain_name, device)
            # Handle other types
            else:
                processed[domain_name] = process_domain_value(
                    process_domain_data_vectorized(v, domain_name),
                    domain_name,
                    device
                )
    
    # Handle tuple format
    if isinstance(batch, tuple):
        for item in batch:
            if isinstance(item, dict):
                process_dict(item)
    
    # Handle list format
    elif isinstance(batch, list) and batch:
        if isinstance(batch[0], dict):
            for item in batch:
                process_dict(item)
    
    # Handle dictionary format
    elif isinstance(batch, dict):
        process_dict(batch)
    
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    return processed

# Replace original functions with vectorized versions
process_batch = process_batch_vectorized
process_domain_data = process_domain_data_vectorized

# Patch analyze_pid_new to use our implementations
import analyze_pid_new
analyze_pid_new.process_batch = process_batch 
analyze_pid_new.process_domain_data = process_domain_data

# Try to import shimmer and dataset modules
try:
    # Add simple-shapes-dataset to path first
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple-shapes-dataset"))
    
    # Import shimmer modules
    from shimmer.modules.domain import DomainModule
    from shimmer.modules.gw_module import GWModule
    from gw_module_configurable_fusion import GWModuleConfigurableFusion
    from shimmer_ssd.config import LoadedDomainConfig, DomainModuleVariant
    from shimmer_ssd.modules.domains.pretrained import load_pretrained_module
    from shimmer_ssd.ckpt_migrations import migrate_model
    
    # Import dataset modules (after path append)
    from simple_shapes_dataset.data_module import SimpleShapesDataModule
    from simple_shapes_dataset.domain import DomainDesc, SimpleShapesPretrainedVisual, SimpleShapesText
    
    SHIMMER_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Required shimmer or dataset modules not found: {e}. This script requires real data and will not run with dummy or synthetic data.")

# Add ML imports after other imports
import warnings

# Import gmm-torch for Gaussian Mixture Models
from gmm import GaussianMixture

def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Sweep configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['method', 'metric', 'parameters']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in sweep config: {missing_fields}")
    
    # Check and validate early termination configuration
    if 'early_terminate' not in config:
        print("\nWarning: No early_terminate configuration found in sweep config.")
        print("No pruning will occur. Consider adding early termination settings:")
        print("early_terminate:")
        print("  type: asha")
        print("  min_iter: 5")
        print("  reduction_factor: 3")
    else:
        # Validate early termination settings if present
        early_term = config['early_terminate']
        if not isinstance(early_term, dict):
            raise ValueError("early_terminate must be a dictionary")
        
        # Check type-specific required fields
        if early_term.get('type') == 'asha':
            required_asha = ['min_iter', 'reduction_factor']
            missing_asha = [field for field in required_asha if field not in early_term]
            if missing_asha:
                raise ValueError(f"ASHA early termination requires: {missing_asha}")
        elif early_term.get('type') == 'hyperband':
            required_hyperband = ['min_iter', 'max_iter', 'eta']
            missing_hyperband = [field for field in required_hyperband if field not in early_term]
            if missing_hyperband:
                raise ValueError(f"Hyperband early termination requires: {missing_hyperband}")
        else:
            raise ValueError("early_terminate.type must be either 'asha' or 'hyperband'")
    
    return config

def setup_wandb_sweep(config_path: str, project: str = 'pid-classifier-sweep') -> str:
    """Initialize wandb sweep from config file."""
    # Load and validate sweep configuration
    sweep_config = load_sweep_config(config_path)
    
    # Override project name if specified in config
    if 'project' in sweep_config:
        project = sweep_config['project']
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Sweep configuration loaded from: {config_path}")
    
    return sweep_id

def setup_data_module(dataset_path: str, batch_size: int, domain_modules: dict, 
                     train_size: Optional[int] = None,
                     val_size: Optional[int] = None,
                     test_size: Optional[int] = None) -> SimpleShapesDataModule:
    """Set up the data module with proper configuration."""
    print(f"Setting up data module with path: {dataset_path}")
    print(f"Dataset split sizes - Train: {train_size or 'all'}, Val: {val_size or 'all'}, Test: {test_size or 'all'}")
    
    # Create domain classes and args
    domain_classes = {}
    domain_args = {}
    
    # Set up domain classes based on loaded modules
    for domain_name, domain_module in domain_modules.items():
        if domain_name == "v_latents":
            domain_classes[DomainDesc(base="v", kind="v_latents")] = SimpleShapesPretrainedVisual
            domain_args["v_latents"] = {
                "presaved_path": "calmip-822888_epoch=282-step=1105680_future.npy",
                "use_unpaired": False,
                # Add size limits to domain args
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size
            }
        elif domain_name == "t":
            domain_classes[DomainDesc(base="t", kind="t")] = SimpleShapesText
            # Add size limits to text domain args too
            domain_args["t"] = {
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size
            }
    
    # Define domain proportions
    domain_proportions = {}
    for domain_name in domain_modules.keys():
        domain_proportions[frozenset([domain_name])] = 1.0
    
    # Create custom collate function
    def custom_collate_fn(batch):
        """Custom collate function to handle variable-sized batches."""
        from torch.utils.data._utils.collate import default_collate
        try:
            # Handle dictionary batches
            if isinstance(batch, dict):
                return {k: custom_collate_fn(v) for k, v in batch.items()}
            
            # Handle list batches
            if isinstance(batch, list):
                # If the list contains dictionaries, process each dictionary
                if len(batch) > 0 and isinstance(batch[0], dict):
                    # Get all keys from all dictionaries
                    all_keys = set().union(*[d.keys() for d in batch])
                    
                    # Create a dictionary with lists of values for each key
                    result = {}
                    for key in all_keys:
                        values = []
                        for d in batch:
                            if key in d:
                                values.append(d[key])
                        if values:
                            try:
                                result[key] = default_collate(values)
                            except Exception:
                                result[key] = values
                    return result
                
                # Try default collation
                try:
                    return default_collate(batch)
                except Exception:
                    return batch
            
            # For other types, try default collation
            try:
                return default_collate(batch)
            except Exception:
                return batch
            
        except Exception as e:
            print(f"Error in collate_fn: {e}")
            return batch
    
    # Create data module with no workers and custom collate function
    data_module = SimpleShapesDataModule(
        dataset_path=dataset_path,
        domain_classes=domain_classes,
        domain_proportions=domain_proportions,
        batch_size=batch_size,
        num_workers=0,  # No workers to avoid shared memory issues
        seed=42,
        domain_args=domain_args,
        collate_fn=custom_collate_fn
    )
    
    # Setup data module
    data_module.setup()
    
    # Print dataset information
    print("\nDataset Information:")
    for domain, dataset in data_module.train_dataset.items():
        print(f"Train domain {domain}: {len(dataset)} samples")
    if data_module.val_dataset:
        for domain, dataset in data_module.val_dataset.items():
            print(f"Val domain {domain}: {len(dataset)} samples")
    if data_module.test_dataset:
        for domain, dataset in data_module.test_dataset.items():
            print(f"Test domain {domain}: {len(dataset)} samples")
    
    return data_module

def compute_and_cache_clusters(data, num_clusters, method='kmeans', random_state=42, normalize=False, cache_dir=None):
    """
    Compute clusters once and cache the results.
    
    Args:
        data: Data to cluster
        num_clusters: Number of clusters to create
        method: Clustering method ('kmeans' or 'gmm')
        random_state: Random seed
        normalize: Whether to normalize data
        cache_dir: Directory to save cache
        
    Returns:
        clusterer: Fitted clustering model
    """
    if cache_dir is None:
        cache_dir = './cluster_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create fingerprint based on data shape and content hash
    shape = data.shape
    # Get a stable hash of the data content
    data_content = data.cpu().numpy()
    data_hash = hashlib.sha256(data_content.tobytes()).hexdigest()[:8]
    
    # Create cache filename using stable parameters
    cache_name = f"clusters_{method}_{num_clusters}_{shape[0]}x{shape[1]}_{data_hash}_{random_state}_{normalize}.pt"
    cache_path = os.path.join(cache_dir, cache_name)
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"\nLoading cached clusters from: {cache_path}")
        return torch.load(cache_path)
    
    print(f"\nComputing clusters (this will be done only once)...")
    
    # Convert to numpy and normalize if requested
    data_np = data.cpu().numpy()
    if normalize:
        scaler = StandardScaler()
        data_np = scaler.fit_transform(data_np)
    
    # Create and fit clusterer
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(
            n_clusters=num_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        clusterer.fit(data_np)
    else:  # method == 'gmm'
        # Create and fit TorchGMM
        clusterer = TorchGMM(
            n_components=num_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            covariance_type='full'
        )
        
        # Convert data back to tensor if normalized
        if normalize:
            data_np = torch.from_numpy(data_np).float()
            if torch.cuda.is_available():
                data_np = data_np.cuda()
        
        # Fit the model
        clusterer.fit(data_np)
    
    # Save to cache
    print(f"Saving clusters to: {cache_path}")
    torch.save(clusterer, cache_path)
    
    return clusterer

def precompute_and_cache_features(data_module, domain_modules, device, batch_size, cache_dir, fusion_model=None):
    """
    Precompute and cache domain features efficiently using the generate_samples_from_dataset_fixed function
    from analyze_pid_new.py.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create model fingerprint based on state dicts
    checksums = {}
    for domain_name, module in domain_modules.items():
        state_bytes = pickle.dumps(module.state_dict())
        checksums[domain_name] = hashlib.sha256(state_bytes).hexdigest()[:8]
    
    # Add fusion model checksum if provided
    if fusion_model is not None:
        state_bytes = pickle.dumps(fusion_model.state_dict())
        checksums['fusion'] = hashlib.sha256(state_bytes).hexdigest()[:8]
    
    # Create cache filename based on model fingerprints
    cache_name = f"latents_{'_'.join(f'{k}_{v}' for k,v in sorted(checksums.items()))}.pt"
    cache_path = os.path.join(cache_dir, cache_name)
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"\nLoading cached features from: {cache_path}")
        return torch.load(cache_path)
    
    print("\nPrecomputing features (this will be done only once)...")
    
    # Ensure all domain modules are in eval mode
    for module in domain_modules.values():
        module.eval()
    
    # Get domain names
    domain_names = list(domain_modules.keys())
    if len(domain_names) != 2:
        raise ValueError(f"Exactly 2 domains required, got {domain_names}")
    
    cached_features = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get the number of samples for this split
        # First try CLI args from global_args
        if global_args is not None:
            n_samples = getattr(global_args, f"{split}_size", None)
        else:
            n_samples = None
            
        # If no CLI arg specified, get the actual dataset size
        if n_samples is None:
            try:
                dataset = getattr(data_module, f"{split}_dataset")
                if isinstance(dataset, dict):  # Handle case where dataset is a dict of domain datasets
                    # Use size of first domain's dataset
                    first_domain_dataset = next(iter(dataset.values()))
                    n_samples = len(first_domain_dataset)
                else:
                    n_samples = len(dataset)
                print(f"Using full {split} dataset size: {n_samples}")
            except (AttributeError, TypeError) as e:
                print(f"Warning: Could not determine {split} dataset size: {e}")
                print("Will use all available samples")
                n_samples = None
        else:
            print(f"Using CLI-specified {split} size: {n_samples}")
        
        # Use generate_samples_from_dataset_fixed from analyze_pid_new to properly handle domains
        if fusion_model is not None:
            # If fusion model exists, use it
            samples = generate_samples_from_dataset_fixed(
                model=fusion_model,
                data_module=data_module,
                domain_names=domain_names,
                split=split,
                n_samples=n_samples,  # Now using proper split size
                batch_size=batch_size,
                device=device,
                use_gw_encoded=False  # Use raw latents
            )
            
            # Store the samples - this includes domain latents, GW encoded, and decoded representations
            cached_features[split] = {
                'v_latents': samples.get(f"{domain_names[0]}_latent", samples.get(f"{domain_names[0]}_gw_encoded")),
                't': samples.get(f"{domain_names[1]}_latent", samples.get(f"{domain_names[1]}_gw_encoded")),
                'gw_rep': samples.get("gw_rep")
            }
        else:
            # If no fusion model, just get the encoded representations from each domain module
            # First, create containers for encodings
            all_encodings = {}
            for domain_name in domain_names:
                all_encodings[domain_name] = []
            
            # Get the appropriate dataloader for this split
            if split == 'train':
                dataloader = data_module.train_dataloader()
            elif split == 'val':
                dataloader = data_module.val_dataloader()
            elif split == 'test':
                dataloader = data_module.test_dataloader()
            else:
                raise ValueError(f"Unknown split: {split}")
            
            # Process batches and encode with domain modules
            total_samples = 0
            max_samples = n_samples if n_samples is not None else float('inf')
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx % 10 == 0:
                        print(f"Processing batch {batch_idx}")
                    
                    try:
                        # Process batch using analyze_pid_new.process_batch
                        processed_batch = process_batch(batch, device)
                        
                        # Check if we have all required domains
                        if not all(domain in processed_batch for domain in domain_names):
                            print(f"Skipping batch {batch_idx} - missing domains")
                            continue
                        
                        # Get batch size and check if we would exceed max_samples
                        batch_size = next(iter(processed_batch.values())).size(0)
                        if total_samples + batch_size > max_samples:
                            # Truncate batch to exactly reach max_samples
                            samples_needed = max_samples - total_samples
                            for domain_name in processed_batch:
                                processed_batch[domain_name] = processed_batch[domain_name][:samples_needed]
                            batch_size = samples_needed
                        
                        # Encode each domain
                        for domain_name, domain_module in domain_modules.items():
                            if domain_name not in processed_batch:
                                continue
                            
                            domain_data = processed_batch[domain_name]
                            
                            # Handle text domain specially - it expects a dict with 'bert' key
                            if domain_name == 't':
                                encoded_data = domain_module.encode({'bert': domain_data})
                            else:
                                encoded_data = domain_module.encode(domain_data)
                            
                            # Handle (mean, var) tuple if returned
                            if isinstance(encoded_data, tuple):
                                encoded_data = encoded_data[0]  # Take only the mean
                            
                            all_encodings[domain_name].append(encoded_data.cpu())
                        
                        # Update total samples count
                        total_samples += batch_size
                        
                        # Stop if we've reached max_samples
                        if total_samples >= max_samples:
                            print(f"Reached requested number of samples ({max_samples})")
                            break
                        
                        # Periodically clear cache
                        if batch_idx % 50 == 0:
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue
            
            # Concatenate all encodings
            try:
                concatenated_encodings = {}
                for domain_name, encodings in all_encodings.items():
                    if encodings:  # Check if there are any encodings
                        concatenated_encodings[domain_name] = torch.cat(encodings, dim=0)
                    else:
                        print(f"Warning: No encodings for domain {domain_name}")
                
                # Store the encodings - no fusion model, so no GW rep
                cached_features[split] = {
                    'v_latents': concatenated_encodings.get('v_latents'),
                    't': concatenated_encodings.get('t'),
                    'gw_rep': None
                }
            except Exception as e:
                print(f"Error concatenating encodings for {split}: {e}")
                continue
        
        print(f"Processed {split} split - shapes:")
        if cached_features[split]['v_latents'] is not None:
            print(f"Visual features: {cached_features[split]['v_latents'].shape}")
        if cached_features[split]['t'] is not None:
            print(f"Text features: {cached_features[split]['t'].shape}")
        if cached_features[split]['gw_rep'] is not None:
            print(f"GW representation: {cached_features[split]['gw_rep'].shape}")
    
    # Verify we have required splits
    missing_splits = {'train'} - set(cached_features.keys())
    if missing_splits:
        raise ValueError(f"Missing required split: train")
    
    # Save cache
    print(f"\nSaving feature cache to {cache_path}")
    torch.save(cached_features, cache_path)
    
    return cached_features

def compute_sweep_clusters(precomputed_data, sweep_id, num_clusters=10, cluster_method='kmeans', cluster_seed=42, normalize_inputs=False):
    """Compute clusters once at the start of a sweep and save them for all splits."""
    
    # Create sweep-specific clusters directory
    sweep_clusters_dir = os.path.join('sweep_clusters', sweep_id)
    os.makedirs(sweep_clusters_dir, exist_ok=True)
    
    # Create a config string that uniquely identifies the clustering setup
    config_str = f"clusters{num_clusters}_method{cluster_method}_seed{cluster_seed}_norm{normalize_inputs}"
    clusters_path = os.path.join(sweep_clusters_dir, f'cluster_data_{config_str}.pt')
    
    # Check if clusters already exist for this sweep with these exact parameters
    if os.path.exists(clusters_path):
        print(f"\nLoading pre-computed sweep clusters from: {clusters_path}")
        try:
            cluster_data = load_cluster_data(clusters_path)
            print(f"[DEBUG] Loaded cluster_data keys: {list(cluster_data.keys())}")
            
            # Create appropriate clusterer based on method
            if cluster_method == 'kmeans':
                clusterer = CustomKMeans(cluster_data['centers'])
            else:  # GMM
                print("[DEBUG] Creating TorchGMM from saved parameters")
                clusterer = TorchGMM(
                    n_components=num_clusters,
                    random_state=cluster_seed,
                    n_init=1,
                    max_iter=300,
                    covariance_type='full'
                )
                clusterer.load_state_dict(cluster_data['gmm_state'])
                print("[DEBUG] Successfully created TorchGMM clusterer")
            
            # Get train labels and probs
            train_labels = cluster_data.get('train_labels')
            train_probs = cluster_data.get('train_probs')
            
            print(f"[DEBUG] After loading, train_labels is None: {train_labels is None}")
            print(f"[DEBUG] After loading, train_probs is None: {train_probs is None}")
            
            if train_labels is None or (cluster_method == 'gmm' and train_probs is None):
                print("Labels or probabilities not found in saved cluster data, regenerating...")
                # Get training data
                train_data = precomputed_data['train']
                train_gw = train_data['gw_rep']
                if train_gw.is_cuda:
                    train_gw = train_gw.cpu()
                
                # Apply normalization if needed
                if normalize_inputs and 'normalization_params' in cluster_data:
                    norm_params = cluster_data['normalization_params']
                    train_gw = (train_gw - norm_params['mean']) / norm_params['scale']
                
                # Generate labels using the loaded clusterer
                if cluster_method == 'kmeans':
                    train_labels = torch.tensor(clusterer.predict(train_gw.numpy()), dtype=torch.long)
                else:  # GMM
                    # Always regenerate both probabilities and labels for GMM if either is missing
                    train_probs = torch.from_numpy(clusterer.predict_proba(train_gw.numpy())).float()
                    train_labels = train_probs.argmax(dim=1)
                    
                # Update the saved cluster data
                cluster_data['train_labels'] = train_labels
                if train_probs is not None:
                    cluster_data['train_probs'] = train_probs
                    
                # Debug info about the data we're saving
                print(f"[DEBUG] Generated new train_labels shape: {train_labels.shape}, type: {type(train_labels)}")
                if train_probs is not None:
                    print(f"[DEBUG] Generated new train_probs shape: {train_probs.shape}, type: {type(train_probs)}")
                    
                # Save updated data
                save_cluster_data(cluster_data, clusters_path)
                print(f"Updated clusters saved to: {clusters_path}")
            
            # Final check to confirm we have valid data to return
            if train_labels is None:
                raise ValueError("Critical error: train_labels is still None after regeneration attempts")
            
            if cluster_method == 'gmm' and train_probs is None:
                raise ValueError("Critical error: train_probs is still None for GMM after regeneration attempts")
            
            return clusterer, train_labels, train_probs, cluster_data.get('normalization_params')
            
        except Exception as e:
            print(f"[DEBUG] Error loading cluster cache: {e}")
            # If loading fails for any reason, remove the corrupted cache and recompute
            os.remove(clusters_path)
            print(f"Removed corrupted cluster cache, will recompute")
    
    print(f"\nComputing clusters for sweep {sweep_id} (this will be done only once)...")
    
    # Get training data
    if 'train' not in precomputed_data:
        raise ValueError("Training split required for computing clusters")
    
    train_data = precomputed_data['train']
    if 'gw_rep' not in train_data:
        raise ValueError("GW representation required for clustering")
    
    train_gw = train_data['gw_rep']
    if train_gw.is_cuda:
        train_gw = train_gw.cpu()
    
    # Initialize normalization if needed
    normalization_params = None
    if normalize_inputs:
        mean = train_gw.mean(dim=0)
        std = train_gw.std(dim=0)
        normalization_params = {
            'mean': mean,
            'scale': std
        }
        # Apply normalization
        train_gw = (train_gw - mean) / std
    
    # Compute clusters based on method
    if cluster_method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=cluster_seed,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(train_gw.numpy())
        
        clusterer = CustomKMeans(kmeans.cluster_centers_)
        
        cluster_data = {
            'centers': torch.from_numpy(kmeans.cluster_centers_).float(),
            'train_labels': torch.from_numpy(kmeans.labels_).long(),
            'normalization_params': normalization_params
        }
        train_probs = None
        train_labels = cluster_data['train_labels']
        
    else:  # GMM
        print("[DEBUG] Using TorchGMM for clustering")
        # Create and fit TorchGMM
        clusterer = TorchGMM(
            n_components=num_clusters,
            random_state=cluster_seed,
            n_init=10,
            max_iter=300,
            covariance_type='full'
        )
        
        try:
            print("[DEBUG] Fitting TorchGMM...")
            clusterer.fit(train_gw.numpy())  # Convert to numpy for sklearn GMM
            print("[DEBUG] TorchGMM fit successful")
            
            # Get probabilities and predictions
            print("[DEBUG] Computing probabilities and predictions...")
            train_probs = torch.from_numpy(clusterer.predict_proba(train_gw.numpy())).float()
            train_labels = train_probs.argmax(dim=1)
            
            print("[DEBUG] Successfully computed probabilities and predictions")
            
            # Save all necessary GMM attributes
            cluster_data = {
                'gmm_state': clusterer.state_dict(),
                'train_probs': train_probs,
                'train_labels': train_labels,
                'normalization_params': normalization_params
            }
            
            print("[DEBUG] TorchGMM clustering completed successfully")
            
        except Exception as e:
            print(f"[DEBUG] Error during TorchGMM: {e}")
            traceback.print_exc()
            raise
    
    # Final safety check to ensure we have valid data to return
    if train_labels is None:
        raise ValueError("Critical error: train_labels is None after clustering")
        
    if cluster_method == 'gmm' and train_probs is None:
        raise ValueError("Critical error: train_probs is None for GMM after clustering")
    
    # Save cluster data
    try:
        save_cluster_data(cluster_data, clusters_path)
        print(f"Saved sweep clusters to: {clusters_path}")
        
        # Double-check the saved file
        if os.path.exists(clusters_path):
            saved_size = os.path.getsize(clusters_path)
            print(f"[DEBUG] Saved file size: {saved_size} bytes")
            if saved_size < 1000:  # Suspiciously small file
                print("[DEBUG] Warning: Saved file is suspiciously small")
    except Exception as e:
        print(f"[DEBUG] Error saving cluster data: {e}")
        # If we can't save, we still return the computed values
    
    return clusterer, train_labels, train_probs, normalization_params

def get_datasets(precomputed_data, num_clusters=10, cluster_method='kmeans', cluster_seed=42, normalize_inputs=False):
    """Create datasets from precomputed features using fixed sweep clusters.
    
    Returns:
        tuple: (clusterer, train_ds, val_ds, test_ds) - The fitted clusterer and datasets for each split
    """
    # Get sweep ID
    if not wandb.run or not wandb.run.sweep_id:
        raise RuntimeError("This function must be run as part of a wandb sweep")
    sweep_id = wandb.run.sweep_id
    
    # Get or compute fixed sweep clusters
    clusterer, train_labels, train_probs, normalization_params = compute_sweep_clusters(
        precomputed_data=precomputed_data,
        sweep_id=sweep_id,
        num_clusters=num_clusters,
        cluster_method=cluster_method,
        cluster_seed=cluster_seed,
        normalize_inputs=normalize_inputs
    )
    
    # Debug: verify train_labels and train_probs are not None
    print(f"\n[DEBUG] train_labels is None: {train_labels is None}")
    if train_labels is not None:
        print(f"[DEBUG] train_labels shape: {train_labels.shape}, type: {type(train_labels)}")
    
    print(f"[DEBUG] train_probs is None: {train_probs is None}")
    if train_probs is not None and cluster_method == 'gmm':
        print(f"[DEBUG] train_probs shape: {train_probs.shape}, type: {type(train_probs)}")
    
    # Process splits and create datasets
    splits = ['train', 'val', 'test']
    datasets = {}
    
    # Verify all required splits are present
    missing_splits = set(splits) - set(precomputed_data.keys())
    if missing_splits:
        raise ValueError(f"Missing required splits in precomputed data: {missing_splits}")
    
    # Initialize text feature normalization if needed
    text_norm_params = None
    if normalize_inputs:
        t_scaler = StandardScaler()
        t_features_train = precomputed_data['train']['t'].cpu().numpy()
        t_scaler.fit(t_features_train)
        text_norm_params = {
            'mean': torch.from_numpy(t_scaler.mean_).float(),
            'scale': torch.from_numpy(t_scaler.scale_).float()
        }
    
    for split in splits:
        split_data = precomputed_data[split]
        
        # Get features for both domains
        v_features = split_data['v_latents'].cpu() if split_data['v_latents'].is_cuda else split_data['v_latents']
        t_features = split_data['t'].cpu() if split_data['t'].is_cuda else split_data['t']
        
        # Normalize text features if needed
        if normalize_inputs and text_norm_params is not None:
            t_features = (t_features - text_norm_params['mean']) / text_norm_params['scale']
        
        if cluster_method == 'kmeans':
            # For validation and test splits, use the saved clusterer to predict labels
            if split != 'train':
                gw_features = split_data['gw_rep'].cpu() if split_data['gw_rep'].is_cuda else split_data['gw_rep']
                # Normalize GW features if needed using saved parameters
                if normalize_inputs and normalization_params is not None:
                    gw_features = (gw_features - normalization_params['mean']) / normalization_params['scale']
                # Use clusterer to predict labels
                labels = torch.tensor(clusterer.predict(gw_features.numpy()), dtype=torch.long)
            else:
                # For training split, use the pre-computed labels
                labels = train_labels
            
            # Create dataset with hard labels
            datasets[split] = LatentDataset(
                v_features=v_features,
                t_features=t_features,
                labels=labels
            )
        else:  # GMM
            gw_features = split_data['gw_rep'].cpu() if split_data['gw_rep'].is_cuda else split_data['gw_rep']
            # Normalize GW features if needed
            if normalize_inputs and normalization_params is not None:
                gw_features = (gw_features - normalization_params['mean']) / normalization_params['scale']
            
            # For training split, use pre-computed probabilities
            if split == 'train':
                probs = train_probs
                if probs is None:
                    print(f"[DEBUG] Critical error: train_probs is None for training split in GMM mode")
                    # Regenerate probs as a last resort
                    probs = torch.from_numpy(clusterer.predict_proba(gw_features.numpy())).float()
            else:
                # For val/test, compute soft assignments using GMM
                probs = torch.from_numpy(clusterer.predict_proba(gw_features.numpy())).float()
            
            # Create dataset with soft probabilities
            datasets[split] = SoftLatentDataset(
                v_features=v_features,
                t_features=t_features,
                probs=probs
            )
            
            # Safety check - verify dataset was created with valid probs
            if probs is None:
                print(f"[DEBUG] Critical error: probs is still None for {split} split")
    
    if not datasets:
        raise ValueError("No datasets were created")
    
    # Return clusterer along with the datasets
    return clusterer, datasets['train'], datasets['val'], datasets['test']

def create_discriminator(input_dim, hidden_dim, num_layers, activation, num_labels, dropout_rate=0.1):
    """Create a discriminator network.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of hidden layers
        activation (str): Activation function name
        num_labels (int): Number of output classes (from clustering)
        dropout_rate (float, optional): Dropout probability. Defaults to 0.1
    
    Returns:
        nn.Sequential: The discriminator network
    """
    layers = []
    current_dim = input_dim
    
    # Add hidden layers
    for _ in range(num_layers):
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout_rate)
        ])
        current_dim = hidden_dim
    
    # Add output layer with correct number of classes
    layers.append(nn.Linear(hidden_dim, num_labels))
    
    return nn.Sequential(*layers)

def get_activation(name):
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU
    }
    
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}. Choose from {list(activations.keys())}")
    
    # Initialize with proper parameters
    if name == 'gelu':
        return nn.GELU(approximate='tanh')  # Use tanh approximation for better numerical stability
    else:
        return activations[name]()

def get_optimizer(name, parameters, lr, weight_decay, momentum=0.9):
    """Get optimizer by name."""
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def get_scheduler(name, optimizer, num_epochs, step_size=10, gamma=0.1, warmup_epochs=5):
    """Get learning rate scheduler by name."""
    try:
        if name == 'none':
            return None
        elif name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif name == 'cosine':
            # Check PyTorch version for LinearLR and SequentialLR support
            if not hasattr(torch.optim.lr_scheduler, 'LinearLR') or not hasattr(torch.optim.lr_scheduler, 'SequentialLR'):
                print("\nWarning: PyTorch version < 1.11 detected. Using CosineAnnealingLR without warmup.")
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs,
                    eta_min=1e-7
                )
            
            # Use CosineAnnealingLR with optional linear warmup
            if warmup_epochs > 0:
                # Create scheduler list with warmup + cosine annealing
                scheduler1 = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,  # Start at 10% of base lr
                    end_factor=1.0,    # End at base lr
                    total_iters=warmup_epochs
                )
                scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs - warmup_epochs,  # Remaining epochs after warmup
                    eta_min=1e-7  # Minimum learning rate
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[scheduler1, scheduler2],
                    milestones=[warmup_epochs]  # Switch from warmup to cosine at this epoch
                )
            else:
                # Just use cosine annealing without warmup
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs,
                    eta_min=1e-7  # Minimum learning rate
                )
        else:
            raise ValueError(f"Unknown scheduler: {name}")
    except Exception as e:
        print(f"Error creating scheduler: {e}")
        return None

class ClusteringError(Exception):
    """Custom exception for clustering failures."""
    def __init__(self, method: str, message: str, original_error: Exception = None):
        self.method = method
        self.original_error = original_error
        super().__init__(f"Clustering failed for method '{method}': {message}\nOriginal error: {str(original_error)}")

def create_synthetic_labels(data, num_clusters=10, method='kmeans', random_state=42, normalize=False):
    """Create synthetic labels using different clustering methods.
    
    Args:
        data (torch.Tensor): Input data to cluster
        num_clusters (int): Number of clusters to create
        method (str): Clustering method ('kmeans' or 'gmm')
        random_state (int): Random seed for reproducibility
        normalize (bool): Whether to normalize the data before clustering
    
    Returns:
        torch.Tensor: Cluster labels for each data point
        
    Raises:
        ClusteringError: If clustering fails with detailed error information
        ValueError: For invalid input parameters
    """
    # Input validation
    if not isinstance(data, torch.Tensor):
        raise ValueError("Data must be a torch.Tensor")
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2")
    if method not in ['kmeans', 'gmm']:
        raise ValueError("method must be either 'kmeans' or 'gmm'")
    
    # Convert to numpy for sklearn
    data_np = data.cpu().numpy()
    
    # Normalize if requested
    if normalize:
        try:
            scaler = StandardScaler()
            data_np = scaler.fit_transform(data_np)
        except Exception as e:
            raise ClusteringError(method, "Failed to normalize data", e)
    
    try:
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(
                n_clusters=num_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
        else:  # method == 'gmm'
            from sklearn.mixture import GaussianMixture
            clusterer = GaussianMixture(
                n_components=num_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300,
                covariance_type='full',
                init_params='kmeans'  # Use kmeans initialization for stability
            )
        
        # Fit and predict
        labels = clusterer.fit_predict(data_np)
        
        # Convert to torch tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Verify the output
        if len(labels) != len(data):
            raise ClusteringError(method, f"Number of labels ({len(labels)}) does not match number of data points ({len(data)})")
        
        unique_clusters = torch.unique(labels)
        if len(unique_clusters) != num_clusters:
            # This is a warning condition but not necessarily a failure
            print(f"Warning: Found {len(unique_clusters)} unique clusters out of {num_clusters} requested")
            print("Unique cluster counts:", torch.bincount(labels).tolist())
            
            # If we have too few clusters, this is a serious problem
            if len(unique_clusters) < num_clusters / 2:
                raise ClusteringError(
                    method,
                    f"Clustering produced too few unique clusters ({len(unique_clusters)} < {num_clusters}/2)"
                )
        
        return labels
        
    except Exception as e:
        # Log the full error with traceback
        print(f"Error during {method} clustering:")
        traceback.print_exc()
        
        # Raise custom error with detailed information
        raise ClusteringError(
            method,
            "Failed to perform clustering - check logs for details",
            e
        ) from e

def save_discriminator_checkpoint(model, optimizer, epoch, val_acc, filename):
    """Save discriminator model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(checkpoint, filename)

def load_discriminator_checkpoint(model, optimizer, filename):
    """Load discriminator model checkpoint."""
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_acc']

def save_precomputed_samples(samples, save_dir):
    """Save precomputed samples to disk."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(samples, os.path.join(save_dir, 'precomputed_samples.pt'))

def load_precomputed_samples(save_dir):
    """Load precomputed samples from disk."""
    samples_path = os.path.join(save_dir, 'precomputed_samples.pt')
    if os.path.exists(samples_path):
        return torch.load(samples_path, weights_only=True)
    return None

def get_cache_path(fusion_dir: str, dataset_path: str, train_size: Optional[int], val_size: Optional[int], test_size: Optional[int]) -> str:
    """
    Automatically construct cache path based on model and dataset parameters.
    
    Args:
        fusion_dir: Path to fusion model checkpoint
        dataset_path: Path to dataset
        train_size: Number of training samples to use (None = use all)
        val_size: Number of validation samples to use (None = use all)
        test_size: Number of test samples to use (None = use all)
        
    Returns:
        Cache directory path
    """
    # Extract model info from fusion path
    model_name = os.path.basename(fusion_dir)
    # Extract weights from model name (e.g., model_epoch_020_v_latents_0.5_t_0.5...)
    weights_match = re.search(r'v_latents_(\d+\.\d+)_t_(\d+\.\d+)', model_name)
    if weights_match:
        v_weight, t_weight = weights_match.groups()
        model_str = f"fusion_v{v_weight}_t{t_weight}"
    else:
        model_str = "fusion"
    
    # Extract epoch
    epoch_match = re.search(r'epoch_(\d+)', model_name)
    if epoch_match:
        model_str += f"_epoch{epoch_match.group(1)}"
    
    # Get dataset name and sizes
    dataset_name = os.path.basename(dataset_path)
    
    # Format sizes, using 'all' for None values
    def format_size(size: Optional[int]) -> str:
        if size is None:
            return "all"
        return f"{size//1000}k" if size >= 1000 else str(size)
    
    sizes_str = f"{dataset_name}_{format_size(train_size)}_{format_size(val_size)}_{format_size(test_size)}"
    
    # Construct cache path
    cache_dir = os.path.join("precomputed_samples", model_str, sizes_str)
    return cache_dir

def precompute_domain_encodings(data_module, domain_modules, device, batch_size=32, cache_dir=None):
    """
    Precompute and cache domain encodings for all splits.
    
    Args:
        data_module: The data module containing the datasets
        domain_modules: Dictionary of domain modules
        device: Device to run computation on
        batch_size: Batch size for processing
        cache_dir: Directory to save cache files (default: './domain_cache')
        
    Returns:
        Dict containing the cached encodings for all splits
    """
    if cache_dir is None:
        cache_dir = './domain_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename based on domain module checksums
    checksums = {}
    for domain_name, module in domain_modules.items():
        # Get state dict as bytes and compute hash
        state_bytes = pickle.dumps(module.state_dict())
        checksums[domain_name] = hashlib.sha256(state_bytes).hexdigest()[:8]
    
    cache_name = f"domain_cache_{'_'.join(f'{k}_{v}' for k,v in sorted(checksums.items()))}.pt"
    cache_path = os.path.join(cache_dir, cache_name)
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"\nFound existing domain encoding cache: {cache_path}")
        return torch.load(cache_path)
    
    print("\nPrecomputing domain encodings (this will be done only once)...")
    
    # Ensure all domain modules are in eval mode
    for module in domain_modules.values():
        module.eval()
    
    # Process each split
    splits = ['train', 'val', 'test']
    cached_data = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get the appropriate dataset
        if split == 'train':
            dataset = data_module.train_dataset
        elif split == 'val':
            dataset = data_module.val_dataset
        else:  # test
            dataset = data_module.test_dataset
        
        if not dataset:
            print(f"No {split} dataset found, skipping...")
            continue
        
        # Create dataloader with no shuffling to maintain order
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Initialize storage for this split
        split_data = {domain: [] for domain in domain_modules.keys()}
        
        # Process batches
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{total_batches}")
            
            # Process batch and move to device
            processed_batch = process_batch(batch, device)
            
            # Extract encodings for each domain
            with torch.no_grad():
                for domain_name, module in domain_modules.items():
                    if domain_name in processed_batch:
                        encoding = module(processed_batch[domain_name])
                        # Move to CPU to save memory
                        split_data[domain_name].append(encoding.cpu())
        
        # Concatenate all batches for each domain
        cached_data[split] = {
            domain: torch.cat(tensors, dim=0) 
            for domain, tensors in split_data.items() 
            if tensors  # Only include domains that had data
        }
        
        print(f"Processed {split} split - shapes:")
        for domain, tensor in cached_data[split].items():
            print(f"{domain}: {tensor.shape}")
    
    # Save cache
    print(f"\nSaving domain encoding cache to {cache_path}")
    torch.save(cached_data, cache_path)
    
    return cached_data

def process_small_batch_cpu(batch, domain_modules):
    """Process small batches on CPU for better efficiency."""
    results = {}
    for domain_name, module in domain_modules.items():
        if domain_name not in batch:
            continue
        
        # Move small batch to CPU
        module = module.cpu()
        data = batch[domain_name].cpu()
        
        with torch.no_grad():
            results[domain_name] = module(data)
            
        # Move module back to GPU if available
        if torch.cuda.is_available():
            module.cuda()
            
    return results

def process_batch_optimized(batch, domain_modules, device, min_batch_size=32):
    """Process batch with optimization for small batches."""
    batch_size = len(next(iter(batch.values())))
    
    # For very small batches, process on CPU
    if batch_size < min_batch_size:
        return process_small_batch_cpu(batch, domain_modules)
    
    # For normal batches, process on GPU with vectorization
    results = {}
    for domain_name, module in domain_modules.items():
        if domain_name not in batch:
            continue
        
        with torch.no_grad():
            data = batch[domain_name].to(device, non_blocking=True)
            results[domain_name] = module(data)
            
    return results

def compute_and_log_cluster_metrics(model, data_loader, clusterer, device, num_clusters, cluster_method, prefix, epoch):
    """Compute and log per-cluster accuracy metrics and visualizations."""
    try:
        # Initialize arrays to store predictions and true labels/probs
        all_preds = []
        all_true = []
        cluster_correct = np.zeros(num_clusters)
        cluster_total = np.zeros(num_clusters)
        
        # Evaluate model
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):  # Standardize autocast usage
            for x1, x2, target in data_loader:
                # Move data to device
                x1 = x1.to(device)
                x2 = x2.to(device)
                target = target.to(device)  # For GMM, this is a float tensor of shape (B, K)
                
                # Get model predictions based on discriminator type
                if prefix == 'discrim_domain1_val' or prefix == 'discrim_domain1_test':
                    out = model(x1)  # Use only x1 for domain1 discriminator
                elif prefix == 'discrim_domain2_val' or prefix == 'discrim_domain2_test':
                    out = model(x2)  # Use only x2 for domain2 discriminator
                else:  # joint discriminator
                    out = model(torch.cat([x1, x2], dim=1))  # Use concatenated input only for joint
                
                # Handle predictions based on clustering method
                if cluster_method == 'kmeans':
                    _, pred = out.max(1)
                    pred = pred.cpu().numpy()
                    true = target.cpu().numpy()
                else:  # GMM
                    log_q = F.log_softmax(out, dim=1)
                    pred = log_q.argmax(dim=1).cpu().numpy()
                    true = target.argmax(dim=1).cpu().numpy()
                
                # Accumulate predictions
                all_preds.extend(pred)
                all_true.extend(true)
                
                # Update per-cluster counts
                for k in range(num_clusters):
                    mask = (true == k)
                    cluster_total[k] += mask.sum()
                    cluster_correct[k] += (pred[mask] == k).sum()
        
        # Compute per-cluster accuracies
        cluster_accs = np.zeros(num_clusters)
        for k in range(num_clusters):
            if cluster_total[k] > 0:
                cluster_accs[k] = cluster_correct[k] / cluster_total[k]
        
        # 1. Log histogram of cluster accuracies - simplified version
        wandb.log({
            f"{prefix}/cluster_acc_dist": wandb.Histogram(cluster_accs),
            "epoch": epoch
        })
        
        # 2. Log percentile statistics
        percentiles = [10, 25, 50, 75, 90]
        qs = np.percentile(cluster_accs, percentiles)
        wandb.log({
            f"{prefix}/acc_p{p}": q for p, q in zip(percentiles, qs)
        })
        
        # 3. Log worst clusters info with proper type casting
        n_worst = 5  # Number of worst clusters to track
        worst_indices = np.argsort(cluster_accs)[:n_worst]
        worst_clusters_data = []
        for idx in worst_indices:
            worst_clusters_data.append([
                int(idx),  # Explicitly cast to Python int
                int(cluster_total[idx]),  # Explicitly cast to Python int
                float(cluster_accs[idx])  # Already casting to Python float
            ])
        
        # Create a wandb Table for worst clusters
        worst_clusters_table = wandb.Table(
            data=worst_clusters_data,
            columns=["Cluster ID", "Size", "Accuracy"]
        )
        wandb.log({f"{prefix}/worst_clusters": worst_clusters_table})
        
        # 4. Visualize cluster centers with both t-SNE and UMAP
        try:
            from sklearn.manifold import TSNE
            import umap
            import matplotlib.pyplot as plt
            
            # Get cluster centers based on method
            if cluster_method == 'kmeans':
                centers = clusterer.cluster_centers_
            else:  # GMM
                centers = clusterer.means_
            
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # t-SNE visualization
            try:
                tsne = TSNE(n_components=2, random_state=42)
                centers_2d_tsne = tsne.fit_transform(centers)
                
                scatter1 = ax1.scatter(
                    centers_2d_tsne[:, 0],
                    centers_2d_tsne[:, 1],
                    c=cluster_accs,
                    cmap='viridis',
                    s=100
                )
                ax1.set_title(f'{prefix} Cluster Centers (t-SNE)')
                plt.colorbar(scatter1, ax=ax1, label='Cluster Accuracy')
            except Exception as e:
                print(f"Error in t-SNE visualization: {e}")
                ax1.text(0.5, 0.5, 't-SNE failed', ha='center', va='center')
            
            # UMAP visualization
            try:
                reducer = umap.UMAP(random_state=42)
                centers_2d_umap = reducer.fit_transform(centers)
                
                scatter2 = ax2.scatter(
                    centers_2d_umap[:, 0],
                    centers_2d_umap[:, 1],
                    c=cluster_accs,
                    cmap='viridis',
                    s=100
                )
                ax2.set_title(f'{prefix} Cluster Centers (UMAP)')
                plt.colorbar(scatter2, ax=ax2, label='Cluster Accuracy')
            except Exception as e:
                print(f"Error in UMAP visualization: {e}")
                ax2.text(0.5, 0.5, 'UMAP failed', ha='center', va='center')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Log to wandb with explicit figure handling
            wandb.log({
                f"{prefix}/cluster_centers_projections": wandb.Image(fig),
                "epoch": epoch
            })
            plt.close(fig)
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in cluster metrics computation: {e}")
        import traceback
        traceback.print_exc()

def cleanup_dataloader(loader):
    """Cleanup DataLoader resources properly."""
    if loader is not None:
        # Close the dataloader and its workers
        try:
            loader._iterator = None  # Reset iterator
            if hasattr(loader, 'pin_memory') and loader.pin_memory:
                # Clear pin memory thread
                if hasattr(loader, '_pin_memory_thread'):
                    loader._pin_memory_thread_done_event.set()
                    loader._pin_memory_thread.join()
            # Close workers if they exist
            if loader.num_workers > 0:
                loader._worker_pids_set = False
                if hasattr(loader, '_workers'):
                    for worker in loader._workers:
                        if worker is not None:
                            worker.join()
        except Exception as e:
            print(f"Warning: Error during dataloader cleanup: {e}")
        finally:
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def optimized_collate_fn(batch: List[Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Optimized collate function with vectorized operations and smart inspection.
    
    Args:
        batch: List of samples to collate
        
    Returns:
        Collated batch as tensor or dict of tensors
    """
    # Fast path: check if batch is simple tensor list
    if all(isinstance(x, torch.Tensor) for x in batch[:min(len(batch), 3)]):  # Check first 3 samples
        try:
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        except:
            pass  # Fall through to full logic if pad_sequence fails
    
    # Fast path: check if batch is list of tuples with same length
    if all(isinstance(x, tuple) and len(x) == len(batch[0]) for x in batch[:min(len(batch), 3)]):
        try:
            return default_collate(batch)
        except:
            pass
    
    # Handle dictionary batches with vectorized operations
    if isinstance(batch[0], dict):
        # Get all keys from first few samples
        all_keys = set().union(*[d.keys() for d in batch[:min(len(batch), 3)]])
        
        result = {}
        for key in all_keys:
            # Collect tensors for this key
            tensors = []
            for sample in batch:
                if key in sample:
                    value = sample[key]
                    # Handle special cases
                    if hasattr(value, 'bert'):
                        tensors.append(value.bert)
                    elif isinstance(value, torch.Tensor):
                        tensors.append(value)
                    elif isinstance(value, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in value):
                        # Handle list/tuple of tensors
                        tensors.append(torch.stack(value))
                    else:
                        try:
                            tensors.append(torch.as_tensor(value))
                        except:
                            tensors.append(value)
            
            if tensors:
                try:
                    # Vectorized stack operation
                    if all(isinstance(t, torch.Tensor) for t in tensors):
                        if all(t.shape == tensors[0].shape for t in tensors):
                            result[key] = torch.stack(tensors)
                        else:
                            # Handle variable-sized tensors using pad_sequence
                            try:
                                result[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
                            except:
                                # If pad_sequence fails, fall back to manual padding
                                max_size = max(t.size(0) for t in tensors)
                                padded = []
                                for t in tensors:
                                    if t.size(0) < max_size:
                                        pad_size = max_size - t.size(0)
                                        # Build padding tuples in reverse order
                                        padding = [(0, 0)] * (t.dim() - 1)  # Later dimensions
                                        padding = [(0, pad_size)] + padding  # First dimension
                                        # Reverse padding list before flattening
                                        padding = padding[::-1]
                                        # Flatten padding list
                                        pad_args = [p for pair in padding for p in pair]
                                        t = F.pad(t, pad_args)
                                    padded.append(t)
                                result[key] = torch.stack(padded)
                    else:
                        # Fall back to default_collate for this key
                        result[key] = default_collate(tensors)
                except Exception as e:
                    # If stacking fails, keep as list
                    result[key] = tensors
        
        return result
    
    # Handle tuple/list batches with vectorized operations
    if isinstance(batch[0], (tuple, list)):
        try:
            # Try to stack each element position
            elements = list(zip(*batch))  # Transpose batch
            result = []
            
            for elem_list in elements:
                if all(isinstance(x, torch.Tensor) for x in elem_list):
                    # All tensors - try vectorized stack
                    if all(x.shape == elem_list[0].shape for x in elem_list):
                        result.append(torch.stack(elem_list))
                    else:
                        # Handle variable-sized tensors using pad_sequence
                        try:
                            result.append(torch.nn.utils.rnn.pad_sequence(elem_list, batch_first=True))
                        except:
                            # Fall back to manual padding if pad_sequence fails
                            max_size = max(x.size(0) for x in elem_list)
                            padded = []
                            for x in elem_list:
                                if x.size(0) < max_size:
                                    pad_size = max_size - x.size(0)
                                    # Build padding tuples in reverse order
                                    padding = [(0, 0)] * (x.dim() - 1)  # Later dimensions
                                    padding = [(0, pad_size)] + padding  # First dimension
                                    # Reverse padding list before flattening
                                    padding = padding[::-1]
                                    # Flatten padding list
                                    pad_args = [p for pair in padding for p in pair]
                                    x = F.pad(x, pad_args)
                                padded.append(x)
                            result.append(torch.stack(padded))
                else:
                    # Try default_collate for this position
                    try:
                        result.append(default_collate(elem_list))
                    except:
                        result.append(elem_list)
            
            return tuple(result) if isinstance(batch[0], tuple) else result
        except Exception as e:
            pass  # Fall through to default_collate if vectorized operations fail
    
    # Final fallback: try default_collate
    try:
        return default_collate(batch)
    except Exception as e:
        print(f"Warning: Collate failed with {str(e)}, returning raw batch")
        return batch

@dataclass
class CheckpointInfo:
    """Container for checkpoint information."""
    model: nn.Module
    optimizer: torch.optim.Optimizer
    epoch: int
    val_acc: float
    prefix: str
    checkpoint_base: str

class AsyncLogger:
    """Asynchronous logger using a background thread."""
    def __init__(self, maxsize: int = 100):
        self.metric_queue = Queue(maxsize=maxsize)
        self.checkpoint_queue = Queue(maxsize=maxsize)
        self.stop_event = Event()
        self.worker_thread = Thread(target=self._log_worker, daemon=True)
        self.worker_thread.start()
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Non-blocking log of metrics."""
        try:
            self.metric_queue.put_nowait(metrics)
        except queue.Full:
            print("Warning: Metric queue full, skipping log")
    
    def enqueue_checkpoint(self, checkpoint_info: CheckpointInfo) -> None:
        """Non-blocking enqueue of checkpoint save request."""
        try:
            self.checkpoint_queue.put_nowait(checkpoint_info)
        except queue.Full:
            print("Warning: Checkpoint queue full, skipping save")
    
    def _log_worker(self) -> None:
        """Background thread for logging and saving checkpoints."""
        while not self.stop_event.is_set():
            # Process metrics
            try:
                # Try to get metrics with timeout to allow checking stop_event
                metrics = self.metric_queue.get(timeout=0.1)
                wandb.log(metrics)
                self.metric_queue.task_done()
            except queue.Empty:
                pass
            
            # Process checkpoints
            try:
                checkpoint_info = self.checkpoint_queue.get_nowait()
                # Save checkpoint
                checkpoint_path = os.path.join(
                    checkpoint_info.checkpoint_base,
                    f"{checkpoint_info.prefix}_best.pt"
                )
                save_discriminator_checkpoint(
                    checkpoint_info.model,
                    checkpoint_info.optimizer,
                    checkpoint_info.epoch,
                    checkpoint_info.val_acc,
                    checkpoint_path
                )
                print(f"Saved best checkpoint for {checkpoint_info.prefix} with val_acc: {checkpoint_info.val_acc:.2f}%")
                self.checkpoint_queue.task_done()
            except queue.Empty:
                pass
    
    def shutdown(self) -> None:
        """Shutdown the logger thread cleanly."""
        print("Shutting down async logger...")
        self.stop_event.set()
        
        # Process remaining items
        while not self.metric_queue.empty():
            try:
                metrics = self.metric_queue.get_nowait()
                wandb.log(metrics)
                self.metric_queue.task_done()
            except queue.Empty:
                break
        
        while not self.checkpoint_queue.empty():
            try:
                checkpoint_info = self.checkpoint_queue.get_nowait()
                checkpoint_path = os.path.join(
                    checkpoint_info.checkpoint_base,
                    f"{checkpoint_info.prefix}_best.pt"
                )
                save_discriminator_checkpoint(
                    checkpoint_info.model,
                    checkpoint_info.optimizer,
                    checkpoint_info.epoch,
                    checkpoint_info.val_acc,
                    checkpoint_path
                )
                self.checkpoint_queue.task_done()
            except queue.Empty:
                break
        
        # Wait for queues to be fully processed
        self.metric_queue.join()
        self.checkpoint_queue.join()
        self.worker_thread.join(timeout=5)
        print("Async logger shutdown complete")

def save_cluster_data(cluster_data, file_path):
    """Save cluster data in a format compatible with PyTorch's safe loading."""
    # Everything should already be tensors, just save directly
    torch.save(cluster_data, file_path)

def load_cluster_data(file_path):
    """Load cluster data safely."""
    return torch.load(file_path, weights_only=True)

def train():
    """Training function for each sweep run."""
    global CACHED_FEATURES
    if global_args is None:
        raise RuntimeError("global_args not set. Make sure to call set_global_args() before running train()")
    
    async_logger = None
    train_loader = None
    val_loader = None
    test_loader = None
    
    try:
        # Initialize wandb with reinit=True to ensure each trial gets its own run
        run = wandb.init(
            project=global_args.wandb_project,
            reinit=True,
            name=f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Initialize async logger
        async_logger = AsyncLogger(maxsize=100)
        
        # Get combined config
        config = wandb.config
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint directory with timestamp
        checkpoint_base = f'checkpoints/discriminators/run_{timestamp}'
        os.makedirs(checkpoint_base, exist_ok=True)
        
        # Update config with CLI args that aren't part of the sweep
        config.update({
            "domain_v_ckpt": global_args.domain_v_ckpt,
            "domain_t_ckpt": global_args.domain_t_ckpt,
            "dataset_path": global_args.dataset_path,
            "train_size": global_args.train_size,
            "val_size": global_args.val_size,
            "test_size": global_args.test_size,
            "fusion_dir": global_args.fusion_dir,
            "samples_cache_dir": global_args.samples_cache_dir,
            "cluster_method": global_args.cluster_method,
            "load_upfront": global_args.load_upfront  # Add load_upfront to config
        }, allow_val_change=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize gradient scaler for AMP
        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # Create domain configs and load modules
        domain_configs = [
            {
                "name": "v_latents",
                "checkpoint_path": config.domain_v_ckpt,
                "variant": "pretrained",
                "latent_dim": 12,
                "output_dim": 12,
                "domain_type": "v_latents"
            },
            {
                "name": "t",
                "checkpoint_path": config.domain_t_ckpt,
                "variant": "pretrained",
                "latent_dim": 768,
                "output_dim": 768,
                "domain_type": "t"
            }
        ]
        
        # Load domain modules
        domain_modules = load_domain_modules(domain_configs)
        print(f"Loaded domain modules: {list(domain_modules.keys())}")
        
        # Move domain modules to device and set to eval mode
        for domain_name, domain_module in domain_modules.items():
            domain_modules[domain_name] = domain_module.to(device)
            domain_modules[domain_name].eval()
            if hasattr(domain_module, 'projector'):
                domain_module.projector = domain_module.projector.to(device)
        
        # Load fusion model
        print(f"Loading fusion model from {config.fusion_dir}")
        fusion_model = load_checkpoint(config.fusion_dir, domain_modules, device=device)
        fusion_model = fusion_model.to(device)
        fusion_model.eval()
        
        # Set up data module
        data_module = setup_data_module(
            config.dataset_path,
            config.batch_size,
            domain_modules,
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size
        )
        
        # Get cache directory path
        cache_dir = get_cache_path(
            fusion_dir=config.fusion_dir,
            dataset_path=config.dataset_path,
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size
        )
        
        # Use global cache if available, otherwise compute and store
        global CACHED_FEATURES
        if CACHED_FEATURES is None:
            CACHED_FEATURES = precompute_and_cache_features(
                data_module=data_module,
                domain_modules=domain_modules,
                device=device,
                batch_size=config.batch_size,
                cache_dir=cache_dir,
                fusion_model=fusion_model
            )
        
        # If loading upfront, move all features to GPU once
        if global_args.load_upfront and torch.cuda.is_available():
            print("\nMoving all features to GPU upfront...")
            CACHED_FEATURES = move_features_to_device(CACHED_FEATURES, device)
        
        # Get datasets with cached features using global args for clustering
        try:
            clusterer, train_dataset, val_dataset, test_dataset = get_datasets(
                precomputed_data=CACHED_FEATURES,
                num_clusters=global_args.num_clusters,
                cluster_method=global_args.cluster_method,
                cluster_seed=global_args.cluster_seed,
                normalize_inputs=global_args.normalize_inputs
            )
        except Exception as e:
            print(f"Error getting datasets: {e}")
            traceback.print_exc()
            raise
        
        # Create data loaders based on loading strategy
        train_loader = None
        val_loader = None
        test_loader = None
        
        try:
            # Create train loader
            _, train_loader = get_dataset_and_loader(
                v_features=train_dataset.v_features,
                t_features=train_dataset.t_features,
                labels=train_dataset.labels,
                batch_size=config.batch_size,
                shuffle=True,
                device=device,
                load_upfront=global_args.load_upfront,
                num_workers=8,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=4,
                persistent_workers=True if torch.cuda.is_available() and torch.get_num_threads() > 1 else False,
                collate_fn=optimized_collate_fn
            )
            
            # Create val loader
            _, val_loader = get_dataset_and_loader(
                v_features=val_dataset.v_features,
                t_features=val_dataset.t_features,
                labels=val_dataset.labels,
                batch_size=config.batch_size,
                shuffle=False,
                device=device,
                load_upfront=global_args.load_upfront,
                num_workers=8,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=4,
                persistent_workers=True if torch.cuda.is_available() and torch.get_num_threads() > 1 else False,
                collate_fn=optimized_collate_fn
            )
            
            # Create test loader
            _, test_loader = get_dataset_and_loader(
                v_features=test_dataset.v_features,
                t_features=test_dataset.t_features,
                labels=test_dataset.labels,
                batch_size=config.batch_size,
                shuffle=False,
                device=device,
                load_upfront=global_args.load_upfront,
                num_workers=8,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=4,
                persistent_workers=True if torch.cuda.is_available() and torch.get_num_threads() > 1 else False,
                collate_fn=optimized_collate_fn
            )
            
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            traceback.print_exc()
            raise
        
        # Get input dimensions from first batch
        sample_batch = next(iter(train_loader))
        x1_dim = sample_batch[0].size(1)
        x2_dim = sample_batch[1].size(1)
        
        print(f"\nData dimensions:")
        print(f"x1_dim: {x1_dim}")
        print(f"x2_dim: {x2_dim}")
        
        # Create discriminators
        d1 = create_discriminator(
            input_dim=x1_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.layers,
            activation=config.activation,
            num_labels=global_args.num_clusters,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        d2 = create_discriminator(
            input_dim=x2_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.layers,
            activation=config.activation,
            num_labels=global_args.num_clusters,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        d12 = create_discriminator(
            input_dim=x1_dim + x2_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.layers,
            activation=config.activation,
            num_labels=global_args.num_clusters,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        # Create optimizers
        opt1 = get_optimizer(config.optimizer, d1.parameters(), config.lr, config.weight_decay, config.momentum)
        opt2 = get_optimizer(config.optimizer, d2.parameters(), config.lr, config.weight_decay, config.momentum)
        opt12 = get_optimizer(config.optimizer, d12.parameters(), config.lr, config.weight_decay, config.momentum)
        
        # Create schedulers
        sched1 = get_scheduler(config.lr_schedule, opt1, config.discrim_epochs, config.lr_step_size, config.lr_gamma, config.lr_warmup_epochs)
        sched2 = get_scheduler(config.lr_schedule, opt2, config.discrim_epochs, config.lr_step_size, config.lr_gamma, config.lr_warmup_epochs)
        sched12 = get_scheduler(config.lr_schedule, opt12, config.discrim_epochs, config.lr_step_size, config.lr_gamma, config.lr_warmup_epochs)
        
        # Set up loss function based on clustering method
        criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for all cases
        
        # Training state
        best_val_acc = {
            'discrim_domain1': 0.0,
            'discrim_domain2': 0.0,
            'discrim_joint': 0.0
        }
        patience_counter = {
            'discrim_domain1': 0,
            'discrim_domain2': 0,
            'discrim_joint': 0
        }
        early_stop = {
            'discrim_domain1': False,
            'discrim_domain2': False,
            'discrim_joint': False
        }
        
        # Training loop
        for epoch in range(config.discrim_epochs):
            # Update dataset epoch counter to trigger shuffling if using GPU dataset
            if global_args.load_upfront and isinstance(train_dataset, GPUBatchedDataset):
                train_dataset.set_epoch(epoch)
            
            # Training and validation for each discriminator
            for (model, opt, sched, prefix) in [
                (d1, opt1, sched1, 'discrim_domain1'),
                (d2, opt2, sched2, 'discrim_domain2'),
                (d12, opt12, sched12, 'discrim_joint')
            ]:
                if early_stop[prefix]:
                    continue
                
                # Training phase
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (x1, x2, target) in enumerate(train_loader):
                    # Move to device only if not already there (for upfront loading, tensors are already on device)
                    if not global_args.load_upfront:
                        x1 = x1.to(device, non_blocking=True)
                        x2 = x2.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)
                    
                    opt.zero_grad()
                    
                    # Forward pass with autocast
                    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                        if prefix == 'discrim_domain1':
                            out = model(x1)
                        elif prefix == 'discrim_domain2':
                            out = model(x2)
                        else:  # joint discriminator
                            out = model(torch.cat([x1, x2], dim=1))
                        
                        # Calculate loss and accuracy
                        loss = criterion(out, target)
                        _, predicted = out.max(1)
                        train_total += target.size(0)
                        train_correct += predicted.eq(target).sum().item()
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    
                    train_loss += loss.item()
                
                # Training phase complete
                
                # Only clean memory at major phase transitions
                if epoch % global_args.save_every == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    for x1, x2, target in val_loader:
                        # Move to device only if not already there
                        if not global_args.load_upfront:
                            x1 = x1.to(device, non_blocking=True)
                            x2 = x2.to(device, non_blocking=True)
                            target = target.to(device, non_blocking=True)
                        
                        if prefix == 'discrim_domain1':
                            out = model(x1)
                        elif prefix == 'discrim_domain2':
                            out = model(x2)
                        else:  # joint discriminator
                            out = model(torch.cat([x1, x2], dim=1))
                        
                        # Calculate loss and accuracy
                        loss = criterion(out, target)
                        _, predicted = out.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                        
                        val_loss += loss.item()
                
                # Early stopping logic
                val_acc = 100. * val_correct / val_total if val_total > 0 else 0
                
                # Log both namespaced and top-level metrics
                metrics = {
                    f"{prefix}/train_loss": train_loss / len(train_loader),
                    f"{prefix}/train_acc": 100. * train_correct / train_total,
                    f"{prefix}/val_loss": val_loss / len(val_loader),
                    f"{prefix}/val_acc": val_acc,
                    f"{prefix}/learning_rate": opt.param_groups[0]['lr'],
                }
                
                # Add top-level metrics for joint discriminator
                if prefix == 'discrim_joint':
                    metrics.update({
                        "train_loss": train_loss / len(train_loader),
                        "train_acc": 100. * train_correct / train_total,
                        "val_loss": val_loss / len(val_loader),
                        "val_acc": val_acc,
                        "learning_rate": opt.param_groups[0]['lr'],
                    })
                
                metrics["epoch"] = epoch + 1
                
                # Non-blocking log of metrics
                async_logger.log(metrics)
                
                # Save checkpoint only on improvement
                if val_acc > best_val_acc[prefix]:
                    best_val_acc[prefix] = val_acc
                    patience_counter[prefix] = 0
                    
                    # Non-blocking checkpoint save
                    checkpoint_info = CheckpointInfo(
                        model=model,
                        optimizer=opt,
                        epoch=epoch,
                        val_acc=val_acc,
                        prefix=prefix,
                        checkpoint_base=checkpoint_base
                    )
                    async_logger.enqueue_checkpoint(checkpoint_info)
                else:
                    patience_counter[prefix] += 1
                    if patience_counter[prefix] >= config.patience:
                        early_stop[prefix] = True
                        print(f"{prefix} early stopped at epoch {epoch + 1}")
            
            # Step all schedulers once per epoch after training all discriminators
            for sched in [sched1, sched2, sched12]:
                if sched is not None:
                    sched.step()
            
            # Check if all discriminators have early stopped
            if all(early_stop.values()):
                print(f"All discriminators early stopped at epoch {epoch + 1}")
                break
        
        # Final evaluation on test set
        print("\nEvaluating best models on test set...")
        test_metrics = {}
        
        for prefix, model in [
            ('discrim_domain1', d1),
            ('discrim_domain2', d2),
            ('discrim_joint', d12)
        ]:
            # Try to load best checkpoint, fall back to final if best doesn't exist
            checkpoint_path = f'{checkpoint_base}/{prefix}_best.pt'
            final_checkpoint_path = f'{checkpoint_base}/{prefix}_final.pt'
            
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model_state_dict'])
                print(f"Loaded best checkpoint for {prefix}")
            elif os.path.exists(final_checkpoint_path):
                model.load_state_dict(torch.load(final_checkpoint_path, weights_only=True)['model_state_dict'])
                print(f"Loaded final checkpoint for {prefix}")
            else:
                print(f"Warning: No checkpoint found for {prefix}, using current model state")
            
            # Evaluate on test set
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):  # Consistent usage
                for x1, x2, target in test_loader:
                    # Move to device only if not already there
                    if not global_args.load_upfront:
                        x1 = x1.to(device)
                        x2 = x2.to(device)
                        target = target.to(device)
                    
                    if prefix == 'discrim_domain1':
                        out = model(x1)
                    elif prefix == 'discrim_domain2':
                        out = model(x2)
                    else:  # joint discriminator
                        out = model(torch.cat([x1, x2], dim=1))
                    
                    if global_args.cluster_method == 'kmeans':
                        _, predicted = out.max(1)
                        test_total += target.size(0)
                        test_correct += predicted.eq(target).sum().item()
                    else:  # GMM
                        log_q = F.log_softmax(out, dim=1)
                        pred_labels = log_q.argmax(dim=1)
                        true_labels = target.argmax(dim=1)
                        test_total += target.size(0)
                        test_correct += (pred_labels == true_labels).sum().item()
            
            # Calculate test accuracy
            test_acc = 100. * test_correct / test_total if test_total > 0 else 0
            test_metrics[prefix] = test_acc
            
            # Log both namespaced and top-level metrics
            metrics = {
                f'{prefix}/test_acc': test_acc,
                f'{prefix}/final_val_acc': best_val_acc[prefix]
            }
            
            # Add top-level metrics for joint discriminator
            if prefix == 'discrim_joint':
                metrics.update({
                    'test_acc': test_acc,
                    'final_val_acc': best_val_acc[prefix]
                })
            
            wandb.log(metrics)
            
            print(f"{prefix} Test Accuracy: {test_acc:.2f}%")
            
            # Compute and log cluster metrics on test set
            compute_and_log_cluster_metrics(
                model=model,
                data_loader=test_loader,
                clusterer=clusterer,
                device=device,
                num_clusters=global_args.num_clusters,
                cluster_method=global_args.cluster_method,
                prefix=f"{prefix}_test",
                epoch=epoch
            )
        
        # Update run summary with best results
        final_metrics = {
            'best_val_acc_domain1': best_val_acc['discrim_domain1'],
            'best_val_acc_domain2': best_val_acc['discrim_domain2'],
            'best_val_acc_joint': best_val_acc['discrim_joint'],
            'test_acc_domain1': test_metrics.get('discrim_domain1', 0.0),
            'test_acc_domain2': test_metrics.get('discrim_domain2', 0.0),
            'test_acc_joint': test_metrics.get('discrim_joint', 0.0),
            'best_test_acc': test_metrics.get('discrim_joint', 0.0)
        }
        async_logger.log(final_metrics)
        
        # Shutdown async logger cleanly
        async_logger.shutdown()
        
        # Cleanup dataloaders
        cleanup_dataloader(train_loader)
        cleanup_dataloader(val_loader)
        cleanup_dataloader(test_loader)
        
        # Final cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup phase
        print("\nCleaning up after sweep trial...")
        
        # Ensure async logger is shutdown only if not already stopped
        if async_logger is not None and not async_logger.stop_event.is_set():
            async_logger.shutdown()
        
        # Clean up dataloaders
        if train_loader is not None:
            cleanup_dataloader(train_loader)
        if val_loader is not None:
            cleanup_dataloader(val_loader)
        if test_loader is not None:
            cleanup_dataloader(test_loader)
        
        # Finish wandb run
        wandb.finish()
        
        # Final GPU memory cleanup after trial
        if torch.cuda.is_available():
            # Return all GPU memory to driver to start next trial with clean slate
            torch.cuda.empty_cache()
            
            # Print memory stats if available
            if hasattr(torch.cuda, 'memory_summary'):
                print("\nGPU Memory Summary after cleanup:")
                print(torch.cuda.memory_summary())
        
        print("Sweep trial cleanup complete")

def get_dataset_and_loader(
    v_features: torch.Tensor,
    t_features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    load_upfront: bool,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    collate_fn = None
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    """Create appropriate dataset and loader based on loading strategy."""
    # Debug info about input tensors
    print(f"[DEBUG] Input tensor info:")
    print(f"[DEBUG] v_features: {type(v_features)}, shape: {v_features.shape if hasattr(v_features, 'shape') else 'no shape'}")
    print(f"[DEBUG] t_features: {type(t_features)}, shape: {t_features.shape if hasattr(t_features, 'shape') else 'no shape'}")
    print(f"[DEBUG] labels: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else 'no shape'}")
    
    # Ensure tensors have correct shape
    if v_features.dim() == 3:
        v_features = v_features.squeeze(1)
    if t_features.dim() == 3:
        t_features = t_features.squeeze(1)
    
    # Ensure labels are in the correct format
    if labels.dim() == 2:  # If we have probability distributions
        print("[DEBUG] Converting probability distributions to class indices")
        labels = labels.argmax(dim=1)
    elif labels.dim() == 3:
        labels = labels.squeeze(1)
    elif labels.dim() > 3:
        raise ValueError(f"Unexpected label dimension: {labels.dim()}")
    print(f"[DEBUG] Labels are valid in get_dataset_and_loader: shape {labels.shape}, type {type(labels)}")
    
    if load_upfront:
        print("[DEBUG] Creating GPU dataset...")
        # Move tensors to device before creating dataset
        v_features = v_features.to(device)
        t_features = t_features.to(device)
        labels = labels.to(device)
        dataset = GPUBatchedDataset(v_features, t_features, labels)
        print("[DEBUG] Created GPU dataset, checking labels...")
        print(f"[DEBUG] Dataset labels: {type(dataset.labels)}, shape: {dataset.labels.shape}")
    else:
        dataset = SoftLatentDataset(v_features, t_features, labels)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if load_upfront else num_workers,
        pin_memory=pin_memory if not load_upfront else False,
        prefetch_factor=None if load_upfront else prefetch_factor,
        persistent_workers=persistent_workers if not load_upfront else False,
        collate_fn=collate_fn
    )
    
    # Verify loader output shapes
    sample_batch = next(iter(loader))
    print(f"[DEBUG] Successfully created loader, batch shapes: {[x.shape for x in sample_batch]}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if load_upfront else num_workers,
        pin_memory=pin_memory if not load_upfront else False,
        prefetch_factor=None if load_upfront else prefetch_factor,
        persistent_workers=persistent_workers if not load_upfront else False,
        collate_fn=collate_fn
    )
    
    return dataset, loader

def main():
    """Main function to run the sweep."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for PID classifier training")
    
    # Add load_upfront argument with proper boolean flag pattern
    parser.add_argument(
        "--load-upfront",
        action="store_true",
        help="Load all data to GPU upfront and use GPU-resident dataset"
    )
    parser.add_argument(
        "--no-load-upfront",
        action="store_false",
        dest="load_upfront",
        help="Do not load data to GPU upfront (use standard DataLoader)"
    )
    parser.set_defaults(load_upfront=True)  # Changed default to True
    
    # Add save_every argument
    parser.add_argument("--save-every", type=int, default=5,
                      help="Number of epochs between memory cleanup and model saves")
    
    # Directories
    parser.add_argument("--fusion-dir", default="checkpoints/fusion", help="Directory with fusion checkpoints")
    parser.add_argument("--output-dir", default="pid_results", help="Directory to save results")
    
    # Model configuration
    parser.add_argument("--domain-v-ckpt", type=str, required=True,
                      help="Path to visual domain checkpoint")
    parser.add_argument("--domain-t-ckpt", type=str, required=True,
                      help="Path to text domain checkpoint")
    parser.add_argument("--dataset-path", type=str, required=True,
                      help="Path to dataset")
    
    # Dataset size parameters
    parser.add_argument("--train-size", type=int, default=None,
                      help="Number of training samples to use (default: use all)")
    parser.add_argument("--val-size", type=int, default=None,
                      help="Number of validation samples to use (default: use all)")
    parser.add_argument("--test-size", type=int, default=None,
                      help="Number of test samples to use (default: use all)")
    
    # Sweep parameters
    parser.add_argument("--num-trials", type=int, required=True,
                      help="Number of hyperparameter trials to run in the sweep")
    parser.add_argument("--sweep-config", type=str, default="sweep_config.yaml",
                      help="Path to sweep configuration YAML file")
    parser.add_argument("--wandb-project", type=str, required=True,
                      help="W&B project name for experiment tracking")
    parser.add_argument("--samples-cache-dir", type=str, default="./precomputed_samples",
                      help="Directory to cache generated samples")

    # Clustering parameters
    parser.add_argument("--num-clusters", type=int, default=1000,
                      help="Number of clusters for synthetic labels")
    parser.add_argument("--cluster-method", type=str, default='kmeans',
                      choices=['kmeans', 'gmm'],
                      help="Clustering method to use")
    parser.add_argument("--cluster-seed", type=int, default=42,
                      help="Random seed for clustering")
    
    # Fix normalize-inputs flag with proper boolean pattern
    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        help="Normalize input features"
    )
    parser.add_argument(
        "--no-normalize-inputs",
        action="store_false",
        dest="normalize_inputs",
        help="Do not normalize input features"
    )
    parser.set_defaults(normalize_inputs=True)  # Keep the same default as before

    args = parser.parse_args()
    
    # Set global args for wandb workers using the helper function
    set_global_args(args)
    
    # Load sweep configuration
    sweep_config = load_sweep_config(args.sweep_config)
    
    # Initialize sweep from config file
    sweep_id = setup_wandb_sweep(args.sweep_config, args.wandb_project)
    
    # Run agent with specified number of trials
    wandb.agent(sweep_id, function=train, count=args.num_trials)

if __name__ == '__main__':
    main() 