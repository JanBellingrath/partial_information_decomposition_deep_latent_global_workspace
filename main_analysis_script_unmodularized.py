import gc
import itertools
import json
import math
import os
import random
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Mapping, Any, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import seaborn as sns

# Fix import: Replace incorrect torch.amp with proper import from torch.amp
# Import from torch.amp directly for newer PyTorch versions
from torch.amp import autocast, GradScaler

# Create dummy class for backward compatibility
class DummyAMPModule:
    def __init__(self):
        self.autocast = autocast
        self.GradScaler = GradScaler

# Set amp to use the imported torch.amp for all code
amp = DummyAMPModule()

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Try to import wandb, but make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable experiment tracking.")

# Try to import shimmer, but continue if not available (for help/argparse)
try:
    from shimmer.modules.domain import DomainModule
    from shimmer.modules.gw_module import GWModule
    # Import the GWModuleConfigurableFusion from the local file
    from gw_module_configurable_fusion import GWModuleConfigurableFusion
    # Import the necessary functions
    from minimal_script_with_validation import load_domain_modules
    SHIMMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: module not found ({e}). Running in limited mode (can show help but not execute PID analysis).")
    # Create dummy classes for type hints
    class DomainModule: pass
    class GWModuleConfigurableFusion: pass
    class GWModule: pass
    def load_domain_modules(configs): return {}
    SHIMMER_AVAILABLE = False
    
    # Try to import GWModuleConfigurableFusion from losses_and_weights_GLW_training
    try:
        from losses_and_weights_GLW_training import GWModuleConfigurableFusion
        print("Using GWModuleConfigurableFusion from losses_and_weights_GLW_training")
    except ImportError as e:
        print(f"Warning: Could not import GWModuleConfigurableFusion: {e}")
        # Create a dummy implementation that accepts arguments
        class GWModuleConfigurableFusion(nn.Module):
            def __init__(self, domain_modules=None, workspace_dim=None, gw_encoders=None, 
                         gw_decoders=None, fusion_weights=None, fusion_activation_fn=None):
                super().__init__()
                self.domain_modules = domain_modules
                self.workspace_dim = workspace_dim
                self.gw_encoders = gw_encoders
                self.gw_decoders = gw_decoders
                self.fusion_weights = fusion_weights
                self.fusion_activation_fn = fusion_activation_fn
                self.domain_mods = domain_modules  # Alias for compatibility
                print("Warning: Using dummy GWModuleConfigurableFusion implementation")
                
            def fuse(self, x, selection_scores=None):
                # Dummy implementation
                return torch.zeros(1, self.workspace_dim, device=next(self.parameters()).device)
    
    SHIMMER_AVAILABLE = False
    
    # Function to load domain modules
    def load_domain_modules(configs): 
        return {}
    
    # Try to import the load_checkpoint from losses_and_weights_GLW_training as fallback
    try:
        from losses_and_weights_GLW_training import load_checkpoint as _load_checkpoint
        print("Using load_checkpoint from losses_and_weights_GLW_training")
        def load_checkpoint(checkpoint_path, domain_modules, device):
            return _load_checkpoint(checkpoint_path, domain_modules, device)
    except ImportError as e:
        print(f"Warning: Could not import load_checkpoint from losses_and_weights_GLW_training: {e}")
        def load_checkpoint(checkpoint_path, domain_modules, device):
            raise NotImplementedError("load_checkpoint is not available in limited mode")
    finally:
        # This is needed to satisfy the linter
        pass

# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global performance configuration (will be overridden by command line arguments)
CHUNK_SIZE = 128  # Size of chunks for processing large matrices sequentially
MEMORY_CLEANUP_INTERVAL = 10  # Number of iterations after which to force memory cleanup
USE_AMP = False  # Whether to use automatic mixed precision
PRECISION = torch.float16  # Precision to use with AMP
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean up memory between operations

# ——————————————————————————————————————————————————————————————————————————————
# DATASET
#——————————————————————————————————————————————————————————————————————————————
class MultimodalDataset(Dataset):
    """
    Dataset for multimodal data.
    
    This dataset handles multiple tensors (one per modality) and labels.
    """
    
    def __init__(self, data: List[torch.Tensor], labels: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            data: List of tensors, one per modality
            labels: Tensor of labels
        """
        self.data = data
        self.labels = labels
        
        # Validate data
        n_samples = self.data[0].size(0)
        for i, tensor in enumerate(self.data):
            if tensor.size(0) != n_samples:
                raise ValueError(
                    f"All data tensors must have the same first dimension. "
                    f"Got {tensor.size(0)} for tensor {i}, expected {n_samples}"
                )
        
        if self.labels.size(0) != n_samples:
            raise ValueError(
                f"Labels must have the same first dimension as data. "
                f"Got {self.labels.size(0)}, expected {n_samples}"
            )
        
        # Log dimensions
        print(f"MultimodalDataset init - labels shape: {self.labels.shape}, dim: {self.labels.dim()}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.data[0].size(0)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get sample by index.
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (modality1, modality2, ..., label)
        """
        # Get data for each modality
        modalities = [tensor[idx] for tensor in self.data]
        
        # Get label
        label = self.labels[idx]
        
        # Return as tuple
        return tuple(modalities) + (label,)


# ——————————————————————————————————————————————————————————————————————————————
# SINKHORN PROJECTION
#——————————————————————————————————————————————————————————————————————————————
def sinkhorn_probs(
    matrix: torch.Tensor,
    x1_probs: torch.Tensor,
    x2_probs: torch.Tensor,
    tol: float = 1e-1,
    max_iter: int = 30,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply the Sinkhorn-Knopp algorithm to project a matrix onto a transport polytope.
    Uses out-of-place operations to preserve gradients and chunked checkpointing.
    
    Args:
        matrix: Input matrix to project, shape [m, n]
        x1_probs: Target row sum distribution, shape [m]
        x2_probs: Target column sum distribution, shape [n]
        tol: Convergence tolerance for row/column sums
        max_iter: Maximum number of iterations
        chunk_size: Size of chunks for checkpointing (default: sqrt(max_iter))
        
    Returns:
        Projected matrix that approximately satisfies the marginal constraints
    """
    # Convert to specified precision if using AMP
    if USE_AMP:
        dtype = PRECISION if matrix.is_cuda else torch.float32
    else:
        dtype = torch.float32
    
    # Create copy of matrix with the desired precision to avoid modifying original
    matrix = matrix.to(dtype)
    x1_probs = x1_probs.to(dtype)
    x2_probs = x2_probs.to(dtype)
    
    # Ensure dimensions match for x2_probs
    if x2_probs.size(0) != matrix.size(1):
        if x2_probs.size(0) == 1:
            # Broadcast single probability to match matrix columns
            x2_probs = x2_probs.expand(matrix.size(1))
        else:
            # FIX: Instead of silently truncating, pad or fail depending on circumstances
            if matrix.size(1) > x2_probs.size(0):
                # Pad the probability vector with zeros and renormalize
                padding = torch.zeros(matrix.size(1) - x2_probs.size(0), device=x2_probs.device, dtype=x2_probs.dtype)
                x2_probs_padded = torch.cat([x2_probs, padding], dim=0)
                # Renormalize to ensure probabilities sum to 1
                x2_probs = x2_probs_padded / (x2_probs_padded.sum(dtype=dtype) + 1e-8)
            else:
                # Only truncate if using chunk processing and this is the last chunk
                # Otherwise, dimension mismatch likely indicates a bug
                if matrix.size(1) < x2_probs.size(0) - 128:  # Reasonable chunk size threshold
                    raise ValueError(f"Dimension mismatch in Sinkhorn projection: matrix columns ({matrix.size(1)}) << x2_probs size ({x2_probs.size(0)}). This is likely an error.")
                # Truncate with a warning
                x2_probs = x2_probs[:matrix.size(1)]
                # Renormalize after truncation
                x2_probs = x2_probs / (x2_probs.sum(dtype=dtype) + 1e-8)
                #print(f"Warning: Truncated x2_probs in Sinkhorn from {x2_probs.size(0)} to {matrix.size(1)} elements and renormalized.")
    
    # Ensure dimensions match for x1_probs
    if x1_probs.size(0) != matrix.size(0):
        if x1_probs.size(0) == 1:
            # Broadcast single probability to match matrix rows
            x1_probs = x1_probs.expand(matrix.size(0))
        else:
            # FIX: Instead of silently truncating, pad or fail depending on circumstances
            if matrix.size(0) > x1_probs.size(0):
                # Pad the probability vector with zeros and renormalize
                padding = torch.zeros(matrix.size(0) - x1_probs.size(0), device=x1_probs.device, dtype=x1_probs.dtype)
                x1_probs_padded = torch.cat([x1_probs, padding], dim=0)
                # Renormalize to ensure probabilities sum to 1
                x1_probs = x1_probs_padded / (x1_probs_padded.sum(dtype=dtype) + 1e-8)
            else:
                # Only truncate if using chunk processing and this is the last chunk
                # Otherwise, dimension mismatch likely indicates a bug
                if matrix.size(0) < x1_probs.size(0) - 128:  # Reasonable chunk size threshold
                    raise ValueError(f"Dimension mismatch in Sinkhorn projection: matrix rows ({matrix.size(0)}) << x1_probs size ({x1_probs.size(0)}). This is likely an error.")
                # Truncate with a warning
                x1_probs = x1_probs[:matrix.size(0)]
                # Renormalize after truncation
                x1_probs = x1_probs / (x1_probs.sum(dtype=dtype) + 1e-8)
                print(f"Warning: Truncated x1_probs in Sinkhorn from {x1_probs.size(0)} to {matrix.size(0)} elements and renormalized.")
    
    def sinkhorn_chunk(mat: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Run Sinkhorn iterations for a chunk of steps."""
        for _ in range(n_steps):
            # Normalize columns to match x2_probs - using out-of-place operations
            col_sums = mat.sum(dim=0, keepdim=True, dtype=dtype)
            mat = mat / (col_sums + 1e-8) * x2_probs.unsqueeze(0)
            
            # Normalize rows to match x1_probs - using out-of-place operations
            row_sums = mat.sum(dim=1, keepdim=True, dtype=dtype)
            mat = mat / (row_sums + 1e-8) * x1_probs.unsqueeze(1)
        return mat
    
    # Calculate chunk size (approximately sqrt of max_iter for balanced memory/compute)
    if chunk_size is None:
        chunk_size = int(max_iter ** 0.5)
    
    # Process chunks with checkpointing
    done = 0
    while done < max_iter:
        steps = min(chunk_size, max_iter - done)
        
        # Checkpoint the entire chunk with proper binding of steps
        matrix = torch.utils.checkpoint.checkpoint(
            lambda m, s=steps: sinkhorn_chunk(m, s),
            matrix,
            use_reentrant=False
        )
        done += steps
        
        # Check convergence of both marginals
        if (torch.allclose(matrix.sum(dim=1, dtype=dtype), x1_probs, atol=tol) and
            torch.allclose(matrix.sum(dim=0, dtype=dtype), x2_probs, atol=tol)):
            break
        
        # Explicitly trigger garbage collection every few chunks
        if done % (chunk_size * MEMORY_CLEANUP_INTERVAL) == 0 and AGGRESSIVE_CLEANUP:
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    return matrix.to(torch.float32)  # Convert back to float32 for downstream compatibility


# ——————————————————————————————————————————————————————————————————————————————
# MLP BUILDER
#——————————————————————————————————————————————————————————————————————————————
def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    activation: str
) -> nn.Sequential:
    """
    Create a configurable multi-layer perceptron (MLP) neural network.
    
    The network consists of:
    1. An input layer mapping from input_dim to hidden_dim
    2. A configurable number of hidden layers with the specified activation function
    3. An output layer mapping from hidden_dim to output_dim
    
    Args:
        input_dim: Dimension of the input features
        hidden_dim: Dimension of the hidden layers
        output_dim: Dimension of the output layer
        layers: Number of hidden layers (excluding input and output layers)
        activation: Activation function to use ('relu' or 'tanh')
        
    Returns:
        A PyTorch Sequential module implementing the specified MLP
    """
    # Map activation string to PyTorch activation class
    act_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }
    
    if activation not in act_map:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(act_map.keys())}")
    
    act_layer = act_map[activation]
    
    # Build network layers
    modules = [
        nn.Linear(input_dim, hidden_dim),
        act_layer()
    ]
    
    # Add hidden layers
    for _ in range(layers):
        modules.extend([
            nn.Linear(hidden_dim, hidden_dim),
            act_layer()
        ])
    
    # Add output layer (no activation on final layer)
    modules.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*modules).to(device)


# ——————————————————————————————————————————————————————————————————————————————
# SIMPLE TABULAR DISCRIMINATOR (counts → empirical log‐probs)
#——————————————————————————————————————————————————————————————————————————————
def simple_discrim(
    xs: List[torch.Tensor],
    y: torch.Tensor,
    num_labels: int
) -> Callable[[torch.Tensor, ...], torch.Tensor]:
    """
    Create a simple tabular discriminator using counts to estimate log probabilities.
    
    This discriminator counts co-occurrences of discretized inputs (one-hot encoded) 
    and labels to estimate the conditional probability distribution p(y|x). It's a
    non-parametric alternative to neural network discriminators.
    
    Args:
        xs: List of one-hot modality tensors, each of shape [N, dim_i]
        y: Tensor of shape [N, 1], with integer labels in range [0, num_labels)
        num_labels: Number of possible label values
        
    Returns:
        A callable function f(*inputs) that returns log conditional probabilities log p(y|inputs),
        properly normalized so that for each input x, the sum over all y values is 1.
    """
    # Create tensor shape based on input dimensions and number of labels
    shape = [x.size(1) for x in xs] + [num_labels]
    
    # Initialize probability table with small non-zero values to avoid log(0)
    p = torch.ones(*shape, device=device) * 1e-8
    
    # Count co-occurrences of input indices and labels
    for i in range(y.size(0)):
        # Get indices of maximum values (assumed one-hot) for each input
        input_indices = [torch.argmax(x[i]).item() for x in xs]
        # Add label index
        indices = input_indices + [y[i].item()]
        # Increment count for this combination
        p[tuple(indices)] += 1
    
    # BUG FIX: We need to normalize per input x, not globally
    # Loop through all possible input combinations
    input_shapes = [x.size(1) for x in xs]
    input_indices_list = list(itertools.product(*[range(s) for s in input_shapes]))
    
    # Normalize each conditional distribution p(y|x)
    for input_indices in input_indices_list:
        # Get the slice of p corresponding to this input combination
        idx = list(input_indices)
        # Sum over all possible y values for this x
        marginal_sum = p[tuple(idx + [slice(None)])].sum()
        if marginal_sum > 0:
            # Normalize to get p(y|x)
            p[tuple(idx + [slice(None)])] /= marginal_sum
    
    def discriminator_function(*inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of labels given inputs.
        
        Args:
            *inputs: Input tensors, one for each modality
            
        Returns:
            Log probability tensor
        """
        # Get indices of maximum values for each input
        indices = [torch.argmax(inp, dim=1) for inp in inputs]
        # Look up log probability
        return torch.log(p[tuple(indices)])
    
    return discriminator_function


# ——————————————————————————————————————————————————————————————————————————————
# LEARNED DISCRIMINATOR NETWORK
#——————————————————————————————————————————————————————————————————————————————
class Discrim(nn.Module):
    """
    Neural network discriminator for predicting class labels from domain features.
    
    This discriminator uses a multi-layer perceptron to learn the conditional
    probability distribution p(y|x) for classification tasks.
    
    Attributes:
        mlp: Neural network module that maps input features to class logits
    """
    
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        num_labels: int,
        layers: int,
        activation: str
    ):
        """
        Initialize a discriminator network.
        
        Args:
            x_dim: Total dimension of input features (sum of all modality dimensions)
            hidden_dim: Dimension of hidden layers
            num_labels: Number of output classes
            layers: Number of hidden layers
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)
        
    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits for given input features.
        
        Args:
            *xs: Input tensors from different modalities
            
        Returns:
            Logits tensor of shape [batch_size, num_labels]
        """
        # Concatenate all inputs along feature dimension
        x = torch.cat(xs, dim=-1)
        return self.mlp(x)


# ——————————————————————————————————————————————————————————————————————————————
# CORE ALIGNMENT MODULE
#——————————————————————————————————————————————————————————————————————————————
class CEAlignment(nn.Module):
    """
    Conditional Entropy Alignment module for PID estimation.
    
    This module learns to align probability distributions between domains using
    a transport-based approach, mapping from domain representations to a shared
    embedding space conditioned on labels.
    
    Attributes:
        num_labels: Number of possible label values
        mlp1: Neural network mapping first domain to embedding space
        mlp2: Neural network mapping second domain to embedding space
    """
    
    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_labels: int,
        layers: int,
        activation: str
    ):
        """
        Initialize CE Alignment module.
        
        Args:
            x1_dim: Dimension of the first domain's features
            x2_dim: Dimension of the second domain's features
            hidden_dim: Dimension of hidden layers in MLPs
            embed_dim: Dimension of label-conditioned embeddings
            num_labels: Number of possible label values
            layers: Number of hidden layers in MLPs
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()
        self.num_labels = num_labels
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        p_y_x1: torch.Tensor,
        p_y_x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment between domain representations.
        
        Args:
            x1: First domain features, shape [batch_size, x1_dim]
            x2: Second domain features, shape [batch_size, x2_dim]
            p_y_x1: Conditional probabilities p(y|x1), shape [batch_size, num_labels]
            p_y_x2: Conditional probabilities p(y|x2), shape [batch_size, num_labels]
            
        Returns:
            Coupling matrix of shape [batch_size, batch_size, num_labels]
        """
        batch_size = x1.size(0)
        
        # Ensure x2 has the same batch size as x1
        if x2.size(0) != batch_size:
            x2 = x2[:batch_size]
            p_y_x2 = p_y_x2[:batch_size]
        
        # Project into label-conditioned embeddings
        # Shape: [batch_size, num_labels, embed_dim]
        q1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1))
        q2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))

        # Normalize embeddings for numerical stability
        q1 = (q1 - q1.mean(dim=2, keepdim=True)) / torch.sqrt(q1.var(dim=2, keepdim=True) + 1e-8)
        q2 = (q2 - q2.mean(dim=2, keepdim=True)) / torch.sqrt(q2.var(dim=2, keepdim=True) + 1e-8)

        # Compute affinity for each label across all pairs of samples
        # Shape: [batch_size, batch_size, num_labels]
        aff = torch.einsum('bce, dce -> bdc', q1, q2) / math.sqrt(q1.size(-1))
        
        # Apply max-stable trick to prevent numerical overflow in exp()
        # 1. For each label c, subtract the per-label max so the largest exponent is zero
        aff_max = aff.reshape(-1, aff.size(-1)).max(dim=0)[0]  # shape [num_labels]
        aff_centered = aff - aff_max.view(1, 1, -1)  # broadcast to [batch_size, batch_size, num_labels]
        
        # 2. Exponentiate the centered affinities (no overflow risk now)
        A = torch.exp(aff_centered)  # Now no entry exceeds exp(0)=1

        # Apply Sinkhorn projection for each label
        couplings = []
        for c in range(self.num_labels):
            # Use empirical total mass of each label slice instead of normalizing to 1
            # Get the total mass from each conditional distribution
            mass1 = p_y_x1[:, c].sum() + 1e-8   # ≈ batch_size · p(y=c) 
            mass2 = p_y_x2[:, c].sum() + 1e-8   # ≈ batch_size · p(y=c)
            
            # Use the average mass as target to preserve the true label-frequency information
            target = 0.5 * (mass1 + mass2)
            
            # Rescale the input distributions to have matching masses
            p_y_x1_c = p_y_x1[:, c] * (target / mass1)
            p_y_x2_c = p_y_x2[:, c] * (target / mass2)
            
            # Project to transport polytope with marginals p(y=c|x1) and p(y=c|x2)
            coupling_c = sinkhorn_probs(
                A[..., c],
                p_y_x1_c,
                p_y_x2_c
            )
            couplings.append(coupling_c)
            
        # Stack along label dimension
        # Shape: [batch_size, batch_size, num_labels]
        P = torch.stack(couplings, dim=-1)
        
        # Ensure it's a true probability distribution
        P = P / (P.sum() + 1e-8)
        
        return P


# ——————————————————————————————————————————————————————————————————————————————
# FULL PID‑INFORMATION MODULE
#——————————————————————————————————————————————————————————————————————————————
class CEAlignmentInformation(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, discrim_1, discrim_2, discrim_12, p_y):
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        
        # Store discriminators
        self.discrim_1 = discrim_1
        self.discrim_2 = discrim_2
        self.discrim_12 = discrim_12
        
        # Freeze pre-trained discriminators
        for D in (self.discrim_1, self.discrim_2, self.discrim_12):
            D.eval()
            for p in D.parameters():
                p.requires_grad = False
                
        # marginal p(y)
        self.register_buffer('p_y', p_y)
        
        # For mixed precision
        self.use_amp = USE_AMP  # Use the global USE_AMP variable
        # Create GradScaler for mixed precision training
        if self.use_amp:
            self.scaler = amp.GradScaler()  # Remove device parameter
        
        # For memory efficiency - use global parameters
        self.chunk_size = CHUNK_SIZE
        self.aggressive_cleanup = AGGRESSIVE_CLEANUP
        self.memory_cleanup_interval = MEMORY_CLEANUP_INTERVAL

    def forward(self, x1, x2, y):
        """
        Compute PID components between domain representations.
        
        Args:
            x1: First domain features, shape [batch_size, x1_dim]
            x2: Second domain features, shape [batch_size, x2_dim]
            y: Labels, shape [batch_size]
            
        Returns:
            Tuple of (loss, pid_vals, P) where:
            - loss: Scalar tensor for optimization
            - pid_vals: Tensor of shape [4] containing [redundancy, unique1, unique2, synergy]
            - P: Coupling matrix
        """
        batch_size = x1.size(0)
        
        # Forward pass with mixed precision if enabled
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with amp.autocast(device_type):  # Use consistent style with other autocast calls
            # Get label conditional probabilities using discriminators with proper softmax
            p_y_x1 = F.softmax(self.discrim_1(x1), dim=1)  # [batch, num_labels]
            p_y_x2 = F.softmax(self.discrim_2(x2), dim=1)  # [batch, num_labels]
            p_y_x1x2 = F.softmax(self.discrim_12(torch.cat([x1, x2], dim=-1)), dim=1)  # [batch, num_labels]
        
        # Calculate unimodal mutual information terms with explicit dtype for numerical stability
        mi_x1_y = torch.sum(p_y_x1 * torch.log(p_y_x1 / self.p_y.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        mi_x2_y = torch.sum(p_y_x2 * torch.log(p_y_x2 / self.p_y.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        
        # Get coupling matrix from align
        P = self.align(x1, x2, p_y_x1, p_y_x2)  # [batch, batch, num_labels]
        
        # 1) Normalize along the label axis to get q̃(y|x1,x2)
        P_cond = P / (P.sum(dim=-1, keepdim=True) + 1e-8)  # [batch, batch, num_labels]
        
        # 2) Compute the joint mutual information properly using p(y) as denominator
        # Expand p_y for broadcasting
        p_y_expanded = self.p_y.view(1, 1, -1)  # [1, 1, num_labels]
        
        # Calculate proper log ratio for joint MI
        log_ratio = torch.log(P_cond + 1e-8) - torch.log(p_y_expanded + 1e-8)  # [batch, batch, num_labels]
        
        # 3) Compute joint MI by summing over all dimensions, weighted by joint coupling P
        mi_x1x2_y = (P * log_ratio).sum(dim=[1, 2])  # [batch]
        
        # For comparison - calculate joint MI from discriminator (not used in updated PID calculation)
        mi_discrim_x1x2_y = torch.sum(p_y_x1x2 * torch.log(p_y_x1x2 / self.p_y.unsqueeze(0) + 1e-8), 
                                     dim=1, dtype=torch.float32)
        
        # Calculate PID components using the Möbius relations
        # Redundancy = I(X1;Y) + I(X2;Y) - I(X1,X2;Y)
        redundancy = torch.clamp(mi_x1_y + mi_x2_y - mi_x1x2_y, min=0)
        # Unique1 = I(X1;Y) - Redundancy
        unique1 = torch.clamp(mi_x1_y - redundancy, min=0)
        # Unique2 = I(X2;Y) - Redundancy
        unique2 = torch.clamp(mi_x2_y - redundancy, min=0)
        # Synergy = I(X1,X2;Y) - I(X1;Y) - I(X2;Y) + Redundancy
        synergy = torch.clamp(mi_x1x2_y - mi_x1_y - mi_x2_y + redundancy, min=0)
        
        # Calculate scalar loss for optimization (negative because we're maximizing)
        mi_q_y_x1x2 = mi_x1x2_y.mean()
        loss = -mi_q_y_x1x2
        
        # Final cleanup
        if self.aggressive_cleanup and torch.cuda.is_available():
            if torch.cuda.current_stream().query() and self.memory_cleanup_interval > 0:
                torch.cuda.empty_cache()
        
        # Return loss, PID components, and coupling matrix
        pid_vals = torch.stack([redundancy.mean(), unique1.mean(), unique2.mean(), synergy.mean()])
        return loss, pid_vals, P


# ——————————————————————————————————————————————————————————————————————————————
# TRAIN/EVAL LOOPS FOR DISCRIMINATORS & ALIGNMENT
#——————————————————————————————————————————————————————————————————————————————
def train_discrim(model, loader, optimizer, data_type, num_epoch=40, wandb_prefix=None, use_compile=True):
    """Train a Discrim on (X → Y). data_type tells which fields of the batch are features/labels."""
    model.train()
    
    # Apply torch.compile if requested
    if use_compile:
        model = torch.compile(model)
        print("Applied torch.compile to discriminator model")
    
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in loader:
            optimizer.zero_grad()
            # unpack features & label
            xs = [batch[i].float().to(device) for i in data_type[0]]
            y = batch[data_type[1][0]].long().to(device)
            
            # Handle y shape - convert to 1D for CrossEntropyLoss
            if y.dim() > 1:
                if y.size(1) == 1:  # If shape is [N, 1], can simply squeeze
                    y = y.squeeze(-1)
                else:  # If shape is [N, D], we need to handle differently
                    # For this code, we'll use the first column as the target 
                    # but this might need adjustment based on your data
                    y = y[:, 0]
            
            # Get number of classes from model's output layer
            num_classes = model.mlp[-1].out_features
            
            # Ensure labels are non-negative and in the valid range [0, num_classes-1]
            if y.min() < 0 or y.max() >= num_classes:
                # Convert labels to be in range [0, num_classes-1]
                if y.min() < 0:
                    # Shift labels to be non-negative
                    offset = -y.min().item()
                    y = y + offset
                # Then clamp to valid range
                y = torch.clamp(y, 0, num_classes-1)
            
            # Forward pass
            logits = model(*xs)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total
        
        print(f"Discrim Train Epoch {epoch+1}/{num_epoch}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        
        # Log to wandb if enabled
        if HAS_WANDB and wandb_prefix is not None:
            wandb.log({
                f"{wandb_prefix}/train_loss": avg_loss,
                f"{wandb_prefix}/train_acc": accuracy,
                f"{wandb_prefix}/epoch": epoch,
            })
    
    # Return the trained model
    return model

def eval_discrim(model, loader, data_type, wandb_prefix=None):
    model.eval()
    with torch.no_grad():
        total, acc = 0, 0
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in loader:
            xs = [batch[i].float().to(device) for i in data_type[0]]
            y = batch[data_type[1][0]].long().to(device)
            
            # Handle y shape - convert to 1D for evaluation
            if y.dim() > 1:
                if y.size(1) == 1:  # If shape is [N, 1], can simply squeeze
                    y = y.squeeze(-1)
                else:  # If shape is [N, D], we need to handle differently
                    # For this code, we'll use the first column as the target
                    y = y[:, 0]
            
            # Get number of classes from model's output layer
            num_classes = model.mlp[-1].out_features
            
            # Ensure labels are non-negative and in the valid range [0, num_classes-1]
            if y.min() < 0 or y.max() >= num_classes:
                # Convert labels to be in range [0, num_classes-1]
                if y.min() < 0:
                    # Shift labels to be non-negative
                    offset = -y.min().item()
                    y = y + offset
                # Then clamp to valid range
                y = torch.clamp(y, 0, num_classes-1)
            
            logits = model(*xs)
            loss = nn.CrossEntropyLoss()(logits, y)
            test_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            total += y.size(0)
            acc += (preds == y).sum().item()
            
            # Collect predictions and targets for confusion matrix
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
        
        accuracy = acc / total
        avg_loss = test_loss / len(loader)
        print(f"Discrim eval acc: {accuracy:.3f}, loss: {avg_loss:.4f}")
        
        # Log to wandb if enabled
        if HAS_WANDB and wandb_prefix is not None:
            wandb.log({
                f"{wandb_prefix}/eval_loss": avg_loss,
                f"{wandb_prefix}/eval_acc": accuracy,
            })
            
            # Create confusion matrix if we have wandb
            if len(all_preds) > 0 and len(all_targets) > 0:
                try:
                    all_preds = torch.cat(all_preds).numpy()
                    all_targets = torch.cat(all_targets).numpy()
                    wandb.log({f"{wandb_prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=all_targets,
                        preds=all_preds,
                        class_names=[str(i) for i in range(num_classes)]
                    )})
                except Exception as e:
                    print(f"Warning: Could not log confusion matrix to wandb: {e}")
        
        return accuracy, avg_loss

def train_ce_alignment(model, loader, optimizer_class, num_epoch=10, wandb_prefix=None, step_offset=0, use_compile=True):
    """Train a CEAlignment model on data from loader."""
    device = next(model.parameters()).device
    model.train()
    
    # Apply torch.compile if requested
    if use_compile:
        model = torch.compile(model)
        print("Applied torch.compile to CE alignment model")
    
    # Initialize optimizer with only the alignment parameters
    optimizer = optimizer_class(model.align.parameters(), lr=1e-3)
    
    # Calculate total batches and wandb step offset
    num_batches = len(loader)
    wandb_step_offset = step_offset if wandb_prefix else 0
    
    # Initialize statistics
    avg_loss = 0.0
    
    # Train for specified number of epochs
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        total_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            # Extract data
            x1_batch, x2_batch, y_batch = batch
            x1_batch, x2_batch, y_batch = x1_batch.to(device), x2_batch.to(device), y_batch.to(device)
            
            # Forward pass
            loss_tuple = model(x1_batch, x2_batch, y_batch)
            
            # Unpack the tuple - model returns (loss, pid_vals, P)
            loss = loss_tuple[0]  # Get just the loss for backward pass
            pid_vals = loss_tuple[1]  # Get PID values for logging
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            total_batches += 1
            
            # Log every 30 batches
            if HAS_WANDB and wandb_prefix is not None and (batch_idx + 1) % 10 == 0:
                # Calculate global step
                global_step = epoch * num_batches + (batch_idx + 1) + wandb_step_offset
                
                # Log to wandb - using the pid_vals from the forward pass
                wandb.log({
                    f"{wandb_prefix}/batch_loss":        loss.item(),
                    f"{wandb_prefix}/redundancy":        pid_vals[0].item(),
                    f"{wandb_prefix}/unique1":           pid_vals[1].item(),
                    f"{wandb_prefix}/unique2":           pid_vals[2].item(),
                    f"{wandb_prefix}/synergy":           pid_vals[3].item(),
                    "epoch":                             epoch + 1,
                    "batch_in_epoch":                    batch_idx + 1,
                }, step=global_step)
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / total_batches
        avg_loss += avg_epoch_loss
        
        # Log to wandb if requested with adjusted step
        if wandb_prefix is not None and wandb.run is not None:
            wandb.log({
                f"{wandb_prefix}/train_loss": avg_epoch_loss,
                f"{wandb_prefix}/epoch": epoch,
            }, step=epoch + step_offset)
        
        print(f"Epoch {epoch+1}/{num_epoch} - Loss: {avg_epoch_loss:.6f}")
    
    # Calculate overall average loss
    avg_loss /= num_epoch
    
    print(f"Training completed - Average Loss: {avg_loss:.6f}")
    return model

def eval_ce_alignment(model, loader, wandb_prefix=None, step_offset=0):
    model.eval()
    results = []
    aligns  = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            x1 = batch[0].float().to(device)
            x2 = batch[1].float().to(device)
            y  = batch[2].long().to(device)
            loss, pid_vals, P = model(x1, x2, y)
            results.append(pid_vals.cpu())
            aligns.append(P.cpu())
            total_loss += loss.item()
    
    # Calculate average metrics
    avg_loss = total_loss / len(loader)
    avg_results = torch.stack(results).mean(dim=0)
    std_results = torch.stack(results).std(dim=0)
    
    # Log to wandb if enabled
    if HAS_WANDB and wandb_prefix is not None:
        wandb.log({
            f"{wandb_prefix}/eval_loss": avg_loss,
            f"{wandb_prefix}/redundancy": avg_results[0].item(),
            f"{wandb_prefix}/unique1": avg_results[1].item(),
            f"{wandb_prefix}/unique2": avg_results[2].item(),
            f"{wandb_prefix}/synergy": avg_results[3].item(),
            f"{wandb_prefix}/redundancy_std": std_results[0].item(),
            f"{wandb_prefix}/unique1_std": std_results[1].item(),
            f"{wandb_prefix}/unique2_std": std_results[2].item(),
            f"{wandb_prefix}/synergy_std": std_results[3].item(),
        }, step=step_offset)
        
        # Create PID distribution plot
        fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
        pid_components = ['Redundancy', 'Unique1', 'Unique2', 'Synergy']
        ax.bar(pid_components, avg_results.numpy(), yerr=std_results.numpy(), capsize=10, alpha=0.7)
        ax.set_ylabel('Information (bits)')
        ax.set_title('PID Components Distribution')
        ax.grid(alpha=0.3)
        wandb.log({f"{wandb_prefix}/pid_distribution": wandb.Image(fig)}, step=step_offset)
        plt.close(fig)
    
    return torch.stack(results), aligns


# ——————————————————————————————————————————————————————————————————————————————
# HIGH‑LEVEL WRAPPER
#——————————————————————————————————————————————————————————————————————————————
def critic_ce_alignment(x1, x2, labels, num_labels,
                        train_ds, test_ds,
                        discrim_1=None, discrim_2=None, discrim_12=None,
                        learned_discrim=True, shuffle=True,
                        discrim_epochs=40, ce_epochs=10, 
                        wandb_enabled=False, model_name=None,
                        discrim_hidden_dim=64, discrim_layers=5,
                        use_compile=True):
    """
    Core function for Partial Information Decomposition via Conditional Entropy alignment.
    
    This function:
    1. Trains discriminators to predict labels from domain features (if learned_discrim=True)
    2. Trains an alignment model to align conditional distributions across domains
    3. Calculates PID components (redundancy, unique info, synergy) between domains
    
    Args:
        x1: First domain features, shape [batch_size, x1_dim]
        x2: Second domain features, shape [batch_size, x2_dim]
        labels: Integer labels, shape [batch_size]
        num_labels: Number of possible label values
        train_ds: Training dataset (MultimodalDataset)
        test_ds: Test dataset (MultimodalDataset)
        discrim_1: Optional pre-trained discriminator for first domain
        discrim_2: Optional pre-trained discriminator for second domain
        discrim_12: Optional pre-trained joint discriminator
        learned_discrim: Whether to train discriminators or use simple count-based ones
        shuffle: Whether to shuffle the training dataset
        discrim_epochs: Number of epochs to train discriminators
        ce_epochs: Number of epochs to train CE alignment
        wandb_enabled: Whether to log to Weights & Biases
        model_name: Optional name for the model (for logging)
        discrim_hidden_dim: Hidden dimension for discriminator networks
        discrim_layers: Number of layers in discriminator networks
        use_compile: Whether to use torch.compile for model optimization
        
    Returns:
        Tuple of (PID components, mutual information values, coupling matrices, discriminators)
    """
    # Setup device
    if torch.cuda.is_available():
        if not x1.is_cuda:
            x1 = x1.cuda()
        if not x2.is_cuda:
            x2 = x2.cuda()
        if not labels.is_cuda:
            labels = labels.cuda()
    
    # For mixed precision
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = amp.GradScaler()  # Remove device parameter

    # Configure torch.compiler if compilation is requested
    if use_compile and torch.cuda.is_available():
        # Set dynamo config for better compilation
        #torch.dynamo_config.cache_size_limit = 128  # Increase cache size
        # Print diagnostic info about compilation
        print("Torch compiler mode:", torch._dynamo.config.get())
        print("Torch compile backend:", torch._dynamo.config.get_backend())
    
    # Initialize discriminators
    if learned_discrim:
        print("\nBuilding discriminators...")
        # build three Discrims
        d1 = Discrim(x1.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(device)
        d2 = Discrim(x2.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(device)
        d12= Discrim(x1.size(1)+x2.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(device)
        
        # Apply torch.compile if requested - AFTER moving to device
        if use_compile and torch.cuda.is_available():
            print("Applying torch.compile to discriminators...")
            d1 = torch.compile(d1)
            d2 = torch.compile(d2)
            d12 = torch.compile(d12)
        
        print(f"Discriminator input sizes: d1={x1.size(1)}, d2={x2.size(1)}, d12={x1.size(1)+x2.size(1)}")

        loaders = [
          (d1, ([0], [2]), "discrim_domain1"),   # x1 at [0], y at [2]
          (d2, ([1], [2]), "discrim_domain2"),   # x2 at [1], y at [2]
          (d12, ([0,1], [2]), "discrim_joint"),  # (x1,x2) at [0,1], y at [2]
        ]
        
        # Train each discriminator sequentially with memory cleanup between them
        for idx, (model, dt, prefix) in enumerate(loaders):
            print(f"\nTraining discriminator {idx+1}/3 ({prefix})...")
            wandb_prefix = f"{prefix}_{model_name}" if wandb_enabled and model_name else None
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            dl  = DataLoader(train_ds, batch_size=256, shuffle=shuffle, 
                            num_workers=4, pin_memory=True, prefetch_factor=2, 
                            persistent_workers=True if torch.cuda.is_available() else False)
            
            # Train with mixed precision
            model.train()
            for epoch in range(discrim_epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, batch in enumerate(dl):
                    opt.zero_grad()
                    # unpack features & label
                    xs = [batch[i].float().to(device) for i in dt[0]]
                    
                    # Get the label from the specified label index
                    y = batch[dt[1][0]].long().to(device)
                    
                    # Handle y shape - ensure it's 1D
                    if y.dim() > 1:
                        y = y.squeeze()
                    
                    # Use mixed precision
                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                    if USE_AMP:
                        with amp.autocast(device_type):
                            if len(xs) == 1:
                                logits = model(xs[0])
                            else:
                                logits = model(*xs)
                            loss = nn.CrossEntropyLoss()(logits, y)
                        
                        # Scale loss and backprop
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        # Use regular precision
                        if len(xs) == 1:
                            logits = model(xs[0])
                        else:
                            logits = model(*xs)
                        loss = nn.CrossEntropyLoss()(logits, y)
                        loss.backward()
                        opt.step()
                    
                    # Track metrics with regular precision
                    epoch_loss += loss.item()
                    with torch.no_grad():
                        preds = logits.argmax(dim=-1)
                        total += y.size(0)
                        correct += (preds == y).sum().item()
                    
                    # Clear memory
                    del xs, y, logits
                    
                # Calculate epoch metrics
                avg_loss = epoch_loss / len(dl)
                accuracy = correct / total
                
                print(f"Discrim {idx+1}/3 Train Epoch {epoch+1}/{discrim_epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
                
                # Log to wandb if enabled
                if HAS_WANDB and wandb_prefix is not None:
                    wandb.log({
                        f"{wandb_prefix}/train_loss": avg_loss,
                        f"{wandb_prefix}/train_acc": accuracy,
                        f"{wandb_prefix}/epoch": epoch,
                    })
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                total, acc = 0, 0
                test_loss = 0.0
                test_dl = DataLoader(test_ds, batch_size=256, shuffle=False,
                                    num_workers=4, pin_memory=True)
                
                for batch in test_dl:
                    xs = [batch[i].float().to(device) for i in dt[0]]
                    
                    # Get the label from the specified label index
                    y = batch[dt[1][0]].long().to(device)
                    
                    # Handle y shape - ensure it's 1D
                    if y.dim() > 1:
                        y = y.squeeze()
                    
                    if len(xs) == 1:
                        logits = model(xs[0])
                    else:
                        logits = model(*xs)
                    loss = nn.CrossEntropyLoss()(logits, y)
                    test_loss += loss.item()
                    
                    preds = logits.argmax(dim=-1)
                    total += y.size(0)
                    acc += (preds == y).sum().item()
                    
                    # Clear memory
                    del xs, y, logits
                
                accuracy = acc / total
                avg_loss = test_loss / len(test_dl)
                print(f"Discrim {idx+1}/3 eval acc: {accuracy:.3f}, loss: {avg_loss:.4f}")
                
                # Log to wandb if enabled
                if HAS_WANDB and wandb_prefix is not None:
                    wandb.log({
                        f"{wandb_prefix}/eval_loss": avg_loss,
                        f"{wandb_prefix}/eval_acc": accuracy,
                    })
            
            # Clear memory between training discriminators
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

    else:
        print("\nUsing simple discriminators...")
        d1  = simple_discrim([x1], labels, num_labels)
        d2  = simple_discrim([x2], labels, num_labels)
        d12 = simple_discrim([x1, x2], labels, num_labels)

    # 2) build p_y
    print("\nBuilding p_y...")
    with torch.no_grad():
        # Ensure labels are 1D for one-hot encoding
        labels_1d = labels.view(-1)
        print(f"Labels for p_y: shape={labels_1d.shape}, unique values={torch.unique(labels_1d).tolist()}")
        one_hot = F.one_hot(labels_1d, num_labels).float()
        p_y = one_hot.sum(dim=0) / labels_1d.size(0)
        print(f"p_y shape: {p_y.shape}")
        del one_hot, labels_1d

    # 3) instantiate CEAlignmentInformation
    print("\nCreating CEAlignmentInformation model...")
    model = CEAlignmentInformation(
        x1_dim=x1.size(1), x2_dim=x2.size(1),
        hidden_dim=discrim_hidden_dim, embed_dim=10, num_labels=num_labels,
        layers=discrim_layers, activation='relu',
        discrim_1=d1, discrim_2=d2, discrim_12=d12,
        p_y=p_y
    ).to(device)
    
    # Apply torch.compile to the alignment model - AFTER moving to device
    if use_compile and torch.cuda.is_available():
        print("Applying torch.compile to CE alignment model...")
        model = torch.compile(model)

    opt = torch.optim.Adam(model.align.parameters(), lr=1e-3)

    # 4) train the alignment with mixed precision
    print("\nTraining CE alignment...")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=shuffle,
                             num_workers=4, pin_memory=True, prefetch_factor=2,
                             persistent_workers=True if torch.cuda.is_available() else False)
    wandb_prefix = f"ce_alignment_{model_name}" if wandb_enabled and model_name else None
    
    # Calculate step offset based on discriminator training
    # Each discriminator trains for discrim_epochs, and there are 3 discriminators
    wandb_step_offset = discrim_epochs * 3 if wandb_enabled else 0
    
    model.train()
    for epoch in range(ce_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            opt.zero_grad()
            
            # Get data
            x1_batch = batch[0].float().to(device)
            x2_batch = batch[1].float().to(device)
            
            # Get label from the specified label index
            y_batch = batch[2].long().to(device)
            
            # Handle y shape - ensure it's 1D
            if y_batch.dim() > 1:
                y_batch = y_batch.squeeze()
            
            # Use mixed precision
            try:
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Validate shapes before processing
                if x1_batch.size(0) != y_batch.size(0) or x2_batch.size(0) != y_batch.size(0):
                    print(f"WARNING: Batch size mismatch: x1={x1_batch.size(0)}, x2={x2_batch.size(0)}, y={y_batch.size(0)}")
                    # Make sure all have the same batch size by trimming
                    min_batch = min(x1_batch.size(0), x2_batch.size(0), y_batch.size(0))
                    x1_batch = x1_batch[:min_batch]
                    x2_batch = x2_batch[:min_batch]
                    y_batch = y_batch[:min_batch]
                    print(f"Adjusted to min batch size: {min_batch}")
                
                if USE_AMP:
                    with amp.autocast(device_type):
                        # Unpack the model's forward return values correctly
                        loss, pid_vals, _ = model(x1_batch, x2_batch, y_batch)
                    
                    # Scale loss and backprop
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    # Use regular precision
                    loss, pid_vals, _ = model(x1_batch, x2_batch, y_batch)
                    loss.backward()
                    opt.step()
                
                epoch_loss += loss.item()
            except Exception as e:
                print(f"ERROR in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next batch
                continue
            
            # Clear memory
            del x1_batch, x2_batch, y_batch
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"CE Alignment Epoch {epoch+1}/{ce_epochs}, Loss: {avg_loss:.4f}")
        
        # Log to wandb if enabled
        if HAS_WANDB and wandb_prefix is not None:
            wandb.log({
                f"{wandb_prefix}/train_loss": avg_loss,
                f"{wandb_prefix}/epoch": epoch,
            }, step=epoch + wandb_step_offset)
        
        # Clear memory between epochs
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

    # 5) eval
    print("\nEvaluating CE alignment...")
    model.eval()
    
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    test_loss = 0
    all_pids = []
    
    # Use the same step offset for evaluation as for training
    if HAS_WANDB and wandb_prefix is not None:
        # For evaluation metrics
        wandb.log({
            f"{wandb_prefix}/eval_step": wandb_step_offset + ce_epochs
        }, step=wandb_step_offset + ce_epochs)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                x1_batch = batch[0].float().to(device)
                x2_batch = batch[1].float().to(device)
                y_batch = batch[2].long().to(device)
                
                # Handle y shape - ensure it's 1D
                if y_batch.dim() > 1:
                    y_batch = y_batch.squeeze()
                
                # Validate shapes before processing
                if x1_batch.size(0) != y_batch.size(0) or x2_batch.size(0) != y_batch.size(0):
                    print(f"WARNING: Eval batch size mismatch: x1={x1_batch.size(0)}, x2={x2_batch.size(0)}, y={y_batch.size(0)}")
                    # Make sure all have the same batch size by trimming
                    min_batch = min(x1_batch.size(0), x2_batch.size(0), y_batch.size(0))
                    x1_batch = x1_batch[:min_batch]
                    x2_batch = x2_batch[:min_batch]
                    y_batch = y_batch[:min_batch]
                    print(f"Adjusted to min batch size: {min_batch}")
                
                # Use mixed precision for evaluation
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with amp.autocast(device_type):
                    loss, pid, _ = model(x1_batch, x2_batch, y_batch)
                    loss = -loss  # Negative since we're maximizing alignment
                
                test_loss += loss.item()
                all_pids.append(pid.cpu())
            except Exception as e:
                print(f"ERROR in eval batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next batch
                continue
                
            # Clear memory
            del x1_batch, x2_batch, y_batch

    if all_pids:
        avg_test_loss = test_loss / len(test_loader)
        all_pids = torch.stack(all_pids).mean(dim=0)
        
        print(f"\nCE Alignment eval - Loss: {avg_test_loss:.4f}")
        print(f"Final PID components: Redundancy={all_pids[0]:.4f}, Unique1={all_pids[1]:.4f}, Unique2={all_pids[2]:.4f}, Synergy={all_pids[3]:.4f}")
    else:
        print("\nWARNING: No valid evaluation batches processed. Using default values.")
        all_pids = torch.zeros(4)  # Default to zeros if no valid batches
    
    # Final cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    return all_pids, None, (model, d1, d2, d12, p_y)


# ——————————————————————————————————————————————————————————————————————————————
# GW MODEL DATA GENERATION
#——————————————————————————————————————————————————————————————————————————————

def create_synthetic_labels(
    data: torch.Tensor,
    num_clusters: int = 10
) -> torch.Tensor:
    """
    Create synthetic labels for PID analysis using K-means clustering.
    
    This function normalizes data and applies K-means clustering to create
    discrete label categories for use in PID analysis. These synthetic labels
    serve as the target variable for measuring shared information.
    
    Args:
        data: Data tensor to cluster, shape [n_samples, feature_dim]
        num_clusters: Number of clusters to create
        
    Returns:
        Tensor of integer labels of shape [n_samples]
    """
    # Convert to numpy for sklearn clustering
    data_np = data.cpu().numpy()
    
    # Normalize data to improve clustering
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    normalized_data = (data_np - mean) / std
    
    # Import here to avoid dependency if not used
    from sklearn.cluster import KMeans
    
    # Perform K-means clustering with multiple initializations for stability
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(normalized_data)
    
    # Return as PyTorch tensor with shape [n_samples]
    return torch.tensor(labels, dtype=torch.long)

def prepare_pid_data(
    generated_data: Dict[str, torch.Tensor],
    domain_names: List[str],
    source_config: Dict[str, str],
    target_config: str,
    synthetic_labels: Optional[torch.Tensor] = None,
    num_clusters: int = 10,
) -> Tuple[MultimodalDataset, MultimodalDataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data for PID analysis.
    
    Args:
        generated_data: Dictionary of generated data
        domain_names: List of domain names
        source_config: Dictionary mapping domain names to source representations
        target_config: Target representation to use for clustering
        synthetic_labels: Optional synthetic labels
        num_clusters: Number of clusters for synthetic labels
        
    Returns:
        Tuple of (train_dataset, test_dataset, domain1_features, domain2_features, labels)
    """
    print(f"Preparing data for PID analysis with {num_clusters} clusters")
    available_keys = list(generated_data.keys())
    print(f"Available keys in generated_data: {available_keys}")
    
    # If domain_names is empty, try to extract from generated_data keys
    if len(domain_names) < 2:
        print("Warning: Insufficient domain names provided. Attempting to extract from generated data.")
        # Find all possible domain prefixes from the keys
        extracted_domains = []
        for key in available_keys:
            if key == 'gw_rep':  # Skip the special gw_rep key
                continue
                
            # Extract potential domain name by removing _latent, _decoded, etc.
            for suffix in ['_latent', '_decoded', '_gw_encoded']:
                if key.endswith(suffix):
                    domain = key[:-len(suffix)]
                    if domain not in extracted_domains:
                        extracted_domains.append(domain)
        
        print(f"Extracted domain prefixes: {extracted_domains}")
        
        if len(extracted_domains) >= 2:
            domain_names = extracted_domains[:2]  # Use first two domains
        
        print(f"Using domain names: {domain_names}")
        
        if len(domain_names) < 2:
            raise ValueError(f"Need at least 2 domains, got {len(domain_names)} even after extraction")
    
    # For simplicity, use first two domains
    domain_a, domain_b = domain_names[:2]
    
    # Determine source keys - check if source_config contains the domain name as key
    source_key_a = source_config.get(domain_a, None)
    source_key_b = source_config.get(domain_b, None)
    
    # If source_config doesn't have the domain name as key, check alternative formats
    if source_key_a is None:
        # Try with "_latent" suffix
        source_key_a = f"{domain_a}_latent"
        if source_key_a not in generated_data:
            # Try with "_decoded" suffix
            source_key_a = f"{domain_a}_decoded"
    
    if source_key_b is None:
        # Try with "_latent" suffix
        source_key_b = f"{domain_b}_latent"
        if source_key_b not in generated_data:
            # Try with "_decoded" suffix
            source_key_b = f"{domain_b}_decoded"
    
    # Validate that the keys exist
    if source_key_a not in generated_data:
        raise KeyError(f"Source key '{source_key_a}' for domain '{domain_a}' not found in generated data. Available keys: {available_keys}")
    if source_key_b not in generated_data:
        raise KeyError(f"Source key '{source_key_b}' for domain '{domain_b}' not found in generated data. Available keys: {available_keys}")
    if target_config not in generated_data:
        raise KeyError(f"Target key '{target_config}' not found in generated data. Available keys: {available_keys}")
    
    # Get data for each domain
    x1 = generated_data[source_key_a]
    x2 = generated_data[source_key_b]
    
    # Either use provided labels or create synthetic ones
    if synthetic_labels is not None:
        # Use provided labels
        labels = synthetic_labels
    else:
        # Create synthetic labels using the target data
        target_data = generated_data[target_config]
        labels = create_synthetic_labels(target_data, num_clusters)
    
    # Ensure correct device
    x1 = x1.to(device)
    x2 = x2.to(device)
    labels = labels.to(device)
    
    # Create train/test splits
    n_samples = x1.size(0)
    n_train = int(0.8 * n_samples)
    
    # Use first 80% for training, rest for testing (no shuffle for reproducibility)
    train_indices = torch.arange(n_train)
    test_indices = torch.arange(n_train, n_samples)
    
    # Create training dataset
    train_ds = MultimodalDataset(
        data=[x1[train_indices], x2[train_indices]],
        labels=labels[train_indices]
    )
    
    # Create test dataset
    test_ds = MultimodalDataset(
        data=[x1[test_indices], x2[test_indices]],
        labels=labels[test_indices]
    )
    
    # Return data
    return train_ds, test_ds, x1, x2, labels

def process_domain_data(domain_data, domain_name):
    """
    Helper function to extract the actual tensor data from the domain data structure.
    Handles different domain data formats from SimpleShapesDataModule.
    
    Args:
        domain_data: The data returned for a domain
        domain_name: The name of the domain
    
    Returns:
        The tensor data for the domain
    """
    import torch
    
    # For dictionary batch format
    if isinstance(domain_data, dict):
        # Direct key lookup for domain name
        if domain_name in domain_data:
            data = domain_data[domain_name]
        else:
            # Try to find domain in frozenset keys
            for k, v in domain_data.items():
                if isinstance(k, frozenset) and domain_name in k:
                    data = v
                    break
            else:
                data = domain_data  # Fallback to the original data
    else:
        data = domain_data
    
    # Handle Text type objects with bert attribute
    if hasattr(data, 'bert'):
        return data.bert
    # Handle dict with domain_name key
    elif isinstance(data, dict) and domain_name in data:
        value = data[domain_name]
        if hasattr(value, 'bert'):
            return value.bert
        elif isinstance(value, torch.Tensor):
            return value
        else:
            raise ValueError(f"Unsupported nested value type for {domain_name}: {type(value)}")
    # Handle Attribute type objects
    elif hasattr(data, 'category'):
        # Create a tensor from attribute fields
        attrs = [
            data.category.float() if not isinstance(data.category, torch.Tensor) else data.category.float(),
            data.x.float() if not isinstance(data.x, torch.Tensor) else data.x.float(),
            data.y.float() if not isinstance(data.y, torch.Tensor) else data.y.float(),
            data.size.float() if not isinstance(data.size, torch.Tensor) else data.size.float(),
            data.rotation.float() if not isinstance(data.rotation, torch.Tensor) else data.rotation.float(),
            data.color_r.float() if not isinstance(data.color_r, torch.Tensor) else data.color_r.float(),
            data.color_g.float() if not isinstance(data.color_g, torch.Tensor) else data.color_g.float(),
            data.color_b.float() if not isinstance(data.color_b, torch.Tensor) else data.color_b.float()
        ]
        return torch.stack(attrs, dim=-1)
    # Handle tensor data directly
    elif isinstance(data, torch.Tensor):
        # Special handling for v_latents with extra dimensions
        if domain_name == 'v_latents' and data.dim() > 2:
            return data[:, 0, :]  # Take first element along dimension 1
        return data
    # Handle list of data
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"Empty list for {domain_name}")
            
        # Iterate through the list to extract the appropriate data
        processed_items = []
        for item in data:
            # Handle dictionary items (common in dataset batches)
            if isinstance(item, dict):
                if domain_name in item:
                    value = item[domain_name]
                    if hasattr(value, 'bert'):
                        processed_items.append(value.bert)
                    elif isinstance(value, torch.Tensor):
                        processed_items.append(value)
                    else:
                        raise ValueError(f"Unsupported nested value type in list for {domain_name}: {type(value)}")
                else:
                    # Look for values with domain-specific attributes
                    for k, v in item.items():
                        if hasattr(v, 'bert'):
                            processed_items.append(v.bert)
                            break
                        elif isinstance(v, torch.Tensor):
                            processed_items.append(v)
                            break
            # Handle Text objects directly
            elif hasattr(item, 'bert'):
                processed_items.append(item.bert)
            # Handle tensors directly
            elif isinstance(item, torch.Tensor):
                processed_items.append(item)
            # Handle other attribute objects
            elif hasattr(item, 'category'):
                attrs = torch.tensor([
                    item.category.float() if not isinstance(item.category, torch.Tensor) else item.category.float(),
                    item.x.float() if not isinstance(item.x, torch.Tensor) else item.x.float(),
                    item.y.float() if not isinstance(item.y, torch.Tensor) else item.y.float(),
                    item.size.float() if not isinstance(item.size, torch.Tensor) else item.size.float(),
                    item.rotation.float() if not isinstance(item.rotation, torch.Tensor) else item.rotation.float(),
                    item.color_r.float() if not isinstance(item.color_r, torch.Tensor) else item.color_r.float(),
                    item.color_g.float() if not isinstance(item.color_g, torch.Tensor) else item.color_g.float(),
                    item.color_b.float() if not isinstance(item.color_b, torch.Tensor) else item.color_b.float()
                ])
                processed_items.append(attrs)
                
        if not processed_items:
            raise ValueError(f"Could not extract any valid tensors for {domain_name}")
            
        # Stack the processed items
        try:
            return torch.stack(processed_items)
        except Exception as e:
            print(f"Error stacking tensors for {domain_name}: {e}")
            print(f"First processed item type: {type(processed_items[0])}")
            print(f"First processed item shape: {processed_items[0].shape if hasattr(processed_items[0], 'shape') else 'unknown'}")
            raise ValueError(f"Could not stack tensors for {domain_name}: {e}")
    
    # For other cases, we need more handling based on the domain modules
    raise ValueError(f"Unsupported domain data type for {domain_name}: {type(data)}")

def prepare_dataset_pid_data(
    generated_data: Dict[str, torch.Tensor], 
    domain_names: List[str],
    analysis_domain: str,
    synthetic_labels=None, 
    num_clusters: int = 10,
    use_domain_for_labels: str = "both"
):
    """
    Prepare data for PID analysis using dataset-generated samples.
    This function allows analysis between original and decoded domain representations.
    
    Args:
        generated_data: Data dictionary from generate_samples_from_dataset
        domain_names: List of domain names
        analysis_domain: Which domain to analyze ('both' or one of domain_names)
        synthetic_labels: Optional pre-generated labels
        num_clusters: Number of clusters if synthetic_labels is None
        use_domain_for_labels: Parameter kept for backwards compatibility, but ignored. 
                              The function will only cluster on the target (decoded) representation.
        
    Returns:
        Multiple datasets ready for PID analysis
    """
    if len(domain_names) != 2:
        raise ValueError(f"This function requires exactly 2 domains, got {len(domain_names)}")
    
    domain_a, domain_b = domain_names
    
    # Define pairs to analyze
    if analysis_domain == "both" or analysis_domain not in domain_names:
        analysis_pairs = [
            # (first domain data, second domain data, pair name)
            (f"{domain_a}_orig", f"{domain_a}_decoded", f"{domain_a}_orig_vs_decoded"),
            (f"{domain_b}_orig", f"{domain_b}_decoded", f"{domain_b}_orig_vs_decoded")
        ]
    else:
        # Only analyze the specified domain
        analysis_pairs = [
            (f"{analysis_domain}_orig", f"{analysis_domain}_decoded", f"{analysis_domain}_orig_vs_decoded")
        ]
    
    results = {}
    
    for source_key, target_key, pair_name in analysis_pairs:
        if source_key not in generated_data or target_key not in generated_data:
            print(f"Skipping {pair_name} - missing data")
            continue
            
        # Get domain data
        x1 = generated_data[source_key]
        x2 = generated_data[target_key]
        
        # Create or use labels
        if synthetic_labels is None:
            # Always use the target (decoded) data for clustering, not the source
            print(f"Creating synthetic labels using {target_key} representation (target)")
            labels = create_synthetic_labels(x2, num_clusters)
        else:
            labels = synthetic_labels
            
        # Create train/test split
        n_samples = x1.size(0)
        split_idx = int(0.8 * n_samples)
        
        # Shuffle indices
        indices = torch.randperm(n_samples)
        
        # Split data
        train_x1 = x1[indices[:split_idx]]
        train_x2 = x2[indices[:split_idx]]
        train_labels = labels[indices[:split_idx]]
        
        test_x1 = x1[indices[split_idx:]]
        test_x2 = x2[indices[split_idx:]]
        test_labels = labels[indices[split_idx:]]
        
        # Create datasets
        train_ds = MultimodalDataset([train_x1, train_x2], train_labels)
        test_ds = MultimodalDataset([test_x1, test_x2], test_labels)
        
        results[pair_name] = (train_ds, test_ds, x1, x2, labels)
    
    return results

# ——————————————————————————————————————————————————————————————————————————————
# MODEL ANALYSIS FUNCTIONS
# ——————————————————————————————————————————————————————————————————————————————
def analyze_model(
    model_path: str,
    domain_modules: Dict[str, DomainModule],
    output_dir: str,
    source_config: Dict[str, str],
    target_config: str,
    n_samples: int = 10000,
    batch_size: int = 128,
    num_clusters: int = 10,
    discrim_epochs: int = 40,
    ce_epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    discrim_hidden_dim: int = 64,
    discrim_layers: int = 5,
    use_wandb: bool = True,
    wandb_project: str = "pid-analysis",
    wandb_entity: Optional[str] = None,
    data_module=None,
    dataset_split: str = "test",
    use_gw_encoded: bool = False,
    use_compile: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a trained model using PID analysis.
    
    Args:
        model_path: Path to the model checkpoint
        domain_modules: Dictionary of domain modules
        output_dir: Directory to save results
        source_config: Dictionary mapping domain names to their source representations
        target_config: Target representation to use for clustering
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        num_clusters: Number of clusters for synthetic labels
        discrim_epochs: Number of epochs to train discriminators
        ce_epochs: Number of epochs to train CE alignment
        device: Device to use for computation
        discrim_hidden_dim: Hidden dimension for discriminator networks
        discrim_layers: Number of layers in discriminator networks
        use_wandb: Whether to log results to Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        data_module: Optional data module for real data
        dataset_split: Dataset split to use
        use_gw_encoded: Whether to use GW-encoded vectors
        use_compile: Whether to use torch.compile for model optimization
        
    Returns:
        Dictionary containing analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize W&B if requested
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "model_path": model_path,
                "n_samples": n_samples,
                "num_clusters": num_clusters,
                "discrim_epochs": discrim_epochs,
                "ce_epochs": ce_epochs,
                "source_config": source_config,
                "target_config": target_config,
                "use_compile": use_compile
            }
        )
    
    # Load model
    print(f"Loading model from {model_path}")
    model = load_checkpoint(
        checkpoint_path=model_path,
        domain_modules=domain_modules,
        device=device
    )
    
    # Get domain names from model
    domain_names = []
    if hasattr(model, 'domain_mods') and model.domain_mods:
        domain_names = list(model.domain_mods.keys())
        print(f"Domain names from model: {domain_names}")
    else:
        print("Warning: No domain names found in model")
    
    # Generate data for PID analysis
    src_type = "GW-encoded" if use_gw_encoded else "latent"
    print(f"Generating {n_samples} samples {'from ' + dataset_split + ' dataset' if data_module else ''} (using {src_type} vectors as sources)")
    generated_data = generate_samples_from_model(
        model=model,
        domain_names=domain_names,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        use_gw_encoded=use_gw_encoded,
        data_module=data_module,
        dataset_split=dataset_split
    )
    
    # Prepare data for PID analysis
    print(f"Preparing data for PID analysis with {num_clusters} clusters")
    train_ds, test_ds, x1, x2, labels = prepare_pid_data(
        generated_data=generated_data,
        domain_names=domain_names,
        source_config=source_config,
        target_config=target_config,
        num_clusters=num_clusters
    )
    
    # Train the first discriminator for x1
    print("\nTraining discriminator for x1...")
    # Create the discriminator and move it to device
    d1 = Discrim(
        x_dim=x1.size(1),
        hidden_dim=discrim_hidden_dim,
        num_labels=num_clusters,
        layers=discrim_layers,
        activation="relu"
    ).to(device)
    # Create optimizer for first discriminator
    opt1 = torch.optim.Adam(d1.parameters(), lr=1e-3)
    discrim_1 = train_discrim(
        model=d1,
        loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        optimizer=opt1,
        data_type=([0], [2]),  # batch[0] for x1, batch[2] for label
        num_epoch=discrim_epochs,
        wandb_prefix="discrim_1" if use_wandb else None,
        use_compile=use_compile
    )
    
    # Train the second discriminator for x2
    print("\nTraining discriminator for x2...")
    # Create the discriminator and move it to device
    d2 = Discrim(
        x_dim=x2.size(1),
        hidden_dim=discrim_hidden_dim,
        num_labels=num_clusters,
        layers=discrim_layers,
        activation="relu"
    ).to(device)
    # Create optimizer for second discriminator
    opt2 = torch.optim.Adam(d2.parameters(), lr=1e-3)
    discrim_2 = train_discrim(
        model=d2,
        loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        optimizer=opt2,
        data_type=([1], [2]),  # batch[1] for x2, batch[2] for label
        num_epoch=discrim_epochs,
        wandb_prefix="discrim_2" if use_wandb else None,
        use_compile=use_compile
    )
    
    # Train the joint discriminator for x1,x2
    print("\nTraining joint discriminator...")
    # Create the joint discriminator and move it to device
    d12 = Discrim(
        x_dim=x1.size(1) + x2.size(1),  # Combined input dimensions
        hidden_dim=discrim_hidden_dim,
        num_labels=num_clusters,
        layers=discrim_layers,
        activation="relu"
    ).to(device)
    # Create optimizer for joint discriminator
    opt12 = torch.optim.Adam(d12.parameters(), lr=1e-3)
    discrim_12 = train_discrim(
        model=d12,
        loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        optimizer=opt12,
        data_type=([0, 1], [2]),  # batch[0] and batch[1] for x1,x2, batch[2] for label
        num_epoch=discrim_epochs,
        wandb_prefix="discrim_12" if use_wandb else None,
        use_compile=use_compile
    )
    
    # Calculate p_y from labels
    print("\nCalculating p_y distribution...")
    with torch.no_grad():
        # Ensure labels are 1D for one-hot encoding
        labels_1d = labels.view(-1)
        print(f"Labels for p_y: shape={labels_1d.shape}, unique values={torch.unique(labels_1d).tolist()}")
        one_hot = F.one_hot(labels_1d, num_clusters).float()
        p_y = one_hot.sum(dim=0) / labels_1d.size(0)
        print(f"p_y shape: {p_y.shape}")
    
    # Create CE alignment model
    print("\nCreating CE alignment model...")
    ce_model = CEAlignmentInformation(
        x1_dim=x1.size(1),
        x2_dim=x2.size(1),
        hidden_dim=discrim_hidden_dim,
        embed_dim=discrim_hidden_dim,
        num_labels=num_clusters,
        layers=discrim_layers,
        activation="relu",
        discrim_1=discrim_1,
        discrim_2=discrim_2,
        discrim_12=discrim_12,
        p_y=p_y
    ).to(device)
    
    # Create optimizer instance
    ce_optimizer = torch.optim.Adam(ce_model.parameters(), lr=1e-3)
    
    # Train CE alignment model
    print("\nTraining CE alignment...")
    train_ce_alignment(
        model=ce_model,
        loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        optimizer_class=torch.optim.Adam,  # This will be instantiated inside the function
        num_epoch=ce_epochs,
        wandb_prefix="ce" if use_wandb else None,
        use_compile=use_compile
    )
    
    # Evaluate models
    print("\nEvaluating models...")
    discrim_1_results = eval_discrim(
        model=discrim_1,
        loader=DataLoader(test_ds, batch_size=batch_size),
        data_type=([0], [2]),  # batch[0] for x1, batch[2] for label
        wandb_prefix="discrim_1" if use_wandb else None
    )
    
    discrim_2_results = eval_discrim(
        model=discrim_2,
        loader=DataLoader(test_ds, batch_size=batch_size),
        data_type=([1], [2]),  # batch[1] for x2, batch[2] for label
        wandb_prefix="discrim_2" if use_wandb else None
    )
    
    ce_alignment_results = eval_ce_alignment(
        model=ce_model,  # Use the trained model
        loader=DataLoader(test_ds, batch_size=batch_size),
        wandb_prefix="ce" if use_wandb else None
    )
    
    # Save results
    serializable_results = {
        "discrim_1": prepare_for_json(discrim_1_results),
        "discrim_2": prepare_for_json(discrim_2_results),
        "ce_alignment": prepare_for_json(ce_alignment_results),
        "source_config": source_config,
        "target_config": target_config,
        "num_clusters": num_clusters,
        "model_path": model_path,
    }
    
    # Save to file
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Close W&B run
    if wandb_run:
        wandb_run.finish()
    
    return serializable_results

def analyze_multiple_models(
    checkpoint_dir: str,
    domain_modules: Dict[str, DomainModule],
    output_dir: str,
    source_config: Dict[str, str],
    target_config: str,
    n_samples: int = 10000,
    batch_size: int = 128,
    num_clusters: int = 10,
    discrim_epochs: int = 40,
    ce_epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    discrim_hidden_dim: int = 64,
    discrim_layers: int = 5,
    use_wandb: bool = True,
    wandb_project: str = "pid-analysis",
    wandb_entity: Optional[str] = None,
    data_module=None,
    dataset_split: str = "test",
    use_gw_encoded: bool = False,
    use_compile: bool = True,
) -> List[Dict[str, Any]]:
    """
    Analyze multiple models using PID analysis.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        domain_modules: Dictionary of domain modules
        output_dir: Directory to save results
        source_config: Dictionary mapping domain names to their source representations
        target_config: Target representation to use for clustering
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        num_clusters: Number of clusters for synthetic labels
        discrim_epochs: Number of epochs to train discriminators
        ce_epochs: Number of epochs to train CE alignment
        device: Device to use for computation
        discrim_hidden_dim: Hidden dimension for discriminator networks
        discrim_layers: Number of layers in discriminator networks
        use_wandb: Whether to log results to Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        data_module: Optional data module for real data
        dataset_split: Dataset split to use
        use_gw_encoded: Whether to use GW-encoded vectors
        use_compile: Whether to use torch.compile for model optimization
        
    Returns:
        List of dictionaries containing analysis results for each model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize W&B if requested
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "checkpoint_dir": checkpoint_dir,
                "n_samples": n_samples,
                "num_clusters": num_clusters,
                "discrim_epochs": discrim_epochs,
                "ce_epochs": ce_epochs,
                "source_config": source_config,
                "target_config": target_config,
                "use_compile": use_compile,
            }
        )
    
    # Find model checkpoints
    checkpoint_paths = find_latest_model_checkpoints(checkpoint_dir)
    if not checkpoint_paths:
        raise ValueError(f"No model checkpoints found in {checkpoint_dir}")
    
    # Analyze each model
    results = []
    for checkpoint_path in checkpoint_paths:
        print(f"\nAnalyzing model: {checkpoint_path}")
        
        # Create model-specific output directory
        model_name = Path(checkpoint_path).stem
        model_output_dir = os.path.join(output_dir, model_name)
        
        # Analyze model
        model_results = analyze_model(
            model_path=checkpoint_path,
            domain_modules=domain_modules,
            output_dir=model_output_dir,
            source_config=source_config,
            target_config=target_config,
            n_samples=n_samples,
            batch_size=batch_size,
            num_clusters=num_clusters,
            discrim_epochs=discrim_epochs,
            ce_epochs=ce_epochs,
            device=device,
            discrim_hidden_dim=discrim_hidden_dim,
            discrim_layers=discrim_layers,
            use_wandb=use_wandb,
            data_module=data_module,
            dataset_split=dataset_split,
            use_gw_encoded=use_gw_encoded,
            use_compile=use_compile,
        )
        
        results.append(model_results)
    
    # Close W&B run
    if wandb_run:
        wandb_run.finish()
    
    return results

def plot_pid_components(results: List[Dict], output_dir: str, wandb_run=None):
    """
    Plot PID components from multiple models.
    
    Args:
        results: List of PID results
        output_dir: Directory to save plots
        wandb_run: Optional wandb run to log plots to
    """
    if not results:
        return
    
    # Extract fusion weights and PID values
    weights_a = []
    weights_b = []
    redundancy = []
    unique_a = []
    unique_b = []
    synergy = []
    model_names = []
    domains = []
    
    for result in results:
        fusion_weights = result.get("fusion_weights", {})
        pid_values = result.get("pid_values", {})
        result_domains = result.get("domains", [])
        
        if len(result_domains) != 2 or len(fusion_weights) != 2:
            continue
        
        domain_a, domain_b = result_domains
        if not domains:
            domains = [domain_a, domain_b]
            
        weights_a.append(fusion_weights.get(domain_a, 0))
        weights_b.append(fusion_weights.get(domain_b, 0))
        
        redundancy.append(pid_values.get("redundancy", 0))
        unique_a.append(pid_values.get("unique1", 0))
        unique_b.append(pid_values.get("unique2", 0))
        synergy.append(pid_values.get("synergy", 0))
        model_names.append(result.get("model_name", "unknown"))
    
    if not weights_a:
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a scatter plot of redundancy vs. weight ratio
    weight_ratios = [a/b if b > 0 else float('inf') for a, b in zip(weights_a, weights_b)]
    valid_indices = [i for i, x in enumerate(weight_ratios) if x != float('inf')]
    
    if valid_indices:
        # Filter out infinite values
        filtered_ratios = [weight_ratios[i] for i in valid_indices]
        filtered_redundancy = [redundancy[i] for i in valid_indices]
        filtered_unique_a = [unique_a[i] for i in valid_indices]
        filtered_unique_b = [unique_b[i] for i in valid_indices]
        filtered_synergy = [synergy[i] for i in valid_indices]
        filtered_model_names = [model_names[i] for i in valid_indices]
        
        # Set consistent style for all plots
        plt.style.use('seaborn-v0_8-whitegrid')  # Updated to newer matplotlib style name
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Changed from 'Arial' to 'DejaVu Sans'
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        # 1. Create figure for PID components vs weight ratio
        plt.figure(figsize=(12, 8))
        plt.scatter(filtered_ratios, filtered_redundancy, label="Redundancy", marker="o", s=80, alpha=0.7)
        plt.scatter(filtered_ratios, filtered_unique_a, label=f"Unique to {domains[0]}", marker="^", s=80, alpha=0.7)
        plt.scatter(filtered_ratios, filtered_unique_b, label=f"Unique to {domains[1]}", marker="s", s=80, alpha=0.7)
        plt.scatter(filtered_ratios, filtered_synergy, label="Synergy", marker="*", s=100, alpha=0.7)
        
        plt.xlabel("Weight Ratio (Domain A / Domain B)")
        plt.ylabel("Information (bits)")
        plt.title("PID Components vs. Weight Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add annotations for outlier points
        for i, (ratio, red, uniq_a, uniq_b, syn, name) in enumerate(
            zip(filtered_ratios, filtered_redundancy, filtered_unique_a, 
                filtered_unique_b, filtered_synergy, filtered_model_names)):
            # Only annotate outliers or significant points
            if abs(red) > 0.3 or abs(uniq_a) > 0.3 or abs(uniq_b) > 0.3 or abs(syn) > 0.3:
                plt.annotate(name, (ratio, max(red, uniq_a, uniq_b, syn)), 
                            textcoords="offset points", xytext=(0,10), ha='center')
        
        # Save figure
        ratio_plot_path = os.path.join(plots_dir, "pid_vs_weight_ratio.png")
        plt.savefig(ratio_plot_path, dpi=300, bbox_inches="tight")
        
        # Log to wandb if available
        if HAS_WANDB and wandb_run is not None:
            wandb.log({"plots/pid_vs_weight_ratio": wandb.Image(ratio_plot_path)})
        
        plt.close()
        
        # 2. Create a stacked bar chart
        plt.figure(figsize=(14, 10))
        
        # Sort by weight ratio
        sorted_indices = sorted(range(len(filtered_ratios)), key=lambda i: filtered_ratios[i])
        sorted_ratios = [filtered_ratios[i] for i in sorted_indices]
        sorted_redundancy = [filtered_redundancy[i] for i in sorted_indices]
        sorted_unique_a = [filtered_unique_a[i] for i in sorted_indices]
        sorted_unique_b = [filtered_unique_b[i] for i in sorted_indices]
        sorted_synergy = [filtered_synergy[i] for i in sorted_indices]
        sorted_model_names = [filtered_model_names[i] for i in sorted_indices]
        
        # Create x-axis labels with weight ratios
        x_labels = [f"{ratio:.2f}" for ratio in sorted_ratios]
        x = range(len(sorted_ratios))
        
        # Plot stacked bars
        plt.bar(x, sorted_redundancy, label="Redundancy", color="blue", alpha=0.7)
        plt.bar(x, sorted_unique_a, bottom=sorted_redundancy, label=f"Unique to {domains[0]}", color="green", alpha=0.7)
        bottom = [r + ua for r, ua in zip(sorted_redundancy, sorted_unique_a)]
        plt.bar(x, sorted_unique_b, bottom=bottom, label=f"Unique to {domains[1]}", color="orange", alpha=0.7)
        bottom = [b + ub for b, ub in zip(bottom, sorted_unique_b)]
        plt.bar(x, sorted_synergy, bottom=bottom, label="Synergy", color="red", alpha=0.7)
        
        plt.xlabel("Weight Ratio (Domain A / Domain B)")
        plt.ylabel("Information (bits)")
        plt.title("PID Decomposition by Weight Ratio")
        plt.xticks(x, x_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add model names as annotations
        if len(sorted_model_names) <= 20:  # Only add annotations if not too crowded
            for i, model_name in enumerate(sorted_model_names):
                plt.annotate(model_name.split('_')[-1], (i, 0), rotation=90, 
                           xytext=(0, -20), textcoords="offset points", ha='center', fontsize=8)
        
        # Save figure
        stacked_plot_path = os.path.join(plots_dir, "pid_stacked_bars.png")
        plt.savefig(stacked_plot_path, dpi=300, bbox_inches="tight")
        
        # Log to wandb if available
        if HAS_WANDB and wandb_run is not None:
            wandb.log({"plots/pid_stacked_bars": wandb.Image(stacked_plot_path)})
        
        plt.close()
        
        # 3. Create correlation plot of weight ratio vs. PID measures
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation coefficients
        import scipy.stats as stats
        corr_redundancy = stats.pearsonr(filtered_ratios, filtered_redundancy)[0]
        corr_unique_a = stats.pearsonr(filtered_ratios, filtered_unique_a)[0]
        corr_unique_b = stats.pearsonr(filtered_ratios, filtered_unique_b)[0]
        corr_synergy = stats.pearsonr(filtered_ratios, filtered_synergy)[0]
        
        # Sort ratios for trend lines
        sorted_ratio_indices = sorted(range(len(filtered_ratios)), key=lambda i: filtered_ratios[i])
        x_sorted = [filtered_ratios[i] for i in sorted_ratio_indices]
        
        # Redundancy trend
        z = np.polyfit(filtered_ratios, filtered_redundancy, 1)
        p = np.poly1d(z)
        plt.plot(x_sorted, p(x_sorted), "b--", alpha=0.5)
        plt.scatter(filtered_ratios, filtered_redundancy, label=f"Redundancy (r={corr_redundancy:.2f})", 
                   c='blue', alpha=0.7, s=60)
        
        # Unique to domain A trend
        z = np.polyfit(filtered_ratios, filtered_unique_a, 1)
        p = np.poly1d(z)
        plt.plot(x_sorted, p(x_sorted), "g--", alpha=0.5)
        plt.scatter(filtered_ratios, filtered_unique_a, label=f"Unique to {domains[0]} (r={corr_unique_a:.2f})", 
                   c='green', alpha=0.7, s=60)
        
        # Unique to domain B trend
        z = np.polyfit(filtered_ratios, filtered_unique_b, 1)
        p = np.poly1d(z)
        plt.plot(x_sorted, p(x_sorted), "orange", linestyle='--', alpha=0.5)
        plt.scatter(filtered_ratios, filtered_unique_b, label=f"Unique to {domains[1]} (r={corr_unique_b:.2f})", 
                   c='orange', alpha=0.7, s=60)
        
        # Synergy trend
        z = np.polyfit(filtered_ratios, filtered_synergy, 1)
        p = np.poly1d(z)
        plt.plot(x_sorted, p(x_sorted), "r--", alpha=0.5)
        plt.scatter(filtered_ratios, filtered_synergy, label=f"Synergy (r={corr_synergy:.2f})", 
                   c='red', alpha=0.7, s=60)
        
        plt.xlabel("Weight Ratio (Domain A / Domain B)")
        plt.ylabel("Information (bits)")
        plt.title("PID Components vs. Weight Ratio with Correlation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Save figure
        corr_plot_path = os.path.join(plots_dir, "pid_correlation.png")
        plt.savefig(corr_plot_path, dpi=300, bbox_inches="tight")
        
        # Log to wandb if available
        if HAS_WANDB and wandb_run is not None:
            wandb.log({"plots/pid_correlation": wandb.Image(corr_plot_path)})
        
        plt.close()
        
        # 4. Total information vs. weight ratio plot
        plt.figure(figsize=(10, 6))
        
        total_info = [r + ua + ub + s for r, ua, ub, s in zip(
            filtered_redundancy, filtered_unique_a, filtered_unique_b, filtered_synergy)]
        
        # Calculate correlation coefficient for total information
        corr_total = stats.pearsonr(filtered_ratios, total_info)[0]
        
        # Plot total information vs. weight ratio
        plt.scatter(filtered_ratios, total_info, c=filtered_ratios, cmap='viridis', 
                   s=80, alpha=0.8, label=f"Total Information (r={corr_total:.2f})")
        
        # Add trend line
        z = np.polyfit(filtered_ratios, total_info, 1)
        p = np.poly1d(z)
        plt.plot(x_sorted, p(x_sorted), "k--", alpha=0.5)
        
        plt.xlabel("Weight Ratio (Domain A / Domain B)")
        plt.ylabel("Total Information (bits)")
        plt.title("Total Information vs. Weight Ratio")
        plt.colorbar(label="Weight Ratio")
        plt.grid(True, alpha=0.3)
        
        # Add annotations for outlier points
        for i, (ratio, tot, name) in enumerate(zip(filtered_ratios, total_info, filtered_model_names)):
            if tot > np.mean(total_info) + np.std(total_info):
                plt.annotate(name, (ratio, tot), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        # Save figure
        total_info_path = os.path.join(plots_dir, "total_information.png")
        plt.savefig(total_info_path, dpi=300, bbox_inches="tight")
        
        # Log to wandb if available
        if HAS_WANDB and wandb_run is not None:
            wandb.log({"plots/total_information": wandb.Image(total_info_path)})
        
        plt.close()
        
        print(f"Plots saved to {plots_dir}")

def generate_samples_from_dataset(
    model: GWModuleConfigurableFusion,
    data_module,
    domain_names: List[str],
    split: str = "test",
    n_samples: int = 1000,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Generate samples from a GW model using real dataset samples for PID analysis.
    
    Args:
        model: The trained GW model
        data_module: Data module with get_samples method
        domain_names: List of domain names to generate samples for (should be length 2)
        split: Dataset split to use ("train", "val", or "test")
        n_samples: Maximum number of samples to generate
        batch_size: Batch size for generation
        device: Device to run the model on
        
    Returns:
        Dictionary with domain samples and GW workspace representations
    """
    # Use the fixed version
    return generate_samples_from_dataset_fixed(
        model=model,
        data_module=data_module,
        domain_names=domain_names,
        split=split,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device
    )

def load_checkpoint(
    checkpoint_path: str,
    domain_modules: Optional[Mapping[str, DomainModule]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> GWModuleConfigurableFusion:
    """
    Load a GWModuleConfigurableFusion from a checkpoint.
    
    This function is designed to handle different checkpoint formats.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        domain_modules: Dict of domain modules to use with the loaded model
        device: Device to load the model on
    
    Returns:
        Loaded GWModuleConfigurableFusion instance
    """
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model info from checkpoint
    if "model_state_dict" in checkpoint:
        # Standard format from minimal_script_with_validation.py
        state_dict = checkpoint["model_state_dict"]
        hidden_dim = checkpoint.get("hidden_dim", 32)
        n_layers = checkpoint.get("n_layers", 3)
        workspace_dim = checkpoint.get("workspace_dim", 12)
        fusion_weights = checkpoint.get("fusion_weights", None)
    elif "state_dict" in checkpoint:
        # Lightning Module format
        state_dict = checkpoint["state_dict"]
        # Extract model parameters from hparams or defaults
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            hidden_dim = hparams.get("hidden_dim", 32)
            n_layers = hparams.get("n_layers", 3)
            workspace_dim = hparams.get("workspace_dim", 12)
            fusion_weights = hparams.get("fusion_weights", None)
        else:
            # Use defaults
            hidden_dim = 32
            n_layers = 3
            workspace_dim = 12
            fusion_weights = None
    else:
        # Direct state dict format
        state_dict = checkpoint
        # Use defaults
        hidden_dim = 32
        n_layers = 3
        workspace_dim = 12
        fusion_weights = None
    
    # Create encoders and decoders for each domain
    gw_encoders = {}
    gw_decoders = {}
    
    # If domain_modules is None, we can't create the model
    if domain_modules is None:
        raise ValueError("domain_modules must be provided to load_checkpoint")
    
    # Create encoders and decoders
    for domain_name, domain_module in domain_modules.items():
        latent_dim = domain_module.latent_dim
        print(f"Domain '{domain_name}' latent dimension: {latent_dim}")
        
        # Create encoder and decoder
        from shimmer.modules.gw_module import GWEncoder, GWDecoder
        gw_encoders[domain_name] = GWEncoder(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=workspace_dim,
            n_layers=n_layers,
        )
        
        gw_decoders[domain_name] = GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=n_layers,
        )
    
    # If no fusion weights provided, use equal weights
    if fusion_weights is None:
        weight_value = 1.0 / len(domain_modules) if domain_modules else 0.0
        fusion_weights = {name: weight_value for name in domain_modules}
    
    # Create GW module
    gw_module = GWModuleConfigurableFusion(
        domain_modules=domain_modules,
        workspace_dim=workspace_dim,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        fusion_weights=fusion_weights,
    )
    
    # Store architecture parameters for easier checkpoint saving
    gw_module.hidden_dim = hidden_dim
    gw_module.n_layers = n_layers
    
    # Try to load state dict
    try:
        # Clean up state dict keys if they're from LightningModule
        if any(k.startswith('gw_module.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('gw_module.'):
                    new_key = k[len('gw_module.'):]
                    new_state_dict[new_key] = v
                elif not k.startswith('domain_mods.'):  # Skip domain module params
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # Attempt to load state dict
        missing_keys, unexpected_keys = gw_module.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
    except Exception as e:
        print(f"Warning: Error loading state dict: {e}")
        print("Proceeding with newly initialized model")
    
    # Move model to device
    gw_module = gw_module.to(device)
    
    return gw_module

def generate_samples_from_model(
    model: GWModuleConfigurableFusion,
    domain_names: List[str],
    n_samples: int = 10000,
    batch_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_gw_encoded: bool = False,
    data_module=None,
    dataset_split: str = "test"
) -> Dict[str, torch.Tensor]:
    """
    Generate samples for PID analysis.
    
    Args:
        model: The trained GW model
        domain_names: List of domain names to generate samples for
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to run the model on
        use_gw_encoded: If True, use GW-encoded vectors as sources instead of original latent vectors
        data_module: Optional data module for real data
        dataset_split: Dataset split to use if data_module is provided
        
    Returns:
        Dictionary with domain samples and GW workspace representations
    """
    # If data_module is provided, use real data
    if data_module is not None:
        print(f"Using real data from {dataset_split} split for sample generation")
        return generate_samples_from_dataset_fixed(
            model=model,
            data_module=data_module,
            domain_names=domain_names,
            split=dataset_split,
            n_samples=n_samples,
            batch_size=batch_size,
            device=device,
            use_gw_encoded=use_gw_encoded
        )
    
    # Otherwise, provide a warning about using synthetic data
    print(f"WARNING: No data_module provided. Using synthetic random latent vectors for sample generation.")
    print(f"This is NOT RECOMMENDED for accurate PID analysis. Please provide a data_module.")
    
    # If domain_names is empty, use default values for dummy implementation
    if len(domain_names) == 0:
        print("Warning: No domain names provided. Using default domain names 'v_latents' and 't'.")
        domain_names = ["v_latents", "t"]
    
    if len(domain_names) != 2:
        raise ValueError(f"This function requires exactly 2 domains, got {len(domain_names)}")
    
    print(f"Running generate_samples_from_model with domains: {domain_names}")
    model = model.to(device)
    model.eval()
    
    domain_a, domain_b = domain_names
    
    # Initialize containers for original and decoded representations
    domain_a_orig = []
    domain_b_orig = []
    domain_a_gw_encoded = []
    domain_b_gw_encoded = []
    domain_a_decoded = []
    domain_b_decoded = []
    gw_reps = []
    
    # Calculate number of batches needed
    num_batches = (n_samples + batch_size - 1) // batch_size
    total_samples = 0
    
    # Generate data in batches
    with torch.no_grad():
        for batch_idx in range(num_batches):
            if total_samples >= n_samples:
                break
                
            # Generate random latent vectors for each domain
            curr_batch_size = min(batch_size, n_samples - total_samples)
            
            try:
                # Get latent dimensions for each domain
                latent_dim_a = 64  # Default dimension if not found
                latent_dim_b = 64  # Default dimension if not found
                
                # Try to get actual dimensions if available
                if hasattr(model, 'domain_mods') and model.domain_mods:
                    if domain_a in model.domain_mods and hasattr(model.domain_mods[domain_a], 'latent_dim'):
                        latent_dim_a = model.domain_mods[domain_a].latent_dim
                    if domain_b in model.domain_mods and hasattr(model.domain_mods[domain_b], 'latent_dim'):
                        latent_dim_b = model.domain_mods[domain_b].latent_dim
                
                # Generate random latent vectors
                latent_a = torch.randn(curr_batch_size, latent_dim_a, device=device)
                latent_b = torch.randn(curr_batch_size, latent_dim_b, device=device)
                
                # If we're using a dummy model, generate dummy outputs
                if not hasattr(model, 'gw_encoders') or domain_a not in model.gw_encoders or domain_b not in model.gw_encoders:
                    print("Warning: Using dummy encoders/decoders for generation")
                    workspace_dim = getattr(model, 'workspace_dim', 12)  # Default workspace dim if not set
                    
                    # Create dummy GW states and decoded vectors
                    gw_state_a = torch.randn(curr_batch_size, workspace_dim, device=device)
                    gw_state_b = torch.randn(curr_batch_size, workspace_dim, device=device)
                    fused_gw = torch.randn(curr_batch_size, workspace_dim, device=device)
                    decoded_a = torch.randn(curr_batch_size, latent_dim_a, device=device)
                    decoded_b = torch.randn(curr_batch_size, latent_dim_b, device=device)
                else:
                    # Encode to GW representation
                    gw_state_a = model.gw_encoders[domain_a](latent_a)
                    gw_state_b = model.gw_encoders[domain_b](latent_b)
                    
                    # Handle v_latents with extra dimensions (mean and variance)
                    if domain_a == 'v_latents' and gw_state_a.dim() > 2:
                        # Take only the first component (mean vector)
                        gw_state_a = gw_state_a[:, 0, :]
                    
                    if domain_b == 'v_latents' and gw_state_b.dim() > 2:
                        # Take only the first component (mean vector)
                        gw_state_b = gw_state_b[:, 0, :]
                    
                    # Create weighted fusion using model's fusion weights
                    fused_gw = model.fuse({
                        domain_a: gw_state_a,
                        domain_b: gw_state_b
                    }, None)  # Selection scores are ignored in GWModuleConfigurableFusion
                    
                    # Decode fused representation to both domains
                    decoded_a = model.gw_decoders[domain_a](fused_gw)
                    decoded_b = model.gw_decoders[domain_b](fused_gw)
                
                # Store results - use either latents or GW-encoded vectors based on parameter
                if use_gw_encoded:
                    domain_a_orig.append(gw_state_a.cpu())
                    domain_b_orig.append(gw_state_b.cpu())
                else:
                    domain_a_orig.append(latent_a.cpu())
                    domain_b_orig.append(latent_b.cpu())
                
                # Also store GW-encoded vectors for later analysis
                domain_a_gw_encoded.append(gw_state_a.cpu())
                domain_b_gw_encoded.append(gw_state_b.cpu())
                
                # Store decoded representations and fused GW state
                domain_a_decoded.append(decoded_a.cpu())
                domain_b_decoded.append(decoded_b.cpu())
                gw_reps.append(fused_gw.cpu())
                
                total_samples += curr_batch_size
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if total_samples == 0:
        raise ValueError("Could not generate any samples")
    
    print(f"Generated {total_samples} samples")
    
    # Create appropriate source prefix based on what was stored
    source_prefix = "gw_encoded" if use_gw_encoded else "latent"
    
    # Concatenate batches
    try:
        result = {
            f"{domain_a}_{source_prefix}": torch.cat(domain_a_orig, dim=0)[:n_samples],
            f"{domain_b}_{source_prefix}": torch.cat(domain_b_orig, dim=0)[:n_samples],
            f"{domain_a}_gw_encoded": torch.cat(domain_a_gw_encoded, dim=0)[:n_samples],
            f"{domain_b}_gw_encoded": torch.cat(domain_b_gw_encoded, dim=0)[:n_samples],
            f"{domain_a}_decoded": torch.cat(domain_a_decoded, dim=0)[:n_samples],
            f"{domain_b}_decoded": torch.cat(domain_b_decoded, dim=0)[:n_samples],
            "gw_rep": torch.cat(gw_reps, dim=0)[:n_samples]
        }
        print(f"Final output keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"Error concatenating results: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_samples_from_dataset_fixed(
    model: GWModuleConfigurableFusion,
    data_module,
    domain_names: List[str],
    split: str = "test",
    n_samples: int = 1000,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_gw_encoded: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Generate samples from real dataset using the model.
    
    Args:
        model: The trained GW model
        data_module: Data module with train/val/test dataloaders
        domain_names: List of domain names to generate samples for (should be length 2)
        split: Dataset split to use ("train", "val", or "test")
        n_samples: Maximum number of samples to generate
        batch_size: Batch size for generation
        device: Device to run the model on
        use_gw_encoded: Whether to use GW-encoded vectors as sources
        
    Returns:
        Dictionary with domain samples and GW workspace representations
    """
    if len(domain_names) != 2:
        raise ValueError(f"This function requires exactly 2 domains, got {len(domain_names)}")
    
    model = model.to(device)
    model.eval()
    
    domain_a, domain_b = domain_names
    
    # Initialize containers for original and decoded representations
    domain_a_orig = []
    domain_b_orig = []
    domain_a_gw_encoded = []
    domain_b_gw_encoded = []
    domain_a_decoded = []
    domain_b_decoded = []
    gw_reps = []
    
    # Get the appropriate dataloader based on the split
    if split == "train":
        dataloader = data_module.train_dataloader()
    elif split == "val":
        dataloader = data_module.val_dataloader()
    elif split == "test":
        dataloader = data_module.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Calculate number of samples needed
    total_samples = 0
    
    # Process data in batches
    with torch.no_grad():
        try:
            batch_iterator = iter(dataloader)
            
            while total_samples < n_samples:
                try:
                    # Get a batch from the dataloader
                    batch = next(batch_iterator)
                    
                    # Process the batch (extract domain data)
                    processed_batch = process_batch(batch, device)
                    
                    # Check if both domains are in the processed batch
                    if domain_a not in processed_batch or domain_b not in processed_batch:
                        continue
                    
                    # Get data for each domain
                    latent_a = processed_batch[domain_a]
                    latent_b = processed_batch[domain_b]
                    
                    # Apply domain-specific projectors if needed (e.g., for text domain)
                    if domain_a == 't' and hasattr(model.domain_mods[domain_a], 'projector'):
                        latent_a = model.domain_mods[domain_a].projector(latent_a)
                    if domain_b == 't' and hasattr(model.domain_mods[domain_b], 'projector'):
                        latent_b = model.domain_mods[domain_b].projector(latent_b)
                    
                    # Encode to GW representation
                    gw_state_a = model.gw_encoders[domain_a](latent_a)
                    gw_state_b = model.gw_encoders[domain_b](latent_b)
                    
                    # Handle v_latents with extra dimensions (mean and variance)
                    if domain_a == 'v_latents' and gw_state_a.dim() > 2:
                        # Take only the first component (mean vector)
                        gw_state_a = gw_state_a[:, 0, :]
                    
                    if domain_b == 'v_latents' and gw_state_b.dim() > 2:
                        # Take only the first component (mean vector)
                        gw_state_b = gw_state_b[:, 0, :]
                    
                    # Create fused GW representation using model's fusion weights
                    fused_gw = model.fuse({
                        domain_a: gw_state_a,
                        domain_b: gw_state_b
                    }, None)  # Selection scores are ignored in GWModuleConfigurableFusion
                    
                    # Decode fused representation to both domains
                    decoded_a = model.gw_decoders[domain_a](fused_gw)
                    decoded_b = model.gw_decoders[domain_b](fused_gw)
                    
                    # Store results - use either latents or GW-encoded vectors based on parameter
                    if use_gw_encoded:
                        domain_a_orig.append(gw_state_a.cpu())
                        domain_b_orig.append(gw_state_b.cpu())
                    else:
                        domain_a_orig.append(latent_a.cpu())
                        domain_b_orig.append(latent_b.cpu())
                    
                    # Also store GW-encoded vectors for later analysis
                    domain_a_gw_encoded.append(gw_state_a.cpu())
                    domain_b_gw_encoded.append(gw_state_b.cpu())
                    
                    # Store decoded representations and fused GW state
                    domain_a_decoded.append(decoded_a.cpu())
                    domain_b_decoded.append(decoded_b.cpu())
                    gw_reps.append(fused_gw.cpu())
                    
                    # Update the total samples count
                    batch_size = latent_a.size(0)
                    total_samples += batch_size
                    
                    if total_samples % 1000 == 0:
                        print(f"Processed {total_samples} samples")
                    
                    # Stop if we've collected enough samples
                    if total_samples >= n_samples:
                        break
                        
                except StopIteration:
                    print(f"Reached end of dataloader after {total_samples} samples")
                    break
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"Error with data processing: {e}")
            import traceback
            traceback.print_exc()
    
    if total_samples == 0:
        raise ValueError(f"Could not generate any samples from the {split} dataset")
    
    print(f"Generated {total_samples} samples from the {split} dataset")
    
    # Create appropriate source prefix based on what was stored
    source_prefix = "gw_encoded" if use_gw_encoded else "latent"
    
    # Concatenate batches and trim to the requested number of samples
    try:
        result = {
            f"{domain_a}_{source_prefix}": torch.cat(domain_a_orig, dim=0)[:n_samples],
            f"{domain_b}_{source_prefix}": torch.cat(domain_b_orig, dim=0)[:n_samples],
            f"{domain_a}_gw_encoded": torch.cat(domain_a_gw_encoded, dim=0)[:n_samples],
            f"{domain_b}_gw_encoded": torch.cat(domain_b_gw_encoded, dim=0)[:n_samples],
            f"{domain_a}_decoded": torch.cat(domain_a_decoded, dim=0)[:n_samples],
            f"{domain_b}_decoded": torch.cat(domain_b_decoded, dim=0)[:n_samples],
            "gw_rep": torch.cat(gw_reps, dim=0)[:n_samples]
        }
        print(f"Final output keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"Error concatenating results: {e}")
        import traceback
        traceback.print_exc()
        raise

def process_batch(batch, device):
    """Process and normalize batch data."""
    processed_batch = {}
    
    # Handle tuple batch format (common for CombinedLoader)
    if isinstance(batch, tuple):
        for item in batch:
            if isinstance(item, dict):
                for k, v in item.items():
                    # Handle frozenset keys (standard in CombinedLoader)
                    if isinstance(k, frozenset):
                        domain_name = next(iter(k))
                        
                        # Extract tensor from value
                        if hasattr(v, 'bert'):
                            processed_batch[domain_name] = v.bert.to(device)
                        elif isinstance(v, dict) and domain_name in v:
                            value = v[domain_name]
                            if hasattr(value, 'bert'):
                                processed_batch[domain_name] = value.bert.to(device)
                            else:
                                processed_batch[domain_name] = value.to(device)
                        else:
                            processed_batch[domain_name] = v.to(device)
                    else:
                        # Regular key, just move to device
                        processed_batch[k] = v.to(device)
    # Handle different batch formats
    elif isinstance(batch, list) and len(batch) > 0:
        # Handle list batches
        for item in batch:
            if isinstance(item, dict):
                for k, v in item.items():
                    domain_name = next(iter(k)) if isinstance(k, frozenset) else k
                    # Extract tensor from complex objects if needed
                    if hasattr(v, 'bert'):
                        processed_batch[domain_name] = v.bert.to(device)
                    elif isinstance(v, dict) and domain_name in v:
                        value = v[domain_name]
                        if hasattr(value, 'bert'):
                            processed_batch[domain_name] = value.bert.to(device)
                        else:
                            processed_batch[domain_name] = value.to(device)
                    else:
                        processed_batch[domain_name] = v.to(device)
    elif isinstance(batch, dict):
        # Handle dictionary batches
        for k, v in batch.items():
            domain_name = next(iter(k)) if isinstance(k, frozenset) else k
            if hasattr(v, 'bert'):
                processed_batch[domain_name] = v.bert.to(device)
            elif isinstance(v, dict) and domain_name in v:
                value = v[domain_name]
                if hasattr(value, 'bert'):
                    processed_batch[domain_name] = value.bert.to(device)
                else:
                    processed_batch[domain_name] = value.to(device)
            else:
                processed_batch[domain_name] = v.to(device)
    
    # Apply domain-specific processing
    processed_result = processed_batch.copy()
    for domain_name, domain_input in processed_batch.items():
        # Fix shape for v_latents domain (common issue with extra dimensions)
        if domain_name == 'v_latents' and domain_input.dim() > 2:
            # Take only the first element along dimension 1 (mean vector)
            processed_result[domain_name] = domain_input[:, 0, :]
    
    return processed_result

def main():
    """Parse arguments and run PID analysis."""
    parser = argparse.ArgumentParser(description="Analyze information flow in models")
    
    # Training/evaluation mode
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--find-latest-checkpoint", action="store_true", help="Find latest checkpoint in fusion dir")
    parser.add_argument("--single-model", action="store_true", help="Analyze a single model")
    
    # Directories
    parser.add_argument("--fusion-dir", default="checkpoints/fusion", help="Directory with fusion checkpoints")
    parser.add_argument("--output-dir", default="pid_results", help="Directory to save results")
    
    # Model configuration
    parser.add_argument("--source-domain", default="both", choices=["v_latents", "t", "both"],
                       help="Domain to use for correlation with cluster labels")
    parser.add_argument("--domain-configs", nargs="+", default=[], help="Domain configuration files")
    parser.add_argument("--source-config", type=str, default=None, help="Source configuration JSON string")
    parser.add_argument("--target-config", type=str, default=None, help="Target configuration")
    
    # Data generation parameters
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--num-clusters", type=int, default=10, help="Number of clusters for synthetic labels")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training/evaluation")
    
    # Training parameters
    parser.add_argument("--discrim-epochs", type=int, default=40, help="Number of epochs to train discriminators")
    parser.add_argument("--ce-epochs", type=int, default=10, help="Number of epochs to train CE alignment")
    parser.add_argument("--discrim-hidden-dim", type=int, default=64, help="Hidden dimension for discriminator")
    parser.add_argument("--discrim-layers", type=int, default=5, help="Number of layers in discriminator")
    parser.add_argument("--use-compile", action="store_true", help="Use torch.compile for model optimization")
    
    # Dataset parameters
    parser.add_argument("--use-dataset", action="store_true", help="Use real data from the dataset instead of synthetic")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the shapes dataset")
    parser.add_argument("--dataset-split", type=str, default="test", choices=["train", "val", "test"], 
                        help="Dataset split to use for analysis")
    
    # W&B configuration
    parser.add_argument("--wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb-project", default="pid-analysis", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity name")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.test:
        test_fixes()
        return
    
    # Setup domain modules
    domain_modules = {}
    
    # Handle domain configs
    if args.domain_configs:
        try:
            domain_configs = []
            for config_path in args.domain_configs:
                if config_path.endswith(".ckpt"):
                    # This is a direct checkpoint path
                    if "domain_v" in config_path:
                        domain_name = "v_latents"
                    elif "domain_t" in config_path:
                        domain_name = "t"
                    else:
                        domain_name = Path(config_path).stem
                    
                    domain_configs.append({
                        "name": domain_name,
                        "domain_type": domain_name,
                        "checkpoint_path": config_path
                    })
            
            if domain_configs:
                from shimmer_ssd.modules.domains.pretrained import load_pretrained_module
                from shimmer_ssd.config import LoadedDomainConfig, DomainModuleVariant
                
                for config in domain_configs:
                    domain_type = config["domain_type"]
                    if domain_type == "v_latents":
                        domain_variant = DomainModuleVariant.v_latents
                    elif domain_type == "t":
                        domain_variant = DomainModuleVariant.t
                    else:
                        domain_variant = domain_type
                    
                    domain_config = LoadedDomainConfig(
                        domain_type=domain_variant,
                        checkpoint_path=config["checkpoint_path"],
                        args={}
                    )
                    
                    domain_module = load_pretrained_module(domain_config)
                    domain_modules[config["name"]] = domain_module
                    print(f"Loading domain module: {config['name']}")
        except ImportError:
            print("shimmer_ssd not found, cannot load domain modules from config")
    
    # Setup source and target configs
    if args.source_config:
        try:
            source_config = json.loads(args.source_config)
        except json.JSONDecodeError:
            # Try parsing as key-value pairs
            source_config = {}
            parts = args.source_config.split(",")
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    source_config[key.strip()] = value.strip()
    else:
        # Default source config
        source_config = {
            "v_latents": "v_latents_latent", 
            "t": "t_latent"
        }
    
    # Set target config
    target_config = args.target_config
    
    # Create data module if using real data
    data_module = None
    if args.use_dataset:
        try:
            # Add the simple-shapes-dataset directory to path
            import sys
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple-shapes-dataset"))
            
            from simple_shapes_dataset.data_module import SimpleShapesDataModule
            from simple_shapes_dataset.domain import DomainDesc
            
            # Find dataset path
            dataset_path = args.dataset_path
            if dataset_path is None:
                dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_shapes_dataset/simple_shapes_dataset")
                if not os.path.exists(dataset_path):
                    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple-shapes-dataset/sample_dataset")
                    print(f"Full dataset not found, falling back to sample dataset at: {dataset_path}")
            
            print(f"Using dataset at: {dataset_path}")
            
            # Create domain classes and args
            domain_classes = {}
            domain_args = {}
            
            # Set up domain classes based on loaded modules
            for domain_name, domain_module in domain_modules.items():
                if domain_name == "v_latents":
                    from simple_shapes_dataset.domain import SimpleShapesPretrainedVisual
                    domain_classes[DomainDesc(base="v", kind="v_latents")] = SimpleShapesPretrainedVisual
                    
                    # Define presaved path
                    domain_args["v_latents"] = {
                        "presaved_path": "calmip-822888_epoch=282-step=1105680_future.npy",
                        "use_unpaired": False
                    }
                elif domain_name == "t":
                    from simple_shapes_dataset.domain import SimpleShapesText
                    domain_classes[DomainDesc(base="t", kind="t")] = SimpleShapesText
            
            # Define domain proportions
            domain_proportions = {}
            for domain_name in domain_modules.keys():
                domain_proportions[frozenset([domain_name])] = 1.0
                
            # Create custom collate function
            try:
                from torch.utils.data._utils.collate import default_collate
                import torch
                
                def custom_collate_fn(batch):
                    # Simple collate function that handles the basic case
                    # Process list type batches (common in SimpleShapesDataset)
                    if isinstance(batch, list) and len(batch) > 0:
                        if isinstance(batch[0], dict):
                            # Handle domain-specific collation
                            result = {}
                            keys = batch[0].keys()
                            
                            for key in keys:
                                # Skip if key is not in all samples
                                if not all(key in b for b in batch):
                                    continue
                                    
                                # Collect values for this domain
                                try:
                                    # Special handling for text domain
                                    if key == 't':
                                        values = []
                                        for b in batch:
                                            if hasattr(b[key], 'bert'):
                                                # It's a Text object with a bert attribute
                                                values.append(b[key].bert)
                                            elif isinstance(b[key], dict) and 'bert' in b[key]:
                                                # It's a dict with a bert key
                                                values.append(b[key]['bert'])
                                            else:
                                                values.append(b[key])
                                                
                                        # Try to stack tensors if possible
                                        if all(isinstance(v, torch.Tensor) for v in values):
                                            result[key] = torch.stack(values)
                                        else:
                                            result[key] = values
                                    else:
                                        # Standard handling for other domains
                                        values = [b[key] for b in batch]
                                        result[key] = default_collate(values)
                                except Exception as e:
                                    print(f"Warning: Collation error for {key}: {e}")
                                    # If default_collate fails, preserve the list structure
                                    values = [b[key] for b in batch]
                                    result[key] = values
                                    
                            return result
                    
                    # Handle dict batches (often from CombinedLoader)
                    if isinstance(batch, dict):
                        result = {}
                        for domain_key, domain_values in batch.items():
                            # Handle frozenset keys (standard in CombinedLoader)
                            if isinstance(domain_key, frozenset):
                                domain_name = next(iter(domain_key))
                                if isinstance(domain_values, dict) and domain_name in domain_values:
                                    result[domain_name] = domain_values[domain_name]
                                else:
                                    result[domain_name] = domain_values
                            else:
                                result[domain_key] = domain_values
                        return result
                    
                    # Try default collation as fallback
                    try:
                        return default_collate(batch)
                    except Exception:
                        # If all else fails, just return the batch
                        return batch
            except ImportError as e:
                print(f"Error importing collate function: {e}")
                # Define a simple collate function as fallback
                def custom_collate_fn(batch):
                    return batch
            
            print(f"Setting up data module with domain classes: {domain_classes}")
            print(f"Domain proportions: {domain_proportions}")
            
            # Create data module
            data_module = SimpleShapesDataModule(
                dataset_path=dataset_path,
                domain_classes=domain_classes,
                domain_proportions=domain_proportions,
                batch_size=args.batch_size,
                num_workers=0,  # Use 0 for debugging
                seed=42,
                domain_args=domain_args,
                collate_fn=custom_collate_fn
            )
            
            # Setup data module explicitly
            data_module.setup()
            
            # Print dataset information
            train_dataset = data_module.train_dataset
            val_dataset = data_module.val_dataset
            test_dataset = data_module.test_dataset
            
            print("\nDataset Information:")
            for domain, dataset in train_dataset.items():
                print(f"Train domain {domain}: {len(dataset)} samples")
            if val_dataset:
                for domain, dataset in val_dataset.items():
                    print(f"Val domain {domain}: {len(dataset)} samples")
            if test_dataset:
                for domain, dataset in test_dataset.items():
                    print(f"Test domain {domain}: {len(dataset)} samples")
            
            print(f"Using {args.dataset_split} split for analysis")
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            import traceback
            traceback.print_exc()
            data_module = None
            print("Falling back to synthetic data generation")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir or "pid_results"
    model_name = "single_model" if args.single_model else "multiple_models"
    output_dir = os.path.join(output_base, f"{model_name}_{timestamp}")
    
    if args.single_model:
        # Analysis for a single model
        if args.find_latest_checkpoint:
            print(f"Finding latest checkpoint in {args.fusion_dir}")
            latest_checkpoints = find_latest_model_checkpoints(args.fusion_dir, max_configs=1)
            if not latest_checkpoints:
                print(f"No checkpoints found in {args.fusion_dir}")
                return
            
            model_path = latest_checkpoints[0]
            print(f"Using latest checkpoint: {model_path}")
        else:
            model_path = args.fusion_dir
            
        analyze_model(
            model_path=model_path,
            domain_modules=domain_modules,
            output_dir=output_dir,
            source_config=source_config,
            target_config=target_config,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            num_clusters=args.num_clusters,
            discrim_epochs=args.discrim_epochs,
            ce_epochs=args.ce_epochs,
            discrim_hidden_dim=args.discrim_hidden_dim,
            discrim_layers=args.discrim_layers,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            data_module=data_module,  # Pass the data module
            dataset_split=args.dataset_split,  # Specify the split
            use_gw_encoded=False,
            use_compile=args.use_compile,
        )
    else:
        # Analyze multiple models
        analyze_multiple_models(
            checkpoint_dir=args.fusion_dir,
            domain_modules=domain_modules,
            output_dir=output_dir,
            source_config=source_config,
            target_config=target_config,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            num_clusters=args.num_clusters,
            discrim_epochs=args.discrim_epochs,
            ce_epochs=args.ce_epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
            discrim_hidden_dim=args.discrim_hidden_dim,
            discrim_layers=args.discrim_layers,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            data_module=data_module,  # Pass the data module
            dataset_split=args.dataset_split,  # Specify the split
            use_gw_encoded=False,
            use_compile=args.use_compile,
        )

# Add a new test function
def test_autocast_fix():
    """
    Test the autocast and GradScaler fix.
    """
    import torch
    
    print("\nRunning autocast and GradScaler fix test...")
    print("PyTorch version:", torch.__version__)
    
    # Create a simple dataset for testing
    x1 = torch.rand(10, 5)
    x2 = torch.rand(10, 5)  # Make compatible shapes for matrix multiplication
    labels = torch.randint(0, 2, (10,))
    
    # Print input shapes
    print(f"Input shapes: x1={x1.shape}, x2={x2.shape}, labels={labels.shape}")
    
    dataset = MultimodalDataset([x1, x2], labels)
    
    # Testing direct imports first to understand which method is correct
    print("\nTesting different autocast implementations...")
    
    try:
        # Try importing torch.amp autocast directly
        import torch.amp
        print("torch.amp exists and was imported")
        # Print the signature of autocast
        from inspect import signature
        print(f"torch.amp.autocast signature: {signature(torch.amp.autocast)}")
    except (ImportError, AttributeError) as e:
        print(f"Could not import torch.amp: {e}")
    
    try:
        # Try importing torch.cuda.amp directly for CUDA
        if torch.cuda.is_available():
            import torch.cuda.amp
            print("torch.cuda.amp exists and was imported")
            from inspect import signature
            print(f"torch.cuda.amp.autocast signature: {signature(torch.cuda.amp.autocast)}")
            print("Note: torch.cuda.amp has a FutureWarning about deprecation")
    except (ImportError, AttributeError) as e:
        print(f"Could not import torch.cuda.amp: {e}")
    
    try:
        # Try importing torch.cpu.amp directly for CPU
        import torch.cpu.amp
        print("torch.cpu.amp exists and was imported")
        from inspect import signature
        print(f"torch.cpu.amp.autocast signature: {signature(torch.cpu.amp.autocast)}")
    except (ImportError, AttributeError) as e:
        print(f"Could not import torch.cpu.amp: {e}")
    
    # Now try the actual GradScaler
    try:
        # Test GradScaler 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nUsing device: {device}")
        
        # Use the new recommended syntax from torch.amp
        print("Testing torch.amp.GradScaler...")
        scaler = torch.amp.GradScaler()
        print("✅ torch.amp.GradScaler created successfully")
    except Exception as e:
        print(f"❌ Error creating torch.amp.GradScaler: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test autocast with the proper torch.amp namespace
    try:
        print("\nTesting the recommended torch.amp.autocast...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with torch.amp.autocast(device_type=device):
            output = x1 @ x2.t()
            print(f"✅ torch.amp.autocast worked correctly. Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Error with torch.amp.autocast: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed successfully! You should use torch.amp.autocast(device_type='cuda'|'cpu') syntax.")

def plot_stacked_pid(results: List[Dict], output_dir: str, wandb_run=None):
    """
    Plot a single, publication-quality stacked PID components plot using viridis colormap.
    
    Args:
        results: List of PID results
        output_dir: Directory to save plots
        wandb_run: Optional wandb run to log plots to
    """
    if not results:
        return
    
    # Extract fusion weights and PID values
    weights_a = []
    weights_b = []
    redundancy = []
    unique_a = []
    unique_b = []
    synergy = []
    model_names = []
    domains = []
    
    for result in results:
        fusion_weights = result.get("fusion_weights", {})
        pid_values = result.get("pid_values", {})
        result_domains = result.get("domains", [])
        
        if len(result_domains) != 2 or len(fusion_weights) != 2:
            continue
        
        domain_a, domain_b = result_domains
        if not domains:
            domains = [domain_a, domain_b]
            
        weights_a.append(fusion_weights.get(domain_a, 0))
        weights_b.append(fusion_weights.get(domain_b, 0))
        
        redundancy.append(pid_values.get("redundancy", 0))
        unique_a.append(pid_values.get("unique1", 0))
        unique_b.append(pid_values.get("unique2", 0))
        synergy.append(pid_values.get("synergy", 0))
        model_names.append(result.get("model_name", "unknown"))
    
    if not weights_a:
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create weight ratios
    weight_ratios = [a/b if b > 0 else float('inf') for a, b in zip(weights_a, weights_b)]
    valid_indices = [i for i, x in enumerate(weight_ratios) if x != float('inf')]
    
    if valid_indices:
        # Filter out infinite values
        filtered_ratios = [weight_ratios[i] for i in valid_indices]
        filtered_redundancy = [redundancy[i] for i in valid_indices]
        filtered_unique_a = [unique_a[i] for i in valid_indices]
        filtered_unique_b = [unique_b[i] for i in valid_indices]
        filtered_synergy = [synergy[i] for i in valid_indices]
        filtered_model_names = [model_names[i] for i in valid_indices]
        
        # Set publication-quality plot style
        plt.style.use('seaborn-v0_8-whitegrid')  # Updated to newer matplotlib style name
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Changed from 'Arial' to 'DejaVu Sans'
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        # Create a high-resolution figure
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Get colors from viridis colormap - better for color blindness and printing
        # Using plt.get_cmap instead of cm.get_cmap to avoid deprecation warning
        cmap = plt.get_cmap('viridis', 4)
        colors = [cmap(i) for i in range(4)]
        
        # Sort by weight ratio
        sorted_indices = sorted(range(len(filtered_ratios)), key=lambda i: filtered_ratios[i])
        sorted_ratios = [filtered_ratios[i] for i in sorted_indices]
        sorted_redundancy = [filtered_redundancy[i] for i in sorted_indices]
        sorted_unique_a = [filtered_unique_a[i] for i in sorted_indices]
        sorted_unique_b = [filtered_unique_b[i] for i in sorted_indices]
        sorted_synergy = [filtered_synergy[i] for i in sorted_indices]
        sorted_model_names = [filtered_model_names[i] for i in sorted_indices]
        
        # Create x-axis labels with weight ratios
        x_labels = [f"{ratio:.2f}" for ratio in sorted_ratios]
        x = range(len(sorted_ratios))
        
        # Plot stacked bars with viridis colormap
        plt.bar(x, sorted_redundancy, label="Redundancy", color=colors[0], width=0.8)
        plt.bar(x, sorted_unique_a, bottom=sorted_redundancy, label=f"Unique to {domains[0]}", color=colors[1], width=0.8)
        bottom = [r + ua for r, ua in zip(sorted_redundancy, sorted_unique_a)]
        plt.bar(x, sorted_unique_b, bottom=bottom, label=f"Unique to {domains[1]}", color=colors[2], width=0.8)
        bottom = [b + ub for b, ub in zip(bottom, sorted_unique_b)]
        plt.bar(x, sorted_synergy, bottom=bottom, label="Synergy", color=colors[3], width=0.8)
        
        plt.xlabel("Fusion Weight Ratio ($w_{" + domains[0] + "} / w_{" + domains[1] + "}$)")
        plt.ylabel("Information (bits)")
        plt.title("Partial Information Decomposition by Weight Ratio")
        
        # Format x-axis with proper tick spacing
        if len(sorted_ratios) > 10:
            tick_indices = list(range(0, len(sorted_ratios), max(1, len(sorted_ratios) // 10)))
            plt.xticks([x[i] for i in tick_indices], [x_labels[i] for i in tick_indices], rotation=45)
        else:
            plt.xticks(x, x_labels, rotation=45)
        
        # Add a light grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend with attractive layout
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True, fancybox=True, shadow=True)
        
        # Tight layout with additional padding at bottom for legend
        plt.tight_layout(rect=[0, 0.1, 1, 0.98])
        
        # Save figure
        stacked_plot_path = os.path.join(plots_dir, "pid_stacked_viridis.pdf")
        plt.savefig(stacked_plot_path, bbox_inches="tight")
        
        # Also save as PNG for preview
        stacked_plot_png_path = os.path.join(plots_dir, "pid_stacked_viridis.png")
        plt.savefig(stacked_plot_png_path, dpi=300, bbox_inches="tight")
        
        # Log to wandb if available
        if HAS_WANDB and wandb_run is not None:
            wandb.log({"plots/pid_stacked_viridis": wandb.Image(stacked_plot_png_path)})
        
        plt.close()
        
        print(f"PID stacked plot saved to {stacked_plot_path}")
        return stacked_plot_path

def test_simple_discrim():
    """
    Test function to verify that the simple_discrim normalization is working properly.
    
    This test creates synthetic one-hot inputs and labels, then checks that
    the returned conditional probabilities p(y|x) sum to 1 for each x.
    """
    print("\nTesting simple_discrim normalization...")
    
    # Create synthetic data with 2 domains
    batch_size = 100
    dim1, dim2 = 5, 4
    num_labels = 3
    
    # Create one-hot inputs
    x1 = torch.zeros(batch_size, dim1)
    x2 = torch.zeros(batch_size, dim2)
    for i in range(batch_size):
        idx1 = i % dim1
        idx2 = i % dim2
        x1[i, idx1] = 1.0
        x2[i, idx2] = 1.0
    
    # Create synthetic labels
    y = torch.randint(0, num_labels, (batch_size, 1))
    
    # Create discriminator
    discriminator = simple_discrim([x1, x2], y, num_labels)
    
    # Test normalization for a few input combinations
    test_inputs = []
    for i in range(min(dim1, 3)):
        for j in range(min(dim2, 3)):
            one_hot_1 = torch.zeros(1, dim1)
            one_hot_2 = torch.zeros(1, dim2)
            one_hot_1[0, i] = 1.0
            one_hot_2[0, j] = 1.0
            test_inputs.append((one_hot_1, one_hot_2))
    
    for i, (test_x1, test_x2) in enumerate(test_inputs):
        # Get log probabilities for this input combination
        log_probs = discriminator(test_x1, test_x2)
        # Convert log probs to probs
        probs = torch.exp(log_probs)
        # Check normalization - should sum to 1
        prob_sum = probs.sum().item()
        
        x1_idx = test_x1.argmax().item()
        x2_idx = test_x2.argmax().item()
        print(f"Input ({x1_idx}, {x2_idx}): Sum of p(y|x) = {prob_sum:.6f} (Should be 1.0)")
        
        # Check if close to 1
        if not (0.9999 <= prob_sum <= 1.0001):
            print(f"WARNING: Probabilities don't sum to 1: {prob_sum}")
            print(f"Probabilities: {probs.tolist()}")
    
    print("simple_discrim test complete\n")

def test_fixes():
    """
    Test function to verify our fixes for AMP and Sinkhorn projection.
    """
    print("\nTesting AMP and Sinkhorn fixes...")
    
    # 1. Test AMP fix
    print("Testing AMP import fix...")
    try:
        # Test creating a GradScaler
        if torch.cuda.is_available():
            scaler = amp.GradScaler()  # No need to provide device type as argument
        else:
            # For CPU AMP or fallback
            scaler = amp.GradScaler()
        print("✅ GradScaler created successfully")
        
        # Test autocast
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with amp.autocast(device_type=device, enabled=True):  # Correct parameters
            print("✅ autocast context entered successfully")
    except Exception as e:
        print(f"❌ Error with AMP: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Test Sinkhorn projection fix
    print("\nTesting Sinkhorn projection dimension handling...")
    
    # Create test matrices and probability vectors
    matrix = torch.rand(10, 8)
    
    # Test case 1: x_probs too short (padding case)
    x1_short = torch.ones(5) / 5  # Short probability vector
    x2_short = torch.ones(6) / 6  # Short probability vector
    
    try:
        result = sinkhorn_probs(matrix, x1_short, x2_short, tol=0.01, max_iter=10)
        print("✅ Sinkhorn with short probability vectors (padding case) worked")
        # Verify that result dimensions match matrix
        if result.shape == matrix.shape:
            print(f"✅ Result shape {result.shape} matches matrix shape {matrix.shape}")
        else:
            print(f"❌ Result shape {result.shape} doesn't match matrix shape {matrix.shape}")
    except Exception as e:
        print(f"❌ Error with Sinkhorn (short case): {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 2: x_probs too long (truncation case with warning)
    x1_long = torch.ones(15) / 15  # Long probability vector
    x2_long = torch.ones(16) / 16  # Long probability vector
    
    try:
        result = sinkhorn_probs(matrix, x1_long, x2_long, tol=0.1, max_iter=10)
        print("✅ Sinkhorn with long probability vectors (truncation case) worked")
        # Verify that result dimensions match matrix
        if result.shape == matrix.shape:
            print(f"✅ Result shape {result.shape} matches matrix shape {matrix.shape}")
        else:
            print(f"❌ Result shape {result.shape} doesn't match matrix shape {matrix.shape}")
    except Exception as e:
        print(f"❌ Error with Sinkhorn (long case): {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 3: Extreme mismatch (should raise error)
    x1_extreme = torch.ones(500) / 500  # Extremely long probability vector
    x2_extreme = torch.ones(500) / 500  # Extremely long probability vector
    
    try:
        result = sinkhorn_probs(matrix, x1_extreme, x2_extreme, tol=0.1, max_iter=10)
        print("❓ Sinkhorn with extremely long probability vectors didn't raise an error as expected")
    except ValueError as e:
        print(f"✅ Sinkhorn correctly raised an error for extreme mismatch: {e}")
    except Exception as e:
        print(f"❌ Unexpected error with Sinkhorn (extreme case): {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTests completed")

def find_latest_model_checkpoints(base_dir: str, max_configs: Optional[int] = None) -> List[str]:
    """
    Find the latest epoch checkpoint of the v_latents_0.4_t_0.6 model.
    
    Args:
        base_dir: Base directory containing configuration subdirectories
        max_configs: Maximum number of most recent configurations to include
    
    Returns:
        List containing the path to the latest epoch checkpoint of the v_latents_0.4_t_0.6 model
    """
    import re
    
    # Pattern to extract epoch from filename - flexible to handle various formats
    epoch_pattern = re.compile(r'model_epoch_(\d+)')
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory {base_dir} does not exist")
        return []
    
    # Find the config directory for v_latents_0.4_t_0.6
    target_config_dir = None
    for config_dir in Path(base_dir).glob("config_*_v_latents_0.4_t_0.6"):
        if config_dir.is_dir():
            target_config_dir = config_dir
            break
    
    if not target_config_dir:
        print("Warning: Could not find config directory for v_latents_0.4_t_0.6")
        return []
    
    print(f"Found target config directory: {target_config_dir}")
    
    # Find all model checkpoint files in this config directory
    checkpoints = list(target_config_dir.glob("model_epoch_*.pt"))
    
    if not checkpoints:
        print(f"Warning: No checkpoints found in {target_config_dir}")
        return []
    
    # Extract epoch number from each checkpoint filename
    checkpoint_epochs = []
    for checkpoint in checkpoints:
        match = epoch_pattern.search(checkpoint.name)
        if match:
            epoch = int(match.group(1))
            checkpoint_epochs.append((epoch, checkpoint))
    
    # Sort by epoch and get the latest one
    if checkpoint_epochs:
        latest_epoch, latest_checkpoint = max(checkpoint_epochs, key=lambda x: x[0])
        print(f"Found latest checkpoint for v_latents_0.4_t_0.6: {latest_checkpoint.name} (epoch {latest_epoch})")
        return [str(latest_checkpoint)]
    else:
        print("Warning: No valid checkpoints found in target config directory")
        return []

def analyze_multiple_models_from_list(
    checkpoint_list: List[str],
    domain_configs: List[Dict[str, Any]],
    output_dir: str,
    n_samples: int = 10000,
    batch_size: int = 128,
    num_clusters: int = 10,
    discrim_epochs: int = 40,
    ce_epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_domain_for_labels: str = "both",
    discrim_hidden_dim: int = 64,
    discrim_layers: int = 5,
    use_wandb: bool = True,
    wandb_project: str = "pid-analysis",
    wandb_entity: Optional[str] = None,
    data_module=None,
    dataset_split: str = "test",
    use_gw_encoded: bool = False,
    use_compile: bool = True,
) -> List[Dict[str, Any]]:
    """
    Analyze multiple models from a list of checkpoint paths.
    
    Args:
        checkpoint_list: List of checkpoint paths to analyze
        domain_configs: List of domain module configurations
        output_dir: Directory to save results
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        num_clusters: Number of clusters for synthetic labels
        discrim_epochs: Number of epochs to train discriminators
        ce_epochs: Number of epochs to train CE alignment
        device: Device to use for computation
        use_domain_for_labels: Which domain to use for creating labels
        discrim_hidden_dim: Hidden dimension for discriminator networks
        discrim_layers: Number of layers in discriminator networks
        use_wandb: Whether to log results to Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        data_module: Optional data module for real data
        dataset_split: Dataset split to use
        use_gw_encoded: Whether to use GW-encoded vectors
        use_compile: Whether to use torch.compile for model optimization
        
    Returns:
        List of dictionaries containing results for each model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize W&B if requested
    wandb_run = None
    if use_wandb and HAS_WANDB:
        try:
            # Create a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize wandb run
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"multi_model_pid_{timestamp}",
                config={
                    "num_models": len(checkpoint_list),
                    "n_samples": n_samples,
                    "batch_size": batch_size,
                    "num_clusters": num_clusters,
                    "discrim_epochs": discrim_epochs,
                    "ce_epochs": ce_epochs,
                    "use_domain_for_labels": use_domain_for_labels,
                    "discrim_hidden_dim": discrim_hidden_dim,
                    "discrim_layers": discrim_layers,
                    "data_source": "dataset" if data_module else "synthetic",
                    "use_gw_encoded": use_gw_encoded,
                    "use_compile": use_compile,
                }
            )
            print(f"Initialized W&B run: {wandb_run.name}")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            wandb_run = None
    
    # Load domain modules
    domain_modules = load_domain_modules([config for config in domain_configs])
    
    # Initialize results list
    results = []
    
    # Process each checkpoint in the list
    print(f"Processing {len(checkpoint_list)} checkpoints...")
    for i, (checkpoint_path, domain_config) in enumerate(zip(checkpoint_list, domain_configs)):
        print(f"\nAnalyzing model {i+1}/{len(checkpoint_list)}: {checkpoint_path}")
        
        # Get source and target configs based on domain_config
        source_config = {}
        target_config = ""
        analysis_domain = domain_config.get("analysis_domain", use_domain_for_labels)
        
        # Parse source_config from domain_config if available
        if "source_config" in domain_config:
            source_config = domain_config["source_config"]
        else:
            # Use default source config
            source_config = {
                "v_latents": "v_latents_latent", 
                "t": "t_latent"
            }
        
        # Parse target_config from domain_config if available
        if "target_config" in domain_config:
            target_config = domain_config["target_config"]
        else:
            # Use default target config
            target_config = "gw_latent"
        
        try:
            # Load model
            model = load_checkpoint(
                checkpoint_path=checkpoint_path,
                domain_modules=domain_modules,
                device=device
            )
            
            # Get domain names from model
            if hasattr(model, 'domain_mods') and model.domain_mods:
                domain_names = list(model.domain_mods.keys())
                print(f"Domain names from model: {domain_names}")
            else:
                domain_names = []
                print("Warning: No domain names found in model")
                continue  # Skip this model
            
            # Generate data for analysis
            generated_data = generate_samples_from_model(
                model=model,
                domain_names=domain_names,
                n_samples=n_samples,
                batch_size=batch_size,
                device=device,
                use_gw_encoded=use_gw_encoded,
                data_module=data_module,
                dataset_split=dataset_split
            )
            
            # Prepare data for PID analysis
            train_ds, test_ds, x1, x2, labels = prepare_dataset_pid_data(
                generated_data=generated_data,
                domain_names=domain_names,
                analysis_domain=analysis_domain,
                synthetic_labels=None,  # Generate synthetic labels
                num_clusters=num_clusters,
                use_domain_for_labels=use_domain_for_labels
            )
            
            # Use critic_ce_alignment directly since we've already prepared the data
            pid_vals, aligns, models = critic_ce_alignment(
                x1, x2, labels, num_clusters,
                train_ds, test_ds,
                learned_discrim=True, 
                discrim_epochs=discrim_epochs, 
                ce_epochs=ce_epochs,
                wandb_enabled=use_wandb,
                model_name=f"model_{i+1}",
                discrim_hidden_dim=discrim_hidden_dim,
                discrim_layers=discrim_layers,
                use_compile=use_compile,
            )
            
            # Prepare result dictionary
            model_result = {
                "model_index": i,
                "model_path": checkpoint_path,
                "domain_names": domain_names,
                "source_config": source_config,
                "target_config": target_config,
                "pid_values": pid_vals.detach().cpu().numpy().tolist(),
            }
            
            # Add to results list
            results.append(model_result)
            
            # Log to W&B if enabled
            if wandb_run is not None:
                try:
                    # Add model index to log
                    pid_components = ["redundancy", "unique1", "unique2", "synergy"]
                    for j, component in enumerate(pid_components):
                        wandb.log({f"model_{i+1}/{component}": pid_vals[j].item()})
                except Exception as e:
                    print(f"Failed to log to W&B: {e}")
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error analyzing model {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create combined visualizations
    print("\nCreating combined visualizations...")
    try:
        # Plot PID components
        plot_pid_components(results, output_dir, wandb_run)
        
        # Create stacked PID plot for publication
        plot_stacked_pid(results, output_dir, wandb_run)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Close W&B if it was initialized
    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Error closing W&B: {e}")
    
    return results

def get_allowed_sources_and_targets(domain_name: str) -> Dict[str, List[str]]:
    """
    Get the allowed sources and targets for a given domain.
    
    Args:
        domain_name: Name of the domain
        
    Returns:
        Dictionary of allowed sources and targets
    """
    allowed = {}
    
    if domain_name == "v_latents":
        allowed["sources"] = ["v_latents_latent", "v_latents_encoded"]
        allowed["targets"] = ["gw_latent", "gw_decoded"]
    elif domain_name == "t":
        allowed["sources"] = ["t_latent", "t_encoded"]
        allowed["targets"] = ["gw_latent", "gw_decoded"]
    else:
        allowed["sources"] = []
        allowed["targets"] = []
    
    return allowed

def validate_source_target_config(
    source_config: Dict[str, str],
    target_config: str,
    domain_names: List[str]
) -> None:
    """
    Validate the source and target configuration for PID analysis.
    
    Args:
        source_config: Dictionary mapping domain names to their source representations
        target_config: Target representation to use
        domain_names: List of domain names
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate domain names
    if len(domain_names) != 2:
        raise ValueError(f"Exactly 2 domains required, got {len(domain_names)}")
    
    # Validate source configuration
    if len(source_config) != 2:
        raise ValueError(f"Exactly 2 sources required, got {len(source_config)}")
    
    for domain, source in source_config.items():
        if domain not in domain_names:
            raise ValueError(f"Invalid domain name in source config: {domain}")
        
        allowed_sources = get_allowed_sources_and_targets(domain)['sources']
        if source not in allowed_sources:
            raise ValueError(f"Invalid source representation for {domain}: {source}. Allowed: {allowed_sources}")
    
    # Validate target configuration
    allowed_targets = []
    for domain in domain_names:
        allowed_targets.extend(get_allowed_sources_and_targets(domain)['targets'])
    
    if target_config not in allowed_targets:
        raise ValueError(f"Invalid target representation: {target_config}. Allowed: {allowed_targets}")

# First, create a helper function to convert tensors to lists
def prepare_for_json(obj):
    """
    Recursively convert PyTorch tensors and other non-serializable objects to Python primitives.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, (list, tuple)):
        return [prepare_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: prepare_for_json(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        # For custom objects, convert to dict representation
        return {'__class__': obj.__class__.__name__, 
                '__data__': prepare_for_json(obj.__dict__)}
    else:
        # Return the object as is if it's a primitive type
        return obj

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-fixes":
        test_fixes()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-simple-discrim":
        test_simple_discrim()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-autocast-fix":
        test_autocast_fix()
    else:
        main() 