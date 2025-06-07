"""
Synthetic data generation for testing PID analysis pipeline.
Creates Boolean functions with known ground truth PID values.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Dict, Tuple, List, Optional
import numpy as np
from pathlib import Path

#───────────────────────────────#
#     Configuration Section     #
#───────────────────────────────#

# Number of samples per function
DEFAULT_N_SAMPLES = 100_000

# Train/validation/test split fractions
DEFAULT_SPLIT = {
    "train": 0.8,
    "val":   0.1,
    "test":  0.1,
}

# Random seed for reproducibility
DEFAULT_SEED = 42

# Batch size for DataLoader
DEFAULT_BATCH_SIZE = 128

#───────────────────────────────#


def generate_boolean_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    functions: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic Boolean function data.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        functions: List of functions to generate (default: all 16 Boolean functions)
    """
    if functions is None:
        # All 16 basic Boolean functions
        functions = [
            'const_0', 'nor', 'nimp_b_a', 'not_a', 'nimp_a_b', 'not_b', 
            'xor', 'nand', 'and', 'xnor', 'id_b', 'imp_a_b', 
            'id_a', 'imp_b_a', 'or', 'const_1'
        ]
    
    torch.manual_seed(seed)
    
    # Generate random binary inputs
    input_a = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32)
    input_b = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32)
    
    data = {
        'input_a': input_a,
        'input_b': input_b
    }
    
    # Generate requested Boolean functions
    for func_name in functions:
        if func_name == 'and':
            # AND: output = a ∧ b (function 8)
            output = ((input_a > 0.5) & (input_b > 0.5)).float()
        elif func_name == 'or':
            # OR: output = a ∨ b (function 14)
            output = ((input_a > 0.5) | (input_b > 0.5)).float()
        elif func_name == 'xor':
            # XOR: output = a ⊕ b (function 6)
            output = ((input_a > 0.5) ^ (input_b > 0.5)).float()
        elif func_name == 'id_a':
            # Identity A: output = a (function 12, ignores b)
            output = (input_a > 0.5).float()
        elif func_name == 'id_b':
            # Identity B: output = b (function 10, ignores a)
            output = (input_b > 0.5).float()
        elif func_name == 'not_a':
            # NOT A: output = ¬a (function 3)
            output = (input_a <= 0.5).float()
        elif func_name == 'not_b':
            # NOT B: output = ¬b (function 5)
            output = (input_b <= 0.5).float()
        elif func_name == 'nand':
            # NAND: output = ¬(a ∧ b) (function 7)
            output = ~((input_a > 0.5) & (input_b > 0.5))
            output = output.float()
        elif func_name == 'nor':
            # NOR: output = ¬(a ∨ b) (function 1)
            output = ~((input_a > 0.5) | (input_b > 0.5))
            output = output.float()
        elif func_name == 'xnor':
            # XNOR: output = ¬(a ⊕ b) = (a ↔ b) (function 9)
            output = ~((input_a > 0.5) ^ (input_b > 0.5))
            output = output.float()
        elif func_name == 'imp_a_b':
            # Implication A→B: output = ¬a ∨ b (function 11)
            output = ((input_a <= 0.5) | (input_b > 0.5)).float()
        elif func_name == 'imp_b_a':
            # Implication B→A: output = a ∨ ¬b (function 13)
            output = ((input_a > 0.5) | (input_b <= 0.5)).float()
        elif func_name == 'nimp_a_b':
            # Non-implication A↛B: output = a ∧ ¬b (function 4)
            output = ((input_a > 0.5) & (input_b <= 0.5)).float()
        elif func_name == 'nimp_b_a':
            # Non-implication B↛A: output = ¬a ∧ b (function 2)
            output = ((input_a <= 0.5) & (input_b > 0.5)).float()
        elif func_name == 'const_0':
            # Constant 0: output = 0 (function 0)
            output = torch.zeros_like(input_a)
        elif func_name == 'const_1':
            # Constant 1: output = 1 (function 15)
            output = torch.ones_like(input_a)
        else:
            raise ValueError(f"Unknown function: {func_name}. Available functions: "
                           f"and, or, xor, nand, nor, xnor, id_a, id_b, not_a, not_b, "
                           f"imp_a_b, imp_b_a, nimp_a_b, nimp_b_a, const_0, const_1")
        
        data[func_name] = output.reshape(-1, 1)
    
    return data


def make_datasets(
    X: torch.Tensor, 
    Y: torch.Tensor,
    split: Dict[str, float] = None
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create train/val/test datasets from input/output tensors.
    
    Args:
        X: Input tensor
        Y: Output tensor
        split: Dictionary with train/val/test split fractions
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if split is None:
        split = DEFAULT_SPLIT
    
    full_ds = TensorDataset(X, Y)
    n = len(full_ds)
    n_train = int(split["train"] * n)
    n_val   = int(split["val"]   * n)
    n_test  = n - n_train - n_val
    return random_split(full_ds, [n_train, n_val, n_test])


def create_boolean_datasets(
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    functions: Optional[List[str]] = None,
    split: Dict[str, float] = None
) -> Dict[str, Dict[str, TensorDataset]]:
    """
    Create complete datasets for Boolean functions.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        functions: List of Boolean functions to generate (default: all 16 Boolean functions)
        split: Train/val/test split ratios
    
    Returns:
        Dictionary with structure: {function_name: {'train': ds, 'val': ds, 'test': ds}}
    """
    if functions is None:
        # All 16 basic Boolean functions
        functions = [
            'const_0', 'nor', 'nimp_b_a', 'not_a', 'nimp_a_b', 'not_b', 
            'xor', 'nand', 'and', 'xnor', 'id_b', 'imp_a_b', 
            'id_a', 'imp_b_a', 'or', 'const_1'
        ]
    
    if split is None:
        split = DEFAULT_SPLIT
    
    # Generate the raw data
    data = generate_boolean_data(n_samples, seed, functions)
    
    # Create input tensor (A, B)
    inputs = torch.cat([data['input_a'], data['input_b']], dim=1)
    
    datasets = {}
    
    for func_name in functions:
        if func_name in data:
            train_ds, val_ds, test_ds = make_datasets(inputs, data[func_name], split)
            datasets[func_name] = {
                'train': train_ds,
                'val': val_ds, 
                'test': test_ds
            }
    
    return datasets


def get_theoretical_pid_values() -> Dict[str, Dict[str, float]]:
    """
    Return theoretical PID values for Boolean functions.
    
    Based on known information theory results for these functions.
    Values computed using the theoretical framework for PID on Boolean functions.
    
    Returns:
        Dictionary with PID components for each function
    """
    return {
        # Function 0: Constant 0
        'const_0': {
            'unique_a': 0.0,      # Constant: no information from A
            'unique_b': 0.0,      # Constant: no information from B
            'redundant': 0.0,     # Constant: no shared information
            'synergistic': 0.0    # Constant: no synergistic information
        },
        
        # Function 1: NOR (¬(a ∨ b))
        'nor': {
            'unique_a': 0.189,    # NOR: some unique information from A
            'unique_b': 0.189,    # NOR: some unique information from B
            'redundant': 0.311,   # NOR: shared information about output
            'synergistic': 0.0    # NOR: no synergistic information
        },
        
        # Function 2: Non-implication B↛A (¬a ∧ b)
        'nimp_b_a': {
            'unique_a': 0.311,    # B AND NOT A: unique information from A
            'unique_b': 0.311,    # B AND NOT A: unique information from B
            'redundant': 0.0,     # B AND NOT A: no shared information
            'synergistic': 0.311  # B AND NOT A: synergistic information
        },
        
        # Function 3: NOT A (¬a)
        'not_a': {
            'unique_a': 1.0,      # NOT A: all information comes from A
            'unique_b': 0.0,      # NOT A: B provides no information
            'redundant': 0.0,     # NOT A: no shared information
            'synergistic': 0.0    # NOT A: no synergistic information
        },
        
        # Function 4: Non-implication A↛B (a ∧ ¬b)
        'nimp_a_b': {
            'unique_a': 0.311,    # A AND NOT B: unique information from A
            'unique_b': 0.311,    # A AND NOT B: unique information from B
            'redundant': 0.0,     # A AND NOT B: no shared information
            'synergistic': 0.311  # A AND NOT B: synergistic information
        },
        
        # Function 5: NOT B (¬b)
        'not_b': {
            'unique_a': 0.0,      # NOT B: A provides no information
            'unique_b': 1.0,      # NOT B: all information comes from B
            'redundant': 0.0,     # NOT B: no shared information
            'synergistic': 0.0    # NOT B: no synergistic information
        },
        
        # Function 6: XOR (a ⊕ b)
        'xor': {
            'unique_a': 0.0,      # XOR: no unique information from A alone
            'unique_b': 0.0,      # XOR: no unique information from B alone
            'redundant': 0.0,     # XOR: no shared information
            'synergistic': 1.0    # XOR: all information is synergistic
        },
        
        # Function 7: NAND (¬(a ∧ b))
        'nand': {
            'unique_a': 0.189,    # NAND: some unique information from A
            'unique_b': 0.189,    # NAND: some unique information from B
            'redundant': 0.311,   # NAND: shared information about output
            'synergistic': 0.311  # NAND: synergistic information
        },
        
        # Function 8: AND (a ∧ b)
        'and': {
            'unique_a': 0.0,      # AND gate: no unique information from A alone
            'unique_b': 0.0,      # AND gate: no unique information from B alone  
            'redundant': 0.311,   # Shared information about output
            'synergistic': 0.311  # Information only available when both inputs present
        },
        
        # Function 9: XNOR (¬(a ⊕ b)) = (a ↔ b)
        'xnor': {
            'unique_a': 0.0,      # XNOR: no unique information from A alone
            'unique_b': 0.0,      # XNOR: no unique information from B alone
            'redundant': 1.0,     # XNOR: all information is redundant
            'synergistic': 0.0    # XNOR: no synergistic information
        },
        
        # Function 10: Identity B (b)
        'id_b': {
            'unique_a': 0.0,      # Identity B: A provides no information
            'unique_b': 1.0,      # Identity B: all information comes from B
            'redundant': 0.0,     # Identity B: no shared information
            'synergistic': 0.0    # Identity B: no synergistic information
        },
        
        # Function 11: Implication A→B (¬a ∨ b)
        'imp_a_b': {
            'unique_a': 0.189,    # A implies B: some unique information from A
            'unique_b': 0.189,    # A implies B: some unique information from B
            'redundant': 0.311,   # A implies B: shared information
            'synergistic': 0.0    # A implies B: no synergistic information
        },
        
        # Function 12: Identity A (a)
        'id_a': {
            'unique_a': 1.0,      # Identity: all information comes from A
            'unique_b': 0.0,      # Identity: B provides no information
            'redundant': 0.0,     # Identity: no shared information
            'synergistic': 0.0    # Identity: no synergistic information
        },
        
        # Function 13: Implication B→A (a ∨ ¬b)
        'imp_b_a': {
            'unique_a': 0.189,    # B implies A: some unique information from A
            'unique_b': 0.189,    # B implies A: some unique information from B
            'redundant': 0.311,   # B implies A: shared information
            'synergistic': 0.0    # B implies A: no synergistic information
        },
        
        # Function 14: OR (a ∨ b)
        'or': {
            'unique_a': 0.189,    # OR gate: some unique information from A
            'unique_b': 0.189,    # OR gate: some unique information from B
            'redundant': 0.311,   # OR gate: shared information about output
            'synergistic': 0.0    # OR gate: no synergistic information
        },
        
        # Function 15: Constant 1
        'const_1': {
            'unique_a': 0.0,      # Constant: no information from A
            'unique_b': 0.0,      # Constant: no information from B
            'redundant': 0.0,     # Constant: no shared information
            'synergistic': 0.0    # Constant: no synergistic information
        }
    }


def save_synthetic_data(
    data: Dict[str, torch.Tensor], 
    output_dir: str,
    prefix: str = "synthetic_boolean"
) -> Dict[str, str]:
    """
    Save synthetic data to disk.
    
    Args:
        data: Dictionary of tensors to save
        output_dir: Directory to save files
        prefix: Prefix for filenames
    
    Returns:
        Dictionary mapping data keys to saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    for key, tensor in data.items():
        filename = f"{prefix}_{key}.pt"
        filepath = output_path / filename
        torch.save(tensor, filepath)
        saved_files[key] = str(filepath)
    
    return saved_files


def load_synthetic_data(
    data_dir: str,
    prefix: str = "synthetic_boolean",
    functions: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Load synthetic data from disk.
    
    Args:
        data_dir: Directory containing saved data files
        prefix: Prefix used when saving files
        functions: List of function names to load (default: input_a, input_b, and a few key functions)
    
    Returns:
        Dictionary of loaded tensors
    """
    if functions is None:
        functions = ['input_a', 'input_b', 'and', 'xor', 'id_a', 'or', 'const_0', 'const_1']
    
    data_path = Path(data_dir)
    loaded_data = {}
    
    for func_name in functions:
        filename = f"{prefix}_{func_name}.pt"
        filepath = data_path / filename
        if filepath.exists():
            loaded_data[func_name] = torch.load(filepath)
        else:
            print(f"Warning: Could not find {filepath}")
    
    return loaded_data


def create_synthetic_labels(
    data: torch.Tensor,
    num_clusters: int = 10,
    cluster_method: str = 'gmm'
) -> torch.Tensor:
    """
    Create synthetic labels for PID analysis using either GMM or K-means clustering.
    
    This function normalizes data and applies clustering to create either:
    - Soft labels (probability distributions) when using GMM
    - Hard labels (integers) when using K-means
    
    Args:
        data: Data tensor to cluster, shape [n_samples, feature_dim]
        num_clusters: Number of clusters to create
        cluster_method: Either 'gmm' or 'kmeans'
        
    Returns:
        For GMM: Tensor of probabilities of shape [n_samples, num_clusters]
        For kmeans: Tensor of integer labels of shape [n_samples]
    """
    # Convert to numpy for sklearn clustering
    data_np = data.cpu().numpy()
    
    # Normalize data to improve clustering
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    normalized_data = (data_np - mean) / std
    
    if cluster_method == 'kmeans':
        # Import here to avoid dependency if not used
        from sklearn.cluster import KMeans
        
        # Perform K-means clustering with multiple initializations for stability
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=3,  # Reduced from 10 to 3 for faster computation
            max_iter=300
        )
        labels = kmeans.fit_predict(normalized_data)
        
        # Return as PyTorch tensor with shape [n_samples]
        return torch.tensor(labels, dtype=torch.long)
    else:  # GMM
        # Import here to avoid dependency if not used
        from sklearn.mixture import GaussianMixture
        
        # Fit GMM and get probabilities
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type='diag',  # Use diagonal covariance for efficiency
            random_state=42,
            n_init=10,  # Reduced from 10 to 3 for faster computation
            max_iter=300
        )
        gmm.fit(normalized_data)
        probs = gmm.predict_proba(normalized_data)
        
        # Return as PyTorch tensor with shape [n_samples, num_clusters]
        return torch.tensor(probs, dtype=torch.float32)


def create_synthetic_labels_with_model(
    data: torch.Tensor,
    num_clusters: int = 10,
    cluster_method: str = 'gmm'
) -> tuple[torch.Tensor, object]:
    """
    Create synthetic labels and return both labels and the fitted clustering model.
    
    Args:
        data: Data tensor to cluster, shape [n_samples, feature_dim]
        num_clusters: Number of clusters to create
        cluster_method: Either 'gmm' or 'kmeans'
        
    Returns:
        Tuple of (labels, clustering_model)
        For GMM: (probabilities tensor, fitted GMM object)
        For kmeans: (integer labels tensor, fitted KMeans object)
    """
    # Convert to numpy for sklearn clustering
    data_np = data.cpu().numpy()
    
    # Normalize data to improve clustering
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    normalized_data = (data_np - mean) / std
    
    if cluster_method == 'kmeans':
        # Import here to avoid dependency if not used
        from sklearn.cluster import KMeans
        
        # Perform K-means clustering with multiple initializations for stability
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=3,  # Reduced from 10 to 3 for faster computation
            max_iter=300
        )
        labels = kmeans.fit_predict(normalized_data)
        
        # Return labels and the fitted model
        return torch.tensor(labels, dtype=torch.long), kmeans
    else:  # GMM
        # Import here to avoid dependency if not used
        from sklearn.mixture import GaussianMixture
        
        # Fit GMM and get probabilities
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type='diag',  # Use diagonal covariance for efficiency
            random_state=42,
            n_init=10,  # Reduced from 10 to 3 for faster computation
            max_iter=300
        )
        gmm.fit(normalized_data)
        probs = gmm.predict_proba(normalized_data)
        
        # Return probabilities and the fitted model
        return torch.tensor(probs, dtype=torch.float32), gmm


def save_clustering_model(clustering_model: object, filepath: str) -> None:
    """
    Save a fitted clustering model to disk using pickle.
    
    Args:
        clustering_model: Fitted sklearn clustering model (KMeans or GaussianMixture)
        filepath: Path to save the model
    """
    import pickle
    from pathlib import Path
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(clustering_model, f)


def load_clustering_model(filepath: str) -> object:
    """
    Load a fitted clustering model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Fitted sklearn clustering model
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def apply_clustering_model(clustering_model: object, data: torch.Tensor, cluster_method: str = 'gmm') -> torch.Tensor:
    """
    Apply a pre-trained clustering model to new data.
    
    Args:
        clustering_model: Fitted sklearn clustering model
        data: New data to assign to clusters
        cluster_method: Either 'gmm' or 'kmeans' (for return format)
        
    Returns:
        Cluster assignments in same format as create_synthetic_labels
    """
    # Convert to numpy and normalize the same way as training
    data_np = data.cpu().numpy()
    
    # Note: We should save normalization parameters with the model, but for now
    # we'll normalize again (this assumes the data distribution is similar)
    mean = np.mean(data_np, axis=0)
    std = np.std(data_np, axis=0) + 1e-8
    normalized_data = (data_np - mean) / std
    
    if cluster_method == 'kmeans':
        # For KMeans, use predict
        labels = clustering_model.predict(normalized_data)
        return torch.tensor(labels, dtype=torch.long)
    else:  # GMM
        # For GMM, use predict_proba
        probs = clustering_model.predict_proba(normalized_data)
        return torch.tensor(probs, dtype=torch.float32)


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic Boolean data...")
    
    # Generate data
    data = generate_boolean_data(n_samples=10000, functions=['and', 'xor', 'id_a', 'or'])
    
    # Print shapes and sample values
    for name, tensor in data.items():
        print(f"{name}: shape {tensor.shape}, sample values: {tensor[:5].flatten()}")
    
    # Show theoretical PID values
    print("\nTheoretical PID values:")
    pid_values = get_theoretical_pid_values()
    for func, values in pid_values.items():
        print(f"{func}: {values}")
    
    # Create datasets
    datasets = create_boolean_datasets(n_samples=10000, functions=['and', 'xor', 'id_a'])
    
    for func_name, splits in datasets.items():
        print(f"\n{func_name} datasets:")
        for split_name, dataset in splits.items():
            print(f"  {split_name}: {len(dataset)} samples") 