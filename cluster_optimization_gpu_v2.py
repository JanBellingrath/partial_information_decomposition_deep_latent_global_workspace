#!/usr/bin/env python3
"""
GPU-Accelerated Optimal Cluster Selection for PID Analysis in Multimodal Models.

This script extends cluster_optimization.py by using a GPU-accelerated implementation
of Gaussian Mixture Models (GMM-torch) for faster model fitting. It evaluates the optimal
number of clusters for latent spaces in different domains using BIC and AIC metrics.

Usage:
    python cluster_optimization_gpu_v2.py --find-latest-checkpoints --fusion-dir ./checkpoints/fusion \
        --domain-configs ./checkpoints/domain_v.ckpt ./checkpoints/domain_t.ckpt \
        --min-k 5 --max-k 100 --k-step 5 --max-configs 10

Authors: Original by dev team, refactored version
Date: April 2025
"""

import os
import sys
import argparse
import traceback
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# Numerical and ML libraries
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Memory

# Import the GPU-accelerated GMM implementation
try:
    from gmm_torch.gmm import GaussianMixture
except ImportError:
    print("Error: gmm-torch library not found.")
    print("Please install it with: pip install gmm-torch")
    sys.exit(1)

# Try importing wandb for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable experiment tracking.")

# Import key functions from analyze_pid
try:
    # Uncomment for normal operation
    from analyze_pid import (
        generate_samples_from_model,
        load_domain_modules,
        load_checkpoint,
        generate_samples_from_dataset
    )
except ImportError:
    print("Error: analyze_pid.py not found in the current path.")
    print("Required for loading domain modules and generating samples.")
    # Do not exit if we're just using test_data mode
    if '--test-data' not in sys.argv:
        sys.exit(1)

# Configure memory caching
memory = Memory(location='.cache', verbose=0)

# Set up module-level logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cluster_optimization_gpu')


# ================================================
# Core GMM Fitting Functions
# ================================================

def fit_gmm_with_metrics_gpu(
    data: np.ndarray, 
    k: int, 
    n_init: int = 10, 
    covariance_type: str = 'diag', 
    random_state: int = 42,
    device: str = "cuda",
    standardize: bool = True
) -> Dict:
    """
    Fit a GMM with a specific number of components and calculate metrics.
    This function uses the GPU-accelerated gmm-torch implementation.
    
    Args:
        data: Data to fit, shape [n_samples, n_features]
        k: Number of components
        n_init: Number of initializations
        covariance_type: Type of covariance parameter
        random_state: Random seed for reproducibility
        device: Device to use for computation ("cuda" or "cpu")
        standardize: Whether to standardize features before fitting (mean=0, std=1)
    
    Returns:
        Dictionary of metrics including:
            - bic: Bayesian Information Criterion value
            - aic: Akaike Information Criterion value
            - log_likelihood: Log likelihood of the data
            - labels: Cluster assignments
            - gmm: Fitted GMM model
    
    Raises:
        RuntimeError: If all GMM fitting attempts fail
    """
    result = {}
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Set the device
    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {torch_device}")
    
    # Standardize the data if requested (zero mean, unit variance)
    if standardize:
        logger.info("Standardizing features before GMM fitting")
        # Calculate mean and std on CPU to avoid potential GPU memory issues
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        
        # Handle features with zero variance to avoid division by zero
        data_std[data_std < 1e-10] = 1.0
        
        # Standardize the data
        standardized_data = (data - data_mean) / data_std
        
        # Store standardization parameters for later reference
        result['standardization'] = {
            'mean': data_mean,
            'std': data_std
        }
        
        # Use standardized data for subsequent operations
        data_for_tensor = standardized_data
        logger.info(f"Data standardized: mean range [{data_mean.min():.4f}, {data_mean.max():.4f}], std range [{data_std.min():.4f}, {data_std.max():.4f}]")
    else:
        data_for_tensor = data
    
    # Convert numpy data to PyTorch tensor and move to the correct device
    data_tensor = torch.tensor(data_for_tensor, dtype=torch.float16, device=torch_device)
    
    # Fit GMM using gmm-torch - create it on the same device
    best_gmm = None
    best_log_likelihood = -np.inf
    
    for attempt in range(n_init):
        try:
            # Create a new GMM instance for each attempt
            gmm = GaussianMixture(
                n_components=k,
                n_features=data.shape[1],
                covariance_type=covariance_type,
                eps=1e-3,
                init_params="random"
            ).to(torch_device)
            
            # Initialize parameters
            gmm._init_params()
            
            # Make sure all parameters are on the same device
            for param in gmm.parameters():
                param.data = param.data.to(torch_device)
            
            # Fit with a reasonable number of iterations
            gmm.fit(data_tensor, n_iter=100, delta=1e-3, warm_start=False)
            
            # Keep the best model
            if gmm.log_likelihood > best_log_likelihood:
                best_log_likelihood = gmm.log_likelihood
                best_gmm = gmm
                logger.debug(f"Found better model in attempt {attempt+1} with log likelihood: {best_log_likelihood}")
        except Exception as e:
            # delete any large objects
            del gmm, data_tensor
            # release cached CUDA memory to the OS
            torch.cuda.empty_cache()
            # collect any lingering CPU‐side references
            import gc; gc.collect()
            logger.warning(f"GMM fitting attempt {attempt+1} failed: {e}")
            continue
    
    # If none of the attempts succeeded, raise an error
    if best_gmm is None:
        raise RuntimeError("All GMM fitting attempts failed")
    
    # Use the best model
    gmm = best_gmm
    
    # Get cluster assignments and probabilities
    labels = gmm.predict(data_tensor)
    log_probs = gmm.score_samples(data_tensor)
    
    # Move tensors back to CPU for compatibility
    labels_cpu = labels.cpu()
    log_probs_cpu = log_probs.cpu()
    
    # Convert to numpy for metrics calculation
    labels_np = labels_cpu.numpy()
    
    # Calculate metrics using a consistent approach
    # Calculate total log likelihood directly from per-sample log probabilities
    total_log_likelihood = log_probs.sum().item()
    
    # Count parameters: means + covariances + weights - 1
    n_samples = data.shape[0]
    n_dimensions = data.shape[1]
    
    if covariance_type == 'full':
        # For full covariance, each component has d*(d+1)/2 parameters (symmetric matrix)
        n_parameters = k * (n_dimensions + (n_dimensions * (n_dimensions + 1)) // 2 + 1) - 1
    elif covariance_type == 'diag':
        n_parameters = k * (n_dimensions + n_dimensions + 1) - 1
    else:  # 'spherical'
        n_parameters = k * (n_dimensions + 1 + 1) - 1
    
    # Calculate AIC: -2 * log_likelihood + 2 * n_parameters
    aic = -2 * total_log_likelihood + 2 * n_parameters
    
    # Calculate BIC: -2 * log_likelihood + log(n) * n_parameters
    bic = -2 * total_log_likelihood + np.log(n_samples) * n_parameters
    
    # Store metrics
    result['bic'] = bic
    result['aic'] = aic
    result['log_likelihood'] = total_log_likelihood / n_samples  # Store average for backwards compatibility
    result['total_log_likelihood'] = total_log_likelihood  # Store total as well
    result['labels'] = labels_np
    result['gmm'] = gmm
    result['random_state'] = random_state
    
    # Clean up GPU resources
    gmm = gmm.cpu()  # Move to CPU if needed
    del best_gmm, data_tensor, labels, log_probs, labels_cpu, log_probs_cpu
    torch.cuda.empty_cache()
    import gc; gc.collect()
            
    return result


def fit_gmm_range_gpu(
    data: np.ndarray,
    k_range: List[int],
    n_init: int = 10,
    covariance_type: str = 'diag',
    random_state: int = 42,
    device: str = "cuda",
    n_runs_per_k: int = 3,
    standardize: bool = True
) -> Dict:
    """
    Fit GMMs with varying numbers of components and evaluate using multiple metrics.
    Uses GPU acceleration for the fitting process.
    
    Args:
        data: Data to fit, shape [n_samples, n_features]
        k_range: Range of number of components to try
        n_init: Number of initializations for each GMM
        covariance_type: Type of covariance parameter
        random_state: Base random seed for reproducibility
        device: Device to use for computation
        n_runs_per_k: Number of runs with different random seeds for each k value
        standardize: Whether to standardize features before fitting
    
    Returns:
        Dictionary with metrics for each k value
    """
    # Initialize result containers
    results = {
        'k': k_range,
        'bic': [],
        'bic_std': [],
        'aic': [],
        'aic_std': [],
        'log_likelihood': [],
        'log_likelihood_std': [],
        'best_labels': [],  # Store only labels from best model instead of all models
        'all_runs': {}  # Store results from all runs for each k
    }
    
    # For each k, store all metrics from multiple runs
    for k in k_range:
        results['all_runs'][k] = {
            'bic': [],
            'aic': [],
            'log_likelihood': [],
            'random_states': []
            # Not storing models or labels for every run to save memory
        }
    
    # Set base random seed
    base_seed = random_state
    
    # Convert data to tensor once, rather than in each call
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Process each k value sequentially (on GPU)
    logger.info(f"Fitting GMMs on {device} with {n_runs_per_k} runs per k value...")
    for k in tqdm(k_range, desc='Fitting GMMs'):
        # For each k, perform multiple runs with different random seeds
        # Keep track of best model metrics only
        best_bic = float('inf')
        best_labels = None
        best_run_idx = -1
        
        for run in range(n_runs_per_k):
            # Generate a new random seed for this run
            curr_seed = base_seed + run * 1000 + k
            
            try:
                logger.info(f"Fitting GMM with k={k}, run {run+1}/{n_runs_per_k}, seed {curr_seed}")
                result = fit_gmm_with_metrics_gpu(
                    data=data,
                    k=k,
                    n_init=n_init, 
                    covariance_type=covariance_type, 
                    random_state=curr_seed,
                    device=device,
                    standardize=standardize
                )
                
                # Track the best model based on BIC
                if result['bic'] < best_bic:
                    best_bic = result['bic']
                    best_labels = result['labels']
                    best_run_idx = run
                
                # Store metrics from this run
                results['all_runs'][k]['bic'].append(result['bic'])
                results['all_runs'][k]['aic'].append(result['aic'])
                results['all_runs'][k]['log_likelihood'].append(result['log_likelihood'])
                results['all_runs'][k]['random_states'].append(curr_seed)
                
                # Explicitly clean up GPU resources
                if 'gmm' in result:
                    del result['gmm']
                del result
                torch.cuda.empty_cache()
                import gc; gc.collect()
                
            except Exception as e:
                logger.warning(f"Error fitting GMM with k={k}, run {run+1}: {e}")
                traceback.print_exc()
                # Add placeholder values to maintain the sequence for this run
                results['all_runs'][k]['bic'].append(float('inf'))
                results['all_runs'][k]['aic'].append(float('inf'))
                results['all_runs'][k]['log_likelihood'].append(float('-inf'))
                results['all_runs'][k]['random_states'].append(curr_seed)
                
                # Ensure GPU is cleared
                torch.cuda.empty_cache()
                import gc; gc.collect()
        
        # Calculate mean and std from all runs for this k
        bic_values = results['all_runs'][k]['bic']
        aic_values = results['all_runs'][k]['aic']
        ll_values = results['all_runs'][k]['log_likelihood']
        
        # Compute metrics averaging across runs
        valid_bic = [v for v in bic_values if v != float('inf')]
        valid_aic = [v for v in aic_values if v != float('inf')]
        valid_ll = [v for v in ll_values if v != float('-inf')]
        
        # Use mean of valid runs, or inf if all runs failed
        results['bic'].append(np.mean(valid_bic) if valid_bic else float('inf'))
        results['bic_std'].append(np.std(valid_bic) if len(valid_bic) > 1 else 0)
        
        results['aic'].append(np.mean(valid_aic) if valid_aic else float('inf'))
        results['aic_std'].append(np.std(valid_aic) if len(valid_aic) > 1 else 0)
        
        results['log_likelihood'].append(np.mean(valid_ll) if valid_ll else float('-inf'))
        results['log_likelihood_std'].append(np.std(valid_ll) if len(valid_ll) > 1 else 0)
        
        # Only store labels for the best model from all runs to save memory
        results['best_labels'].append(best_labels if best_labels is not None else np.zeros(data.shape[0]))
    
    return results


def find_optimal_k_gpu(metrics: Dict) -> Dict:
    """
    Find the optimal number of clusters based on BIC and AIC metrics.
    
    Args:
        metrics: Dictionary of metrics for different k values
        
    Returns:
        Dictionary with optimal k values for each metric
    """
    results = {}
    
    # BIC: Lower is better
    min_bic_idx = np.argmin(metrics['bic'])
    results['bic'] = metrics['k'][min_bic_idx]
    results['min_bic_value'] = metrics['bic'][min_bic_idx]
    results['bic_std'] = metrics['bic_std'][min_bic_idx]
    
    # AIC: Lower is better
    min_aic_idx = np.argmin(metrics['aic'])
    results['aic'] = metrics['k'][min_aic_idx]
    results['min_aic_value'] = metrics['aic'][min_aic_idx]
    results['aic_std'] = metrics['aic_std'][min_aic_idx]
    
    return results


def visualize_metrics_gpu(
    metrics: Dict,
    domain_name: str,
    output_dir: str,
    wandb_run = None
) -> Dict:
    """
    Create visualizations for model comparison metrics.
    
    Args:
        metrics: Dictionary of metrics
        domain_name: Name of the domain being analyzed
        output_dir: Directory to save visualizations
        wandb_run: Optional wandb run for logging
        
    Returns:
        Dictionary of file paths to generated visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Plot BIC and AIC with standard deviation error bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot BIC with error bars
    ax.errorbar(
        metrics['k'], 
        metrics['bic'], 
        yerr=metrics['bic_std'],
        fmt='o-', 
        label='BIC', 
        color='blue',
        capsize=5, 
        alpha=0.7
    )
    
    # Plot AIC with error bars
    ax.errorbar(
        metrics['k'], 
        metrics['aic'], 
        yerr=metrics['aic_std'],
        fmt='s-', 
        label='AIC', 
        color='red',
        capsize=5, 
        alpha=0.7
    )
    
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Information Criterion')
    ax.set_title(f'BIC and AIC for {domain_name} Domain (Averaged over multiple runs)')
    ax.grid(alpha=0.3)
    
    # Highlight the minimum BIC and AIC
    min_bic_idx = np.argmin(metrics['bic'])
    min_aic_idx = np.argmin(metrics['aic'])
    
    ax.plot(
        metrics['k'][min_bic_idx], 
        metrics['bic'][min_bic_idx], 
        'o', 
        markersize=10, 
        fillstyle='none', 
        color='blue', 
        label=f'Min BIC at k={metrics["k"][min_bic_idx]} (±{metrics["bic_std"][min_bic_idx]:.2f})'
    )
    
    ax.plot(
        metrics['k'][min_aic_idx], 
        metrics['aic'][min_aic_idx], 
        's', 
        markersize=10, 
        fillstyle='none', 
        color='red', 
        label=f'Min AIC at k={metrics["k"][min_aic_idx]} (±{metrics["aic_std"][min_aic_idx]:.2f})'
    )
    
    ax.legend()
    
    # Add annotation about number of runs
    if 'all_runs' in metrics:
        k_example = metrics['k'][0]
        n_runs = len(metrics['all_runs'][k_example]['bic'])
        ax.annotate(
            f'Values averaged over {n_runs} runs per k value', 
            xy=(0.5, 0.02), 
            xycoords='figure fraction', 
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Save plot
    info_criterion_path = os.path.join(output_dir, f'{domain_name}_info_criterion.png')
    plt.savefig(info_criterion_path, dpi=300, bbox_inches='tight')
    plot_files['info_criterion'] = info_criterion_path
    
    # Log to wandb if available
    if wandb_run is not None:
        try:
            wandb.log({f"{domain_name}/info_criterion": wandb.Image(info_criterion_path)})
            
            # Also log the actual values as a line chart
            if HAS_WANDB:
                bic_data = [[k, bic, std] for k, bic, std in zip(metrics['k'], metrics['bic'], metrics['bic_std'])]
                aic_data = [[k, aic, std] for k, aic, std in zip(metrics['k'], metrics['aic'], metrics['aic_std'])]
                
                # Create a wandb Table for detailed metrics
                table = wandb.Table(columns=["k", "BIC", "BIC_std", "AIC", "AIC_std"])
                for i, k in enumerate(metrics['k']):
                    table.add_data(
                        k, 
                        metrics['bic'][i], 
                        metrics['bic_std'][i],
                        metrics['aic'][i],
                        metrics['aic_std'][i]
                    )
                wandb.log({f"{domain_name}/metrics_table": table})
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")
    
    plt.close(fig)
    
    # Plot cluster distribution for the BIC-optimal model
    if 'best_labels' in metrics and metrics['best_labels'][min_bic_idx] is not None:
        # Get cluster assignments for the optimal model
        labels = metrics['best_labels'][min_bic_idx]
        
        # Count samples per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by cluster index
        sorted_idx = np.argsort(unique_labels)
        unique_labels = unique_labels[sorted_idx]
        counts = counts[sorted_idx]
        
        # Plot distribution
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(unique_labels, counts)
        ax.set_xlabel('Cluster Index')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Cluster Distribution (k={metrics["k"][min_bic_idx]}) for {domain_name}')
        ax.grid(alpha=0.3)
        
        # Save plot
        cluster_dist_path = os.path.join(output_dir, f'{domain_name}_cluster_distribution.png')
        plt.savefig(cluster_dist_path, dpi=300, bbox_inches='tight')
        plot_files['cluster_distribution'] = cluster_dist_path
        
        # Log to wandb if available
        if wandb_run is not None:
            try:
                wandb.log({f"{domain_name}/cluster_distribution": wandb.Image(cluster_dist_path)})
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
        
        plt.close(fig)
    
    return plot_files


def generate_domain_report_gpu(
    domain_name: str,
    metrics: Dict,
    optimal_k: Dict,
    shape: Tuple,
    standardized: bool = True
) -> str:
    """
    Generate a detailed report for a domain's cluster analysis.
    
    Args:
        domain_name: Name of the domain
        metrics: Dictionary of metrics
        optimal_k: Dictionary of optimal k values
        shape: Shape of the data
        standardized: Whether data was standardized before GMM fitting
        
    Returns:
        Report as a string
    """
    report = f"=== Cluster Optimization Report for {domain_name} ===\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += f"Data shape: {shape[0]} samples, {shape[1]} dimensions\n"
    report += f"Data preprocessing: {'Standardized (μ=0, σ=1)' if standardized else 'Raw (no standardization)'}\n\n"
    
    # Get number of runs per k
    n_runs_per_k = 1
    if 'all_runs' in metrics and metrics['k']:
        k_example = metrics['k'][0]
        if k_example in metrics['all_runs']:
            n_runs_per_k = len(metrics['all_runs'][k_example]['bic'])
    
    report += f"Analysis performed with {n_runs_per_k} runs per k value\n\n"
    
    report += "Optimal k values for BIC and AIC:\n"
    report += f"  - BIC minimum: k={optimal_k['bic']} (value: {optimal_k['min_bic_value']:.2f} ± {optimal_k['bic_std']:.2f})\n"
    report += f"  - AIC minimum: k={optimal_k['aic']} (value: {optimal_k['min_aic_value']:.2f} ± {optimal_k['aic_std']:.2f})\n\n"
    
    report += "Detailed metrics:\n"
    for i, k in enumerate(metrics['k']):
        report += f"\nK = {k}:\n"
        report += f"  - BIC: {metrics['bic'][i]:.2f} ± {metrics['bic_std'][i]:.2f}\n"
        report += f"  - AIC: {metrics['aic'][i]:.2f} ± {metrics['aic_std'][i]:.2f}\n"
        report += f"  - Log Likelihood: {metrics['log_likelihood'][i]:.2f} ± {metrics['log_likelihood_std'][i]:.2f}\n"
        
        # Add individual run details if available
        if 'all_runs' in metrics and k in metrics['all_runs']:
            report += f"  - Individual run details:\n"
            for run_idx, (bic, aic, ll) in enumerate(zip(
                metrics['all_runs'][k]['bic'],
                metrics['all_runs'][k]['aic'],
                metrics['all_runs'][k]['log_likelihood']
            )):
                rs = metrics['all_runs'][k]['random_states'][run_idx] if 'random_states' in metrics['all_runs'][k] else 'N/A'
                report += f"    - Run {run_idx+1} (seed {rs}): BIC={bic:.2f}, AIC={aic:.2f}, LL={ll:.2f}\n"
    
    return report


def find_latest_model_checkpoints(base_dir: str, max_configs: Optional[int] = None) -> List[str]:
    """
    Find the latest model checkpoint for each configuration in the base directory.
    
    Args:
        base_dir: Base directory containing configuration subdirectories
        max_configs: Maximum number of most recent configurations to include
    
    Returns:
        List of paths to the latest model checkpoints
    """
    import re
    
    latest_checkpoints = []
    
    # Pattern to extract epoch from filename - flexible to handle various formats
    epoch_pattern = re.compile(r'model_epoch_(\d+)')
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory {base_dir} does not exist")
        return latest_checkpoints
    
    # Get all config directories sorted by modification time (most recent first)
    config_dirs = sorted(
        [d for d in Path(base_dir).glob("config_*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True  # Most recently modified first
    )
    
    # If max_configs is specified, limit to only the most recent configs
    if max_configs is not None and max_configs > 0 and len(config_dirs) > max_configs:
        config_dirs = config_dirs[:max_configs]
        logger.info(f"Limiting analysis to the {max_configs} most recent configuration directories")
    
    # Process each config directory
    for config_dir in config_dirs:
        # Find all model checkpoint files in this config directory
        checkpoints = list(config_dir.glob("model_epoch_*.pt"))
        
        if not checkpoints:
            logger.warning(f"No checkpoints found in {config_dir}")
            continue
        
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
            logger.info(f"Found latest checkpoint for {config_dir.name}: {latest_checkpoint.name} (epoch {latest_epoch})")
            latest_checkpoints.append(str(latest_checkpoint))
    
    return latest_checkpoints 


# ================================================
# Main Analysis Functions
# ================================================

def analyze_domain_clusters_gpu(
    model_paths: List[str],
    domain_modules: Dict,
    domain_names: List[str],
    output_dir: str,
    k_range: List[int] = list(range(2, 51, 2)),
    n_samples: int = 10000,
    batch_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_wandb: bool = False,
    wandb_project: str = "cluster-optimization",
    wandb_entity: Optional[str] = None,
    find_latest: bool = False,
    fusion_dir: Optional[str] = None,
    max_configs: Optional[int] = None,
    data_module = None,
    dataset_split: str = "test",
    use_gw_processed: bool = False,
    runs_per_k: int = 3,
    standardize: bool = True
) -> Dict:
    """
    GPU-accelerated analysis of optimal number of clusters for each domain.
    
    Args:
        model_paths: List of model checkpoint paths
        domain_modules: Dictionary of domain modules
        domain_names: List of domain names to analyze
        output_dir: Directory to save results
        k_range: Range of cluster numbers to try
        n_samples: Number of samples to generate per model
        batch_size: Batch size for generation
        device: Device to run on
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
        find_latest: Whether to find the latest checkpoints in the fusion directory
        fusion_dir: Directory containing fusion model configurations
        max_configs: Maximum number of recent configs to include
        data_module: Optional data module for generating samples from real dataset
        dataset_split: Dataset split to use when using data_module ("train", "val", or "test")
        use_gw_processed: Whether to use GW-processed latents (through encode & decode) instead of original
        runs_per_k: Number of runs to perform for each k value with different random seeds
        standardize: Whether to standardize features before fitting GMMs
        
    Returns:
        Dictionary with results for each domain
    """
    # If find_latest is True, override model_paths with latest checkpoints
    if find_latest and fusion_dir is not None:
        logger.info(f"Finding latest checkpoints in {fusion_dir}")
        model_paths = find_latest_model_checkpoints(fusion_dir, max_configs)
        if not model_paths:
            logger.warning("No checkpoints found. Using provided model paths.")
        else:
            logger.info(f"Found {len(model_paths)} latest checkpoints")
            for path in model_paths:
                logger.info(f"  - {path}")
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    results_dir = os.path.join(output_dir, f"cluster_optimization_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize wandb if requested
    wandb_run = None
    if HAS_WANDB and use_wandb:
        try:
            wandb_config = {
                "domain_names": domain_names,
                "n_models": len(model_paths),
                "n_samples": n_samples,
                "k_range": k_range,
                "data_source": "dataset" if data_module else "synthetic",
                "dataset_split": dataset_split if data_module else "N/A",
                "use_gw_processed": use_gw_processed,
                "runs_per_k": runs_per_k,
                "standardize": standardize
            }
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"cluster_optimization_{timestamp}",
                config=wandb_config,
            )
            logger.info(f"Initialized wandb run: {wandb_run.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_run = None
    
    # Ensure we have at least one model path
    if not model_paths:
        logger.error("No model paths found. Please check your configuration.")
        return {}
        
    # Results container
    results = {domain: {} for domain in domain_names}
    
    # Collect samples from all models
    all_samples = {domain: [] for domain in domain_names}
    
    logger.info(f"Collecting samples from {len(model_paths)} models...")
    
    for model_path in tqdm(model_paths, desc="Processing Models"):
        # Generate samples
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file does not exist: {model_path}")
                continue
                
            # Load model and generate samples
            logger.info(f"Loading model from {model_path}")
            try:
                model = load_checkpoint(model_path, domain_modules, device)
            except Exception as e:
                logger.error(f"Failed to load checkpoint {model_path}: {e}")
                traceback.print_exc()
                continue
            
            # Generate samples - use dataset if provided, otherwise generate synthetic
            logger.info(f"Generating {n_samples} samples")
            try:
                if data_module:
                    logger.info(f"Using real data from {dataset_split} split")
                    samples = generate_samples_from_dataset(
                        model=model,
                        data_module=data_module,
                        domain_names=domain_names,
                        split=dataset_split,
                        n_samples=n_samples,
                        batch_size=batch_size,
                        device=device
                    )
                    # Adjust keys to match what's expected by later code
                    # Choose either original latents or GW-processed latents
                    domain_samples = {}
                    for domain in domain_names:
                        if use_gw_processed:
                            # Use GW-processed latents (encoded & decoded)
                            if f"{domain}_decoded" in samples:
                                domain_samples[domain] = samples[f"{domain}_decoded"]
                                logger.info(f"Using GW-processed latents for {domain}")
                            else:
                                logger.warning(f"{domain}_decoded not found in samples")
                        else:
                            # Use original latents
                            if f"{domain}_orig" in samples:
                                domain_samples[domain] = samples[f"{domain}_orig"]
                                logger.info(f"Using original latents for {domain}")
                            else:
                                logger.warning(f"{domain}_orig not found in samples")
                    samples = domain_samples
                else:
                    logger.info("Generating synthetic samples")
                    samples = generate_samples_from_model(
                        model=model,
                        domain_names=domain_names,
                        n_samples=n_samples,
                        batch_size=batch_size,
                        device=device
                    )
            except Exception as e:
                logger.error(f"Failed to generate samples from {model_path}: {e}")
                traceback.print_exc()
                continue
            
            # Collect samples for each domain
            for domain in domain_names:
                if domain in samples:
                    all_samples[domain].append(samples[domain])
                else:
                    logger.warning(f"Domain {domain} not found in samples from {model_path}")
        except Exception as e:
            logger.error(f"Error processing model {model_path}: {e}")
            traceback.print_exc()
    
    # Print debug info about collected samples
    for domain in domain_names:
        sample_count = len(all_samples[domain])
        logger.info(f"Sample collection for {domain}: collected from {sample_count}/{len(model_paths)} models")
        if sample_count == 0:
            logger.warning(f"No samples collected for domain: {domain}")
    
    # Check if we have any samples
    if all(len(samples) == 0 for samples in all_samples.values()):
        logger.error("No samples collected from any model. Aborting analysis.")
        return {}
        
    # Analyze each domain separately
    for domain in domain_names:
        if not all_samples[domain]:
            continue
            
        # Concatenate samples from all models
        domain_data = torch.cat(all_samples[domain], dim=0)
        
        # Special handling for v_latents dimensions
        if domain == 'v_latents' and domain_data.dim() > 2:
            original_shape = domain_data.shape
            # Take only the first component (mean) along dimension 1
            domain_data = domain_data[:, 0, :]
            logger.info(f"Found 3D tensor for {domain}: {original_shape} -> {domain_data.shape}")
        
        # Convert to numpy for sklearn and clear the original tensor to save memory
        domain_data_np = domain_data.cpu().numpy()
        del domain_data
        torch.cuda.empty_cache()
        
        logger.info(f"Analyzing domain: {domain} with shape: {domain_data_np.shape}")
        
        try:
            # Fit GMMs and collect metrics using GPU-accelerated function
            metrics = fit_gmm_range_gpu(
                data=domain_data_np,
                k_range=k_range,
                n_init=10,
                covariance_type='diag',
                random_state=42,
                device=device,
                n_runs_per_k=runs_per_k,
                standardize=standardize
            )
            
            # Find optimal k values
            optimal_k = find_optimal_k_gpu(metrics)
            
            # Visualize metrics
            plot_files = visualize_metrics_gpu(
                metrics=metrics,
                domain_name=domain,
                output_dir=os.path.join(results_dir, domain),
                wandb_run=wandb_run
            )
            
            # Store results (removing large data structures to save memory)
            results[domain] = {
                'optimal_k': optimal_k,
                'metrics': {
                    'k': metrics['k'],
                    'bic': metrics['bic'],
                    'aic': metrics['aic'],
                    'log_likelihood': metrics['log_likelihood'],
                },
                'plot_files': plot_files,
                'sample_shape': domain_data_np.shape,
            }
            
            # Clean up the metrics dictionary to free memory
            if 'all_runs' in metrics:
                del metrics['all_runs']
            if 'best_labels' in metrics:
                del metrics['best_labels']
            
            # Generate detailed report
            report = generate_domain_report_gpu(
                domain_name=domain,
                metrics=metrics,
                optimal_k=optimal_k,
                shape=domain_data_np.shape,
                standardized=standardize
            )
            
            # Save report
            report_path = os.path.join(results_dir, domain, f"{domain}_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            
            # Log report to wandb
            if HAS_WANDB and wandb_run is not None:
                wandb.log({f"{domain}/report": wandb.Table(
                    columns=["Metric", "Value"],
                    data=[[k, v] for k, v in optimal_k.items()]
                )})
                
            logger.info(f"Completed analysis for domain: {domain}")
            
            # Clean up metrics to free memory
            del metrics
            
        except Exception as e:
            logger.error(f"Error analyzing domain {domain}: {e}")
            traceback.print_exc()
        
        # Clean up numpy array to free memory
        del domain_data_np
    
    # Save overall results
    results_summary = {
        'timestamp': timestamp,
        'domains': {}
    }
    
    # Process each domain's results to ensure they're JSON serializable
    for domain in results:
        if domain in results and 'optimal_k' in results[domain]:
            # Only include BIC and AIC
            optimal_k = {
                'bic': int(results[domain]['optimal_k']['bic']) if isinstance(results[domain]['optimal_k']['bic'], (np.integer, int)) else float(results[domain]['optimal_k']['bic']),
                'aic': int(results[domain]['optimal_k']['aic']) if isinstance(results[domain]['optimal_k']['aic'], (np.integer, int)) else float(results[domain]['optimal_k']['aic']),
                'min_bic_value': float(results[domain]['optimal_k']['min_bic_value']),
                'min_aic_value': float(results[domain]['optimal_k']['min_aic_value'])
            }
                    
            # Get shape info as a list of integers
            if 'sample_shape' in results[domain]:
                shape = [int(dim) for dim in results[domain]['sample_shape']]
            else:
                shape = None
                
            results_summary['domains'][domain] = {
                'optimal_k': optimal_k,
                'sample_shape': shape
            }
    
    # Save as JSON
    summary_path = os.path.join(results_dir, f"cluster_optimization_summary_{timestamp}.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=4)
        logger.info(f"Saved summary to {summary_path}")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
        traceback.print_exc()
    
    # Generate final recommendation
    min_bic_values = {domain: results[domain]['optimal_k']['bic'] 
                     for domain in results 
                     if domain in results and 'optimal_k' in results[domain]}
    
    min_aic_values = {domain: results[domain]['optimal_k']['aic'] 
                     for domain in results 
                     if domain in results and 'optimal_k' in results[domain]}
    
    logger.info("\n=== Cluster Optimization Complete ===")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("\nMinimum BIC values (k with lowest BIC):")
    for domain, k in min_bic_values.items():
        logger.info(f"  - {domain}: k={k}")
    
    logger.info("\nMinimum AIC values (k with lowest AIC):")
    for domain, k in min_aic_values.items():
        logger.info(f"  - {domain}: k={k}")
    
    # Log final table to wandb
    if HAS_WANDB and wandb_run is not None:
        try:
            wandb.log({"final_results": wandb.Table(
                columns=["Domain", "BIC min (k)", "AIC min (k)"],
                data=[
                    [
                        domain, 
                        results[domain]['optimal_k']['bic'],
                        results[domain]['optimal_k']['aic']
                    ] for domain in results if domain in results and 'optimal_k' in results[domain]
                ]
            )})
            
            # Finish wandb run
            wandb_run.finish()
        except Exception as e:
            logger.error(f"Error logging to wandb: {e}")
    
    return results


def run_with_test_data(
    test_data_file: str,
    output_dir: str,
    domain_name: str = "test_domain",
    k_range: List[int] = list(range(2, 51, 2)),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_wandb: bool = False,
    wandb_project: str = "cluster-optimization",
    wandb_entity: Optional[str] = None,
    runs_per_k: int = 3,
    standardize: bool = True
) -> Dict:
    """
    Run clustering optimization using test data directly instead of loading models.
    This function is for testing purposes only.
    
    Args:
        test_data_file: Path to test data file (.npy format)
        output_dir: Directory to save results
        domain_name: Name of the domain for reporting
        k_range: Range of cluster numbers to try
        device: Device to use for computation
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
        runs_per_k: Number of runs to perform for each k value with different random seeds
        standardize: Whether to standardize features before GMM fitting
        
    Returns:
        Results dictionary
    """
    from datetime import datetime
    
    logger.info(f"Loading test data from {test_data_file}")
    try:
        data = np.load(test_data_file)
        logger.info(f"Data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    results_dir = os.path.join(output_dir, f"cluster_optimization_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize wandb if requested
    wandb_run = None
    if use_wandb and HAS_WANDB:
        try:
            wandb_config = {
                "domain_name": domain_name,
                "data_file": test_data_file,
                "data_shape": data.shape,
                "k_range": k_range,
                "runs_per_k": runs_per_k,
                "standardize": standardize
            }
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"cluster_optimization_{timestamp}",
                config=wandb_config,
            )
            logger.info(f"Initialized wandb run: {wandb_run.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_run = None
    
    # Convert data to torch tensor for later use
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    logger.info(f"Fitting GMMs for domain: {domain_name} with shape: {data.shape}")
    
    try:
        # Fit GMMs and collect metrics
        metrics = fit_gmm_range_gpu(
            data=data,
            k_range=k_range,
            n_init=10,
            covariance_type='diag',
            random_state=42,  # Use fixed random state for reproducibility
            device=device,
            n_runs_per_k=runs_per_k,
            standardize=standardize
        )
        
        # Find optimal k values
        optimal_k = find_optimal_k_gpu(metrics)
        
        # Create domain directory
        domain_dir = os.path.join(results_dir, domain_name)
        os.makedirs(domain_dir, exist_ok=True)
        
        # Visualize metrics
        plot_files = visualize_metrics_gpu(
            metrics=metrics,
            domain_name=domain_name,
            output_dir=domain_dir,
            wandb_run=wandb_run
        )
        
        # Generate report
        report = generate_domain_report_gpu(
            domain_name=domain_name,
            metrics=metrics,
            optimal_k=optimal_k,
            shape=data.shape,
            standardized=standardize
        )
        
        # Save report
        report_path = os.path.join(domain_dir, f"{domain_name}_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Save results summary
        results_summary = {
            "timestamp": timestamp,
            "domain": domain_name,
            "optimal_k": {
                "bic": int(optimal_k['bic']) if isinstance(optimal_k['bic'], (np.integer, int)) else float(optimal_k['bic']),
                "aic": int(optimal_k['aic']) if isinstance(optimal_k['aic'], (np.integer, int)) else float(optimal_k['aic']),
                "min_bic_value": float(optimal_k['min_bic_value']),
                "min_aic_value": float(optimal_k['min_aic_value']),
                "bic_std": float(optimal_k['bic_std']),
                "aic_std": float(optimal_k['aic_std'])
            },
            "data_shape": list(data.shape),
            "runs_per_k": runs_per_k,
            "standardize": standardize
        }
        
        summary_path = os.path.join(results_dir, f"cluster_optimization_summary_{timestamp}.json")
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=4)
        
        # Log to wandb if available
        if wandb_run is not None:
            try:
                # Log optimum values
                for k, v in optimal_k.items():
                    if v is not None:
                        wandb.log({f"optimal_{k}": v})
                
                # Log plots
                for name, path in plot_files.items():
                    wandb.log({f"{domain_name}/{name}": wandb.Image(path)})
                
                # Log report as text - use Table instead of Text
                with open(report_path, "r") as f:
                    report_content = f.read()
                    # Create a table with report lines
                    report_table = wandb.Table(columns=["Report"])
                    for line in report_content.split('\n'):
                        report_table.add_data(line)
                    wandb.log({f"{domain_name}/report": report_table})
                
                # Finish wandb run
                wandb_run.finish()
            except Exception as e:
                logger.warning(f"Error logging to wandb: {e}")
        
        logger.info("\n=== Cluster Optimization Complete ===")
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"\nBIC minimum for {domain_name}: k={optimal_k['bic']} (value: {optimal_k['min_bic_value']:.2f} ± {optimal_k['bic_std']:.2f})")
        logger.info(f"AIC minimum for {domain_name}: k={optimal_k['aic']} (value: {optimal_k['min_aic_value']:.2f} ± {optimal_k['aic_std']:.2f})")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        traceback.print_exc()
        return None


# ================================================
# Command-Line Interface
# ================================================

def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Optimize number of clusters for PID analysis with GPU acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--checkpoint-dir",
        type=str,
        required=False,
        help="Directory containing model checkpoints"
    )
    model_group.add_argument(
        "--domain-configs",
        type=str,
        nargs="+",
        default=["./checkpoints/domain_v.ckpt", "./checkpoints/domain_t.ckpt"],
        help="Paths to domain module checkpoints"
    )
    model_group.add_argument(
        "--max-models",
        type=int,
        default=5,
        help="Maximum number of models to analyze"
    )
    model_group.add_argument(
        "--find-latest-checkpoints",
        action="store_true",
        help="Find the latest checkpoint for each configuration in the fusion directory"
    )
    model_group.add_argument(
        "--fusion-dir",
        type=str,
        default="./checkpoints/fusion",
        help="Directory containing fusion model configurations"
    )
    model_group.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of most recent configurations to analyze"
    )
    
    # Test data option
    model_group.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data file (.npy format) - use this instead of model checkpoints"
    )
    model_group.add_argument(
        "--domain-name",
        type=str,
        default="test_domain",
        help="Domain name for test data"
    )
    
    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis Parameters")
    analysis_group.add_argument(
        "--output-dir",
        type=str,
        default="cluster_results",
        help="Directory to save results"
    )
    analysis_group.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to generate per model"
    )
    analysis_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for generation"
    )
    analysis_group.add_argument(
        "--min-k",
        type=int,
        default=5,
        help="Minimum number of clusters to try"
    )
    analysis_group.add_argument(
        "--max-k",
        type=int,
        default=100,
        help="Maximum number of clusters to try"
    )
    analysis_group.add_argument(
        "--k-step",
        type=int,
        default=5,
        help="Step size for k values"
    )
    analysis_group.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Specific k values to try (comma-separated list, e.g., '5,10,15,20,30,40,50')"
    )
    analysis_group.add_argument(
        "--runs-per-k",
        type=int,
        default=3,
        help="Number of runs to perform for each k value with different random seeds"
    )
    analysis_group.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable feature standardization before GMM fitting"
    )
    
    # Logging parameters
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log results to wandb"
    )
    logging_group.add_argument(
        "--wandb-project",
        type=str,
        default="cluster-optimization",
        help="Wandb project name"
    )
    logging_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity name"
    )
    
    # Device selection
    device_group = parser.add_argument_group("Device")
    device_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    # Dataset parameters for real data
    dataset_group = parser.add_argument_group("Dataset Parameters")
    dataset_group.add_argument(
        "--use-dataset",
        action="store_true",
        help="Use real dataset instead of generating synthetic samples"
    )
    dataset_group.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use when using real data"
    )
    dataset_group.add_argument(
        "--use-gw-processed",
        action="store_true",
        help="Use GW-processed latents (encoded through GW & decoded) instead of original"
    )
    
    # Parse arguments
    return parser.parse_args()


def main():
    """
    Main entry point for the script.
    
    Returns:
        Exit code: 0 for success, non-zero for errors
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create k range
    if args.k_values:
        try:
            k_range = [int(k) for k in args.k_values.split(',')]
            k_range.sort()  # Ensure k values are sorted
            logger.info(f"Using specified k values: {k_range}")
        except ValueError:
            logger.error("Invalid k-values format. Use comma-separated integers, e.g., '5,10,15,20,30,40,50'")
            return 1
    else:
        k_range = list(range(args.min_k, args.max_k + 1, args.k_step))
        logger.info(f"Using k range from {args.min_k} to {args.max_k} with step {args.k_step}")
    
    # Log runs per k
    logger.info(f"Performing {args.runs_per_k} runs per k value with different random seeds")
    
    # Determine standardization setting
    standardize = not args.no_standardize
    logger.info(f"Feature standardization before GMM fitting: {'Disabled' if not standardize else 'Enabled'}")
    
    # Use test data if provided
    if args.test_data is not None:
        logger.info(f"Using test data: {args.test_data}")
        result = run_with_test_data(
            test_data_file=args.test_data,
            output_dir=args.output_dir,
            domain_name=args.domain_name,
            k_range=k_range,
            device=args.device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            runs_per_k=args.runs_per_k,
            standardize=standardize
        )
        return 0 if result else 1
    
    # Otherwise, use model checkpoints
    if args.checkpoint_dir is None and not args.find_latest_checkpoints:
        logger.error("Either --test-data, --checkpoint-dir, or --find-latest-checkpoints must be provided")
        return 1
    
    if args.checkpoint_dir is not None and not args.find_latest_checkpoints and not os.path.isdir(args.checkpoint_dir):
        logger.error(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
        return 1
    
    # If using latest checkpoints, verify fusion directory exists
    if args.find_latest_checkpoints:
        if not os.path.isdir(args.fusion_dir):
            logger.error(f"Fusion directory does not exist: {args.fusion_dir}")
            return 1
            
        # Print information about what we're looking for
        logger.info(f"Looking for latest checkpoints in {args.fusion_dir}")
        if args.max_configs is not None:
            logger.info(f"Will use up to {args.max_configs} most recent configurations")
    
    # Process domain configs
    domain_configs = []
    for path in args.domain_configs:
        if not os.path.isfile(path):
            logger.error(f"Domain checkpoint does not exist: {path}")
            return 1
            
        # Extract domain name
        domain_name = Path(path).stem.split("_")[-1]
        if domain_name not in ["v", "t"]:
            domain_name = Path(path).stem
        
        # Map domain name from v to v_latents if needed
        domain_type = domain_name
        actual_name = domain_name
        if domain_name == "v":
            actual_name = "v_latents"  # Use v_latents as the actual domain name
        
        domain_configs.append({
            "checkpoint_path": path,
            "name": actual_name,
            "domain_type": domain_type
        })
    
    # Get domain names
    domain_names = [config["name"] for config in domain_configs]
    logger.info(f"Analyzing domains: {domain_names}")
    
    # Load domain modules
    domain_modules = load_domain_modules(domain_configs)
    
    # Initialize data module if requested
    data_module = None
    if args.use_dataset:
        try:
            logger.info("Initializing data module for real data samples")
            from simple_shapes_dataset.data_module import SimpleShapesDataModule
            from simple_shapes_dataset.domain import DomainDesc
            
            # Find dataset path
            dataset_path = "full_shapes_dataset/simple_shapes_dataset"
            if not os.path.exists(dataset_path):
                dataset_path = "simple-shapes-dataset/sample_dataset"
                logger.info(f"Full dataset not found, falling back to sample dataset at: {dataset_path}")
            
            logger.info(f"Using dataset at: {dataset_path}")
            
            # Create domain classes and args
            domain_classes = {}
            domain_args = {}
            
            # Set up domain classes based on loaded modules
            for domain_name in domain_names:
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
                elif domain_name == "attr":
                    from simple_shapes_dataset.domain import SimpleShapesAttributes
                    domain_classes[DomainDesc(base="attr", kind="attr")] = SimpleShapesAttributes
                elif domain_name == "v":
                    from simple_shapes_dataset.domain import SimpleShapesVisual
                    domain_classes[DomainDesc(base="v", kind="v")] = SimpleShapesVisual
            
            # Define domain proportions
            domain_proportions = {}
            for domain_name in domain_names:
                domain_proportions[frozenset([domain_name])] = 1.0
                
            # Create custom collate function
            def simple_collate_fn(batch):
                from torch.utils.data._utils.collate import default_collate
                try:
                    return default_collate(batch)
                except Exception:
                    return batch
            
            data_module = SimpleShapesDataModule(
                dataset_path=dataset_path,
                domain_classes=domain_classes,
                domain_proportions=domain_proportions,
                batch_size=args.batch_size,
                num_workers=4,
                seed=42,
                domain_args=domain_args,
                collate_fn=simple_collate_fn
            )
            data_module.setup()
            logger.info(f"Data module initialized, will use {args.dataset_split} split")
        except ImportError as e:
            logger.warning(f"SimpleShapesDataModule initialization failed: {e}")
            logger.warning("Will use synthetic data instead.")
            args.use_dataset = False
        except Exception as e:
            logger.error(f"Error initializing data module: {e}")
            import traceback
            traceback.print_exc()
            args.use_dataset = False
    
    # Find model checkpoints if not using fusion latest mode
    model_paths = []
    if not args.find_latest_checkpoints and args.checkpoint_dir is not None:
        for path in Path(args.checkpoint_dir).rglob("*.pt"):
            # Skip metadata files
            if "_metadata" in path.name:
                continue
            model_paths.append(str(path))
        
        if not model_paths:
            logger.error(f"No model checkpoints found in {args.checkpoint_dir}")
            return 1
        
        # Limit number of models if needed
        if args.max_models > 0 and len(model_paths) > args.max_models:
            logger.info(f"Limiting analysis to {args.max_models} models (out of {len(model_paths)})")
            model_paths = model_paths[:args.max_models]
        
        logger.info(f"Found {len(model_paths)} model checkpoints")
    
    # Run analysis with GPU acceleration
    results = analyze_domain_clusters_gpu(
        model_paths=model_paths,
        domain_modules=domain_modules,
        domain_names=domain_names,
        output_dir=args.output_dir,
        k_range=k_range,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        find_latest=args.find_latest_checkpoints,
        fusion_dir=args.fusion_dir,
        max_configs=args.max_configs,
        data_module=data_module,
        dataset_split=args.dataset_split,
        use_gw_processed=args.use_gw_processed,
        runs_per_k=args.runs_per_k,
        standardize=standardize
    )
    
    # Check if we got any results
    if not results:
        logger.error("No results were generated. Please check the logs for errors.")
        return 1
        
    logger.info("\nGPU-accelerated cluster optimization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 