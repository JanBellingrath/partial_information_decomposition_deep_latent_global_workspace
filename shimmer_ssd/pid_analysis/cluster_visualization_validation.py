#!/usr/bin/env python3
"""
Cluster Visualization Script for Validation Images

This script visualizes validation images in clusters using GLW model and PID analysis results.
It loads 10,000 validation images, corresponding VAE latents, feeds them through GLW model
to get global workspace representations, and uses clusters from PID analysis to create
visualizations showing 100 images per cluster in 10x10 grids on wandb.

Requirements:
- torchvision (required, no fallbacks)
- sklearn 
- matplotlib
- wandb (optional)
- PIL

Author: Assistant
Created for: Cluster visualization and analysis
"""

# Required standard library imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import json
import os
import sys
import glob

# Import required modules - will fail if not available (no fallbacks)
try:
    from torchvision import transforms
    HAS_TORCHVISION = True
    HAS_CLUSTER_VALIDATION = True  # Main validation flag
except ImportError:
    HAS_TORCHVISION = False
    HAS_CLUSTER_VALIDATION = False  # Disable validation if torchvision unavailable
    print("Warning: torchvision not available, cluster validation will be disabled")

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not available, visualization logging will be disabled")

# Set up path for project imports
sys.path.insert(0, '/home/janerik/shimmer-ssd')

# Project imports - these are only needed when actually running the function
def _get_glw_imports():
    """Lazy import of GLW modules to avoid import errors when just loading the module."""
    try:
        # Use the local load_checkpoint from utils.py that handles different checkpoint formats
        from gw_module_configurable_fusion import GWModuleConfigurableFusion
        from shimmer_ssd.pid_analysis.utils import load_checkpoint
        from losses_and_weights_GLW_training import load_domain_modules
        return GWModuleConfigurableFusion, load_checkpoint, load_domain_modules
    except ImportError as e:
        raise ImportError(f"GLW modules not available: {e}. Make sure you're running from the shimmer-ssd root directory.")


def assign_samples_to_clusters(
    gw_representations: torch.Tensor,
    cluster_centers: torch.Tensor,
    cluster_method: str = 'gmm'
) -> np.ndarray:
    """
    Assign samples to clusters based on cluster centers.
    
    Args:
        gw_representations: Global workspace representations [N, dim]
        cluster_centers: Cluster centers [K, dim]
        cluster_method: Clustering method used ('gmm' or 'kmeans')
        
    Returns:
        Cluster assignments [N]
    """
    gw_numpy = gw_representations.cpu().numpy()
    centers_numpy = cluster_centers.cpu().numpy()
    
    print(f"   🔍 DEBUG - Validation data:")
    print(f"      Shape: {gw_numpy.shape}")
    print(f"      Range: [{gw_numpy.min():.6f}, {gw_numpy.max():.6f}]")
    print(f"      Mean: {gw_numpy.mean():.6f}, Std: {gw_numpy.std():.6f}")
    
    print(f"   🔍 DEBUG - Cluster centers:")
    print(f"      Shape: {centers_numpy.shape}")
    print(f"      Range: [{centers_numpy.min():.6f}, {centers_numpy.max():.6f}]")
    print(f"      Mean: {centers_numpy.mean():.6f}, Std: {centers_numpy.std():.6f}")
    
    if cluster_method == 'gmm':
        # For GMM, assign to cluster with highest probability
        from sklearn.mixture import GaussianMixture
        n_clusters = len(centers_numpy)
        
        # Create a GMM with the given centers as initial means
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        # Initialize with cluster centers as means
        gmm.means_ = centers_numpy
        # Use identity covariances and equal weights initially
        gmm.covariances_ = np.array([np.eye(centers_numpy.shape[1]) for _ in range(n_clusters)])
        gmm.weights_ = np.ones(n_clusters) / n_clusters
        gmm.precisions_cholesky_ = np.array([np.eye(centers_numpy.shape[1]) for _ in range(n_clusters)])
        
        # Predict cluster assignments
        cluster_assignments = gmm.predict(gw_numpy)
    else:
        # For K-means, assign to nearest cluster center
        from scipy.spatial.distance import cdist
        distances = cdist(gw_numpy, centers_numpy, metric='euclidean')
        cluster_assignments = np.argmin(distances, axis=1)
    
    # Debug cluster distribution
    unique, counts = np.unique(cluster_assignments, return_counts=True)
    print(f"   📊 Cluster assignment distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"      Cluster {cluster_id}: {count} samples ({count/len(cluster_assignments)*100:.1f}%)")
    
    return cluster_assignments


def load_validation_images(
    val_images_path: str = "/home/janerik/shimmer-ssd/simple_shapes_dataset/val",
    n_samples: int = 10000,
    deterministic: bool = True
) -> torch.Tensor:
    """
    Load validation images with deterministic ordering.
    
    Args:
        val_images_path: Path to validation images directory
        n_samples: Number of images to load
        deterministic: If True, sort filenames numerically for reproducible ordering
        
    Returns:
        Tensor of shape (n_samples, 3, 64, 64) containing images
    """
    print(f"   📂 Loading images from: {val_images_path}")
    
    # Get all image files
    image_dir = Path(val_images_path)
    if not image_dir.exists():
        raise FileNotFoundError(f"Validation images directory not found: {val_images_path}")
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(ext)))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No image files found in {val_images_path}")
    
    print(f"   📊 Found {len(image_files)} total image files")
    
    if deterministic:
        # Sort files numerically by extracting numbers from filename
        def extract_number(filename):
            # Extract numbers from filename for proper sorting
            import re
            numbers = re.findall(r'\d+', filename.stem)
            if numbers:
                return int(numbers[0])  # Use first number found
            return 0
        
        image_files.sort(key=extract_number)
        print(f"   🔢 Sorted files numerically for deterministic loading")
    
    # Limit to requested number of samples
    if len(image_files) > n_samples:
        image_files = image_files[:n_samples]
        print(f"   ✂️  Truncated to {n_samples} images")
    
    # Define transforms - requires torchvision
    if not HAS_TORCHVISION:
        raise RuntimeError("torchvision is required for image loading but not available")
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # Load images
    images = []
    print(f"   🖼️  Loading {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        if i % 1000 == 0:
            print(f"      Progress: {i}/{len(image_files)}")
        
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply transforms
                img_tensor = transform(img)
                images.append(img_tensor)
                
        except Exception as e:
            print(f"      ⚠️  Failed to load {img_path}: {e}")
            continue
    
    if len(images) == 0:
        raise RuntimeError("No images could be loaded successfully")
    
    # Stack into tensor
    images_tensor = torch.stack(images)
    print(f"   ✅ Successfully loaded {len(images)} images with shape {images_tensor.shape}")
    
    return images_tensor


def load_vae_latents(
    dataset_path: str = "/home/janerik/shimmer-ssd/simple_shapes_dataset",
    split: str = "val",
    latent_filename: str = None,
    n_samples: int = 10000
) -> torch.Tensor:
    """
    Load VAE latents for the visual domain from saved numpy files.
    Uses the same approach as SimpleShapesPretrainedVisual in the main data loading.
    
    Args:
        dataset_path: Path to dataset directory
        split: Dataset split ('val', 'train', 'test')
        latent_filename: Specific filename for latents (if known)
        n_samples: Number of latent vectors to load
        
    Returns:
        torch.Tensor: VAE latents [N, latent_dim]
    """
    print(f"🧠 Loading VAE latents for {split} split")
    
    # Use the same default presaved filename as in the main data loading functionality
    if latent_filename is None:
        latent_filename = "calmip-822888_epoch=282-step=1105680_future.npy"
    
    # Construct the path the same way as SimpleShapesPretrainedVisual
    presaved_path = os.path.join(dataset_path, f"saved_latents/{split}/{latent_filename}")
    
    print(f"   📁 Looking for latents at: {presaved_path}")
    
    if os.path.exists(presaved_path):
        print(f"   Loading latents from: {presaved_path}")
        
        try:
            # Load the same way as SimpleShapesPretrainedVisual
            latents = torch.from_numpy(np.load(presaved_path))
            
            # Limit to requested number of samples
            if len(latents) > n_samples:
                latents = latents[:n_samples]
            
            print(f"   ✅ Loaded {len(latents)} VAE latents: {latents.shape}")
            print(f"   Latent value range: [{latents.min():.3f}, {latents.max():.3f}]")
            
            return latents
            
        except Exception as e:
            print(f"   ❌ Error loading latents from {presaved_path}: {e}")
    else:
        print(f"   ⚠️  Presaved latents not found at: {presaved_path}")
    
    # Fallback: look for any latent files in the saved_latents directory
    saved_latents_dir = os.path.join(dataset_path, "saved_latents", split)
    print(f"   🔍 Searching for alternative latent files in: {saved_latents_dir}")
    
    if os.path.exists(saved_latents_dir):
        npy_files = glob.glob(os.path.join(saved_latents_dir, "*.npy"))
        if npy_files:
            print(f"   Found alternative latent files: {[os.path.basename(f) for f in npy_files]}")
            latent_path = npy_files[0]  # Use the first one found
            
            try:
                latents = torch.from_numpy(np.load(latent_path))
                
                # Limit to requested number of samples
                if len(latents) > n_samples:
                    latents = latents[:n_samples]
                
                print(f"   ✅ Loaded {len(latents)} VAE latents from alternative file: {latents.shape}")
                print(f"   Latent value range: [{latents.min():.3f}, {latents.max():.3f}]")
                
                return latents
                
            except Exception as e:
                print(f"   ❌ Error loading latents from {latent_path}: {e}")
    
    # Final fallback: use the generate function (which now loads presaved latents)
    print(f"   🔄 Using fallback method to load presaved latents")
    return generate_vae_latents_on_the_fly(dataset_path, split, n_samples)


def generate_vae_latents_on_the_fly(
    dataset_path: str,
    split: str,
    n_samples: int
) -> torch.Tensor:
    """
    Load pre-saved VAE latents using the same approach as SimpleShapesPretrainedVisual.
    
    Args:
        dataset_path: Path to dataset directory  
        split: Dataset split
        n_samples: Number of samples to process
        
    Returns:
        torch.Tensor: Pre-saved VAE latents [N, latent_dim]
    """
    print(f"   🔄 Loading pre-saved VAE latents for {split} split...")
    
    try:
        # Use the same presaved path as used in the main data loading functionality
        presaved_filename = "calmip-822888_epoch=282-step=1105680_future.npy"
        presaved_path = os.path.join(dataset_path, f"saved_latents/{split}/{presaved_filename}")
        
        print(f"   📁 Loading from: {presaved_path}")
        
        if not os.path.exists(presaved_path):
            raise FileNotFoundError(f"Pre-saved latents not found at: {presaved_path}")
        
        # Load the latents directly (same as SimpleShapesPretrainedVisual)
        latents = torch.from_numpy(np.load(presaved_path))
        
        # Limit to requested number of samples
        if len(latents) > n_samples:
            latents = latents[:n_samples]
        
        print(f"   ✅ Loaded {len(latents)} pre-saved VAE latents: {latents.shape}")
        print(f"   Latent value range: [{latents.min():.3f}, {latents.max():.3f}]")
        
        return latents
        
    except Exception as e:
        print(f"   ❌ Error loading pre-saved latents: {e}")
        raise RuntimeError(f"Could not load pre-saved VAE latents: {e}")


def load_text_latents(
    dataset_path: str = "/home/janerik/shimmer-ssd/simple_shapes_dataset",
    split: str = "val",
    latent_filename: str = "bert-base-uncased",
    n_samples: int = 10000
) -> torch.Tensor:
    """
    Load text latents for the text domain from saved numpy files.
    
    Args:
        dataset_path: Path to dataset directory
        split: Dataset split ('val', 'train', 'test')
        latent_filename: Text latent filename (default: 'bert-base-uncased')
        n_samples: Number of latent vectors to load
        
    Returns:
        torch.Tensor: Text latents [N, latent_dim]
    """
    print(f"📝 Loading text latents for {split} split")
    
    # Construct path for text latents
    text_path = os.path.join(dataset_path, f"{split}_{latent_filename}.npy")
    
    print(f"   📁 Looking for text latents at: {text_path}")
    
    if os.path.exists(text_path):
        try:
            # Load text latents
            text_latents = torch.from_numpy(np.load(text_path))
            
            # SimpleShapesText applies normalization, so we need to do the same
            # Load normalization parameters
            mean_path = os.path.join(dataset_path, f"mean_{latent_filename}.npy")
            std_path = os.path.join(dataset_path, f"std_{latent_filename}.npy")
            
            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean = torch.from_numpy(np.load(mean_path))
                std = torch.from_numpy(np.load(std_path))
                
                # Apply normalization like SimpleShapesText does
                text_latents = (text_latents - mean) / std
                print(f"   ✅ Applied normalization with mean/std from {latent_filename}")
            else:
                print(f"   ⚠️  No normalization parameters found for {latent_filename}")
            
            # Limit to requested number of samples
            if len(text_latents) > n_samples:
                text_latents = text_latents[:n_samples]
            
            print(f"   ✅ Loaded {len(text_latents)} text latents: {text_latents.shape}")
            print(f"   Text latent value range: [{text_latents.min():.3f}, {text_latents.max():.3f}]")
            
            return text_latents
            
        except Exception as e:
            print(f"   ❌ Error loading text latents from {text_path}: {e}")
    else:
        print(f"   ⚠️  Text latents not found at: {text_path}")
    
    # Try alternative filename
    alt_latent_filename = "latent"
    alt_text_path = os.path.join(dataset_path, f"{split}_{alt_latent_filename}.npy")
    print(f"   🔍 Trying alternative: {alt_text_path}")
    
    if os.path.exists(alt_text_path):
        try:
            text_latents = torch.from_numpy(np.load(alt_text_path))
            
            # Apply normalization for alternative filename
            alt_mean_path = os.path.join(dataset_path, f"{alt_latent_filename}_mean.npy")
            alt_std_path = os.path.join(dataset_path, f"{alt_latent_filename}_std.npy")
            
            if os.path.exists(alt_mean_path) and os.path.exists(alt_std_path):
                mean = torch.from_numpy(np.load(alt_mean_path))
                std = torch.from_numpy(np.load(alt_std_path))
                text_latents = (text_latents - mean) / std
                print(f"   ✅ Applied normalization for {alt_latent_filename}")
            
            if len(text_latents) > n_samples:
                text_latents = text_latents[:n_samples]
            
            print(f"   ✅ Loaded {len(text_latents)} text latents from alternative: {text_latents.shape}")
            return text_latents
            
        except Exception as e:
            print(f"   ❌ Error loading alternative text latents: {e}")
    
    raise RuntimeError(f"Could not load text latents for {split} split")


def load_glw_model(
    checkpoint_path: str,
    device: torch.device = None,
    domain_modules: Dict = None
):
    """
    Load a trained GLW model from checkpoint.
    
    Args:
        checkpoint_path: Path to the GLW model checkpoint
        device: Device to load the model on
        domain_modules: Pre-loaded domain modules to use (if None, will load defaults)
        
    Returns:
        Loaded GLW model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔄 Loading GLW model from {checkpoint_path}")
    
    # Import GLW modules when needed
    GWModuleConfigurableFusion, load_checkpoint, load_domain_modules = _get_glw_imports()
    
    # Use provided domain_modules or load defaults
    if domain_modules is None:
        print("   ⚠️  No domain_modules provided, loading default configs")
        # Load domain modules that the GLW model expects
        domain_configs = [
            {
                "name": "v_latents",
                "domain_type": "v_latents",
                "checkpoint_path": "./checkpoints/domain_v.ckpt",
            },
            {
                "name": "t", 
                "domain_type": "t",
                "checkpoint_path": "./checkpoints/domain_t.ckpt",
            }
        ]
        domain_modules = load_domain_modules(domain_configs)
    else:
        print("   ✅ Using provided domain_modules from main analysis")
    
    # Load the GLW model
    model = load_checkpoint(checkpoint_path, domain_modules, device)
    model.eval()
    
    print(f"   ✅ GLW model loaded successfully")
    print(f"   Workspace dimension: {model.workspace_dim}")
    print(f"   Fusion weights: {model.fusion_weights}")
    
    return model


def encode_to_global_workspace(
    vae_latents: torch.Tensor,
    text_latents: torch.Tensor,
    glw_model,
    device: torch.device = None
) -> torch.Tensor:
    """
    Encode VAE latents and text latents to global workspace representations using proper fusion.
    
    Args:
        vae_latents: VAE latent vectors [N, latent_dim]
        text_latents: Text latent vectors [N, text_dim]
        glw_model: Trained GLW model
        device: Device for computation
        
    Returns:
        torch.Tensor: Global workspace representations [N, workspace_dim]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🧮 Encoding {len(vae_latents)} samples to global workspace")
    print(f"   Visual latents: {vae_latents.shape}")
    print(f"   Text latents: {text_latents.shape}")
    
    vae_latents = vae_latents.to(device)
    text_latents = text_latents.to(device)
    
    # Process in batches to avoid memory issues
    batch_size = 128
    gw_representations = []
    
    with torch.no_grad():
        for i in range(0, len(vae_latents), batch_size):
            v_batch = vae_latents[i:i+batch_size]
            t_batch = text_latents[i:i+batch_size]
            
            # Prepare domain representations dict
            domain_representations = {}
            
            # Process visual domain
            if 'v_latents' in glw_model.domain_mods:
                # Handle v_latents extra dimensions if present (take first component - mean vector)
                if v_batch.dim() > 2:
                    v_batch = v_batch[:, 0, :]  # Take first component (mean vector)
                
                # Visual domain does NOT use a projector - encode directly
                v_encoded = glw_model.gw_encoders['v_latents'](v_batch)
                domain_representations['v_latents'] = v_encoded
            
            # Process text domain
            if 't' in glw_model.domain_mods:
                # Text domain DOES use a projector (BERT 768 → 64 dimensions)
                if hasattr(glw_model.domain_mods['t'], 'projector'):
                    t_projected = glw_model.domain_mods['t'].projector(t_batch)
                else:
                    t_projected = t_batch
                
                # Encode through GLW encoder
                t_encoded = glw_model.gw_encoders['t'](t_projected)
                domain_representations['t'] = t_encoded
            
            # Apply fusion using the model's fuse method
            gw_repr = glw_model.fuse(domain_representations, None)  # selection_scores not used
            
            gw_representations.append(gw_repr.cpu())
            
            if (i + batch_size) % 1000 < batch_size:
                print(f"   Encoded {min(i + batch_size, len(vae_latents))}/{len(vae_latents)} samples")
    
    # Concatenate all representations
    gw_tensor = torch.cat(gw_representations, dim=0)
    print(f"   ✅ Generated {len(gw_tensor)} global workspace representations: {gw_tensor.shape}")
    print(f"   GW value range: [{gw_tensor.min():.3f}, {gw_tensor.max():.3f}]")
    
    return gw_tensor


def create_cluster_visualization(
    images: torch.Tensor,
    cluster_labels: np.ndarray,
    cluster_id: int,
    samples_per_cluster: int = 100,
    grid_size: int = 10
) -> plt.Figure:
    """
    Create a 10x10 grid visualization of images for a specific cluster.
    
    Args:
        images: All validation images [N, C, H, W]
        cluster_labels: Cluster assignments [N]
        cluster_id: ID of the cluster to visualize
        samples_per_cluster: Number of samples to show (up to 100)
        grid_size: Grid size (10 for 10x10)
        
    Returns:
        matplotlib Figure object
    """
    # Get indices of samples belonging to this cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    
    if len(cluster_indices) == 0:
        print(f"   ⚠️  No samples found for cluster {cluster_id}")
        return None
    
    # Limit to requested number of samples
    if len(cluster_indices) > samples_per_cluster:
        cluster_indices = cluster_indices[:samples_per_cluster]
    
    # Get the images for this cluster
    cluster_images = images[cluster_indices]
    
    print(f"   📊 Cluster {cluster_id}: {len(cluster_images)} samples")
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f'Cluster {cluster_id} - {len(cluster_images)} samples', 
                 fontsize=16, fontweight='bold')
    
    # Plot images
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            ax = axes[i, j]
            
            if idx < len(cluster_images):
                img = cluster_images[idx]
                
                # Handle different image formats
                if img.dim() == 3:  # [C, H, W]
                    if img.shape[0] == 1:  # Grayscale
                        img = img.squeeze(0)
                        ax.imshow(img, cmap='gray')
                    elif img.shape[0] == 3:  # RGB
                        img = img.permute(1, 2, 0)
                        ax.imshow(img)
                    else:
                        # Multi-channel, show first channel
                        ax.imshow(img[0], cmap='gray')
                elif img.dim() == 2:  # [H, W]
                    ax.imshow(img, cmap='gray')
                else:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            else:
                # Empty cell
                ax.set_facecolor('#f0f0f0')
            
            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig


def visualize_all_clusters(
    images: torch.Tensor,
    cluster_labels: np.ndarray,
    cluster_metadata: Dict,
    max_clusters: int = 20,
    samples_per_cluster: int = 100,
    use_wandb: bool = True,
    wandb_project: str = "cluster-visualization-validation"
) -> Dict[str, Any]:
    """
    Create visualizations for all clusters and log to wandb.
    
    Args:
        images: All validation images [N, C, H, W]
        cluster_labels: Cluster assignments [N]
        cluster_metadata: Metadata about clustering
        max_clusters: Maximum number of clusters to visualize
        samples_per_cluster: Number of samples per cluster
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        
    Returns:
        Dictionary with visualization results
    """
    print(f"\n🎨 Creating cluster visualizations")
    
    # Initialize wandb if requested
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            config={
                'num_images': len(images),
                'num_clusters': cluster_metadata.get('num_clusters', 'unknown'),
                'clustering_method': cluster_metadata.get('method', 'unknown'),
                'max_clusters_visualized': max_clusters,
                'samples_per_cluster': samples_per_cluster,
                'source_file': cluster_metadata.get('source_file', 'unknown')
            }
        )
        print(f"   ✅ Initialized wandb project: {wandb_project}")
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)
    
    print(f"   📊 Found {num_clusters} unique clusters")
    print(f"   🎯 Visualizing up to {max_clusters} clusters")
    
    # Limit clusters if needed
    if num_clusters > max_clusters:
        # Select most frequent clusters
        cluster_counts = np.bincount(cluster_labels)
        top_clusters = np.argsort(cluster_counts)[::-1][:max_clusters]
        unique_clusters = top_clusters
        print(f"   🔝 Selected top {max_clusters} clusters by frequency")
    
    results = {
        'total_clusters': num_clusters,
        'visualized_clusters': len(unique_clusters),
        'cluster_summaries': {}
    }
    
    # Create visualization for each cluster
    for i, cluster_id in enumerate(unique_clusters):
        print(f"\n   🎨 Creating visualization for cluster {cluster_id} ({i+1}/{len(unique_clusters)})")
        
        try:
            fig = create_cluster_visualization(
                images=images,
                cluster_labels=cluster_labels,
                cluster_id=cluster_id,
                samples_per_cluster=samples_per_cluster,
                grid_size=10
            )
            
            if fig is not None:
                # Count samples in this cluster
                cluster_count = np.sum(cluster_labels == cluster_id)
                results['cluster_summaries'][cluster_id] = {
                    'total_samples': cluster_count,
                    'visualized_samples': min(cluster_count, samples_per_cluster)
                }
                
                # Log to wandb if available
                if use_wandb and HAS_WANDB and wandb.run is not None:
                    wandb_image = wandb.Image(fig, caption=f"Cluster {cluster_id} ({cluster_count} samples)")
                    wandb.log({f"cluster_{cluster_id:03d}": wandb_image})
                    print(f"      ✅ Logged to wandb: cluster_{cluster_id:03d}")
                
                # Close figure to save memory
                plt.close(fig)
            
        except Exception as e:
            print(f"      ❌ Error creating visualization for cluster {cluster_id}: {e}")
            continue
    
    # Log summary to wandb
    if use_wandb and HAS_WANDB and wandb.run is not None:
        summary_table = []
        for cluster_id, summary in results['cluster_summaries'].items():
            summary_table.append([
                cluster_id,
                summary['total_samples'],
                summary['visualized_samples']
            ])
        
        if summary_table:
            wandb_table = wandb.Table(
                data=summary_table,
                columns=["Cluster ID", "Total Samples", "Visualized Samples"]
            )
            wandb.log({"cluster_summary": wandb_table})
            print(f"   ✅ Logged summary table to wandb")
    
    print(f"\n🎉 Visualization complete!")
    print(f"   Total clusters found: {results['total_clusters']}")
    print(f"   Clusters visualized: {results['visualized_clusters']}")
    
    return results


def visualize_validation_clusters(
    images: torch.Tensor,
    cluster_labels: np.ndarray,
    cluster_metadata: Dict,
    max_clusters: int = 20,
    samples_per_cluster: int = 100,
    wandb_run = None
) -> Dict[str, Any]:
    """
    Create cluster visualizations specifically for validation that integrates with existing wandb run.
    
    Args:
        images: All validation images [N, C, H, W]
        cluster_labels: Cluster assignments [N]
        cluster_metadata: Metadata about clustering
        max_clusters: Maximum number of clusters to visualize
        samples_per_cluster: Number of samples per cluster
        wandb_run: Existing wandb run (if any)
        
    Returns:
        Dictionary with visualization results
    """
    print(f"\n🎨 Creating validation cluster visualizations")
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)
    
    print(f"   📊 Found {num_clusters} unique clusters")
    print(f"   🎯 Visualizing up to {max_clusters} clusters")
    
    # Limit clusters if needed
    if num_clusters > max_clusters:
        # Select most frequent clusters
        cluster_counts = np.bincount(cluster_labels)
        top_clusters = np.argsort(cluster_counts)[::-1][:max_clusters]
        unique_clusters = top_clusters
        print(f"   🔝 Selected top {max_clusters} clusters by frequency")
    
    results = {
        'total_clusters': num_clusters,
        'visualized_clusters': len(unique_clusters),
        'cluster_summaries': {}
    }
    
    # Create visualization for each cluster
    for i, cluster_id in enumerate(unique_clusters):
        print(f"\n   🎨 Creating validation visualization for cluster {cluster_id} ({i+1}/{len(unique_clusters)})")
        
        try:
            fig = create_cluster_visualization(
                images=images,
                cluster_labels=cluster_labels,
                cluster_id=cluster_id,
                samples_per_cluster=samples_per_cluster,
                grid_size=10
            )
            
            if fig is not None:
                # Count samples in this cluster
                cluster_count = np.sum(cluster_labels == cluster_id)
                results['cluster_summaries'][cluster_id] = {
                    'total_samples': cluster_count,
                    'visualized_samples': min(cluster_count, samples_per_cluster)
                }
                
                # Log to existing wandb run if available
                if wandb_run is not None and HAS_WANDB:
                    wandb_image = wandb.Image(fig, caption=f"Validation Cluster {cluster_id} ({cluster_count} samples)")
                    wandb.log({f"validation_cluster_{cluster_id:03d}": wandb_image})
                    print(f"      ✅ Logged to existing wandb run: validation_cluster_{cluster_id:03d}")
                
                # Close figure to save memory
                plt.close(fig)
            
        except Exception as e:
            print(f"      ❌ Error creating validation visualization for cluster {cluster_id}: {e}")
            continue
    
    # Log summary to wandb if available
    if wandb_run is not None and HAS_WANDB:
        summary_table = []
        for cluster_id, summary in results['cluster_summaries'].items():
            summary_table.append([
                f"Validation Cluster {cluster_id}",
                summary['total_samples'],
                summary['visualized_samples']
            ])
        
        if summary_table:
            wandb_table = wandb.Table(
                data=summary_table,
                columns=["Cluster ID", "Total Samples", "Visualized Samples"]
            )
            wandb.log({"validation_cluster_summary": wandb_table})
            print(f"   ✅ Logged validation summary table to existing wandb run")
    
    print(f"\n🎉 Validation visualization complete!")
    print(f"   Total clusters found: {results['total_clusters']}")
    print(f"   Clusters visualized: {results['visualized_clusters']}")
    
    return results


def run_cluster_validation_from_results(
    model_path: str,
    domain_modules: Dict,
    analysis_results: Dict[str, Any],
    wandb_run = None,
    validation_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Professional interface to run cluster validation using results from PID analysis.
    
    This function is designed to be called from the main PID analysis script
    to validate cluster meaningfulness using validation images.
    
    Args:
        model_path: Path to the GLW model checkpoint
        domain_modules: Domain modules dict from main analysis
        analysis_results: Results dictionary from main PID analysis containing:
            - 'cluster_labels': Cluster assignments from training
            - 'generated_data': Generated data containing GW representations
            - 'cluster_metadata': Metadata about clustering method
        wandb_run: Current wandb run from main analysis (if any)
        validation_config: Configuration for validation containing:
            - 'val_images_path': Path to validation images
            - 'dataset_path': Path to validation dataset root
            - 'n_samples': Number of validation samples
            - 'max_clusters': Max clusters to visualize
            - 'samples_per_cluster': Samples per cluster visualization
        
    Returns:
        Dictionary with validation results that can be integrated into main results
    """
    if not HAS_TORCHVISION:
        print("⚠️  Skipping cluster validation: torchvision not available")
        return {'status': 'skipped', 'reason': 'torchvision_unavailable'}
    
    # Create single device instance for consistency throughout validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    print("\n" + "="*60)
    print("🔬 CLUSTER VALIDATION - VALIDATION IMAGES")
    print("="*60)
    print("🎯 Validating cluster meaningfulness using validation data")
    print("="*60)
    
    # Extract required data from analysis results
    cluster_labels = analysis_results.get('cluster_labels', None)
    generated_data = analysis_results.get('generated_data', {})
    cluster_metadata = analysis_results.get('cluster_metadata', {
        'method': 'unknown',
        'num_clusters': 'unknown'
    })
    
    if cluster_labels is None:
        print("❌ No cluster labels found in analysis results")
        return {'status': 'failed', 'reason': 'no_cluster_labels'}

    if cluster_metadata is None:
        print("❌ No cluster metadata found in analysis results")
        return {'status': 'failed', 'reason': 'no_cluster_metadata'}
    
    #TODO what is generated data?
    
    # Ensure cluster_labels is a numpy array
    if torch.is_tensor(cluster_labels):
        cluster_labels = cluster_labels.cpu().numpy()
    elif not isinstance(cluster_labels, np.ndarray):
        cluster_labels = np.array(cluster_labels)
    
    # Set default validation config
    if validation_config is None:
        validation_config = {}
    
    val_images_path = validation_config.get('val_images_path', "/home/janerik/shimmer-ssd/simple_shapes_dataset/val")
    dataset_path = validation_config.get('dataset_path', "/home/janerik/shimmer-ssd/simple_shapes_dataset")
    n_samples = validation_config.get('n_samples', 10000)
    max_clusters = validation_config.get('max_clusters', 20) 
    samples_per_cluster = validation_config.get('samples_per_cluster', 100)
    
    print(f"📂 Validation Images: {val_images_path}")
    print(f"📊 Samples: {n_samples}")
    print(f"🎯 Max Clusters: {max_clusters}")
    print(f"🖼️  Samples per Cluster: {samples_per_cluster}")
    
    try:
        # Step 1: Load validation images
        print("\n📸 Step 1: Loading validation images")
        images = load_validation_images(
            val_images_path=val_images_path,
            n_samples=n_samples,
            deterministic=True
        )
        
        # Step 2: Load VAE latents
        print("\n🧠 Step 2: Loading VAE latents")
        vae_latents = load_vae_latents(
            dataset_path=dataset_path,
            split="val",
            n_samples=n_samples
        )
        
        # Step 3: Load text latents
        print("\n📝 Step 3: Loading text latents")
        text_latents = load_text_latents(
            dataset_path=dataset_path,
            split="val",
            latent_filename="bert-base-uncased",
            n_samples=n_samples
        )
        
        # Verify correspondence between all data types
        data_lengths = [len(images), len(vae_latents), len(text_latents)]
        if not all(length == data_lengths[0] for length in data_lengths):
            print(f"⚠️  Data length mismatch:")
            print(f"   Images: {len(images)}")
            print(f"   VAE latents: {len(vae_latents)}")
            print(f"   Text latents: {len(text_latents)}")
            
            min_len = min(data_lengths)
            images = images[:min_len]
            vae_latents = vae_latents[:min_len]
            text_latents = text_latents[:min_len]
            print(f"   Truncated all to {min_len} samples for correspondence")
        else:
            print(f"✅ All data types have {len(images)} samples - correspondence verified")
        
        # Step 4: Load GLW model
        print("\n🔄 Step 4: Loading GLW model")
        glw_model = load_glw_model(model_path, device, domain_modules)
        
        # Step 5: Encode to global workspace
        print("\n🧮 Step 5: Encoding to global workspace")
        gw_representations = encode_to_global_workspace(
            vae_latents=vae_latents,
            text_latents=text_latents,
            glw_model=glw_model,
            device=device
        )
        
        # Step 6: Predict cluster assignments using training clusters
        print("\n🎯 Step 6: Predicting cluster assignments from training clusters")
        
        # Extract cluster centers from generated data
        cluster_centers = None
        cluster_method = cluster_metadata.get('method', 'gmm')
        
        if 'cluster_centers' in generated_data:
            cluster_centers = generated_data['cluster_centers']
            print(f"   ✅ Using pre-computed cluster centers ({len(cluster_centers)} centers)")
            
            # DEBUG: Print cluster center statistics
            centers_numpy_debug = cluster_centers.cpu().numpy() if torch.is_tensor(cluster_centers) else np.array(cluster_centers)
            print(f"   🔍 DEBUG - Pre-computed cluster centers:")
            print(f"      Shape: {centers_numpy_debug.shape}")
            print(f"      Range: [{centers_numpy_debug.min():.6f}, {centers_numpy_debug.max():.6f}]")
            print(f"      Mean: {centers_numpy_debug.mean():.6f}, Std: {centers_numpy_debug.std():.6f}")
            
            # Verify we have the clustering target data for consistency
            if 'clustering_target_data' in generated_data:
                clustering_data = generated_data['clustering_target_data']
                print(f"   📊 Clustering target data available: {clustering_data.shape}")
                
                # DEBUG: Print clustering target data statistics
                clustering_numpy_debug = clustering_data.cpu().numpy() if torch.is_tensor(clustering_data) else np.array(clustering_data)
                print(f"   🔍 DEBUG - Original clustering target data:")
                print(f"      Shape: {clustering_numpy_debug.shape}")
                print(f"      Range: [{clustering_numpy_debug.min():.6f}, {clustering_numpy_debug.max():.6f}]")
                print(f"      Mean: {clustering_numpy_debug.mean():.6f}, Std: {clustering_numpy_debug.std():.6f}")
            else:
                print(f"   ⚠️  No clustering target data found, cluster centers may be inconsistent")
                
        elif 'clustering_target_data' in generated_data:
            print(f"   🔄 No pre-computed cluster centers, computing from clustering target data")
            
            # Use the exact same data that was used for clustering
            clustering_data = generated_data['clustering_target_data']
            if torch.is_tensor(clustering_data):
                clustering_numpy = clustering_data.cpu().numpy()
            else:
                clustering_numpy = np.array(clustering_data)
                
            n_clusters = len(np.unique(cluster_labels))
            
            print(f"   🔄 Computing cluster centers from clustering target data ({cluster_method})")
            if 'gmm' in cluster_method.lower():
                from sklearn.mixture import GaussianMixture
                clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                clusterer.fit(clustering_numpy)
                cluster_centers = torch.from_numpy(clusterer.means_)
            else:  # kmeans
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                clusterer.fit(clustering_numpy)
                cluster_centers = torch.from_numpy(clusterer.cluster_centers_)
            
            print(f"   ✅ Computed {len(cluster_centers)} cluster centers from clustering target data")
            
        elif 'gw_rep' in generated_data:
            print(f"   🔄 Fallback: Computing cluster centers from 'gw_rep' (may not match original clustering)")
            
            # Compute cluster centers from training GW representations
            from sklearn.cluster import KMeans
            from sklearn.mixture import GaussianMixture
            
            training_gw_tensor = generated_data['gw_rep']
            if torch.is_tensor(training_gw_tensor):
                training_gw = training_gw_tensor.cpu().numpy()
            else:
                training_gw = np.array(training_gw_tensor)
            n_clusters = len(np.unique(cluster_labels))
            
            print(f"   🔄 Computing cluster centers from gw_rep fallback ({cluster_method})")
            if 'gmm' in cluster_method.lower():
                clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                clusterer.fit(training_gw)
                cluster_centers = torch.from_numpy(clusterer.means_)
            else:  # kmeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                clusterer.fit(training_gw)
                cluster_centers = torch.from_numpy(clusterer.cluster_centers_)
            
            print(f"   ✅ Computed {len(cluster_centers)} cluster centers from gw_rep fallback")
        else:
            print("❌ No cluster centers, clustering target data, or GW representations found in generated data")
            print(f"   Available keys in generated_data: {list(generated_data.keys())}")
            return {'status': 'failed', 'reason': 'no_clustering_data'}
        
        # Predict cluster assignments for validation data
        if cluster_centers is not None:
            # Apply proper standardization using TRAINING statistics
            print(f"   🔄 Applying proper standardization for cluster assignment")
            
            # Apply separate standardization to both validation data and cluster centers
            print(f"   🔄 Applying separate standardization for consistent cluster assignment")
            
            from sklearn.preprocessing import StandardScaler
            
            # Get raw data
            gw_numpy = gw_representations.cpu().numpy()
            centers_numpy = cluster_centers.cpu().numpy()
            
            print(f"   📊 Raw data statistics:")
            print(f"      Validation: mean={gw_numpy.mean():.6f}, std={gw_numpy.std():.6f}")
            print(f"      Centers: mean={centers_numpy.mean():.6f}, std={centers_numpy.std():.6f}")
            
            # Fit separate scalers
            scaler_validation = StandardScaler()
            scaler_centers = StandardScaler()
            
            # Standardize validation data
            gw_normalized = scaler_validation.fit_transform(gw_numpy)
            
            # Standardize cluster centers
            centers_normalized = scaler_centers.fit_transform(centers_numpy)
            
            print(f"   ✅ Applied separate standardization:")
            print(f"      Validation: mean={gw_normalized.mean():.6f}, std={gw_normalized.std():.6f}")
            print(f"      Centers: mean={centers_normalized.mean():.6f}, std={centers_normalized.std():.6f}")
            
            # Use normalized data for cluster assignment
            cluster_assignments = assign_samples_to_clusters(
                torch.from_numpy(gw_normalized).float(),
                torch.from_numpy(centers_normalized).float(),
                cluster_method
            )
            
            # Use the results from the standardization above
            predicted_labels = cluster_assignments
        else:
            print("❌ No cluster centers available for prediction")
            return {'status': 'failed', 'reason': 'no_cluster_centers_for_prediction'}
        # Step 7: Create cluster distribution histogram
        print("\n📊 Step 7: Creating cluster distribution histogram")
        
        # Create histogram of cluster assignments
        unique_clusters, cluster_counts = np.unique(predicted_labels, return_counts=True)
        
        print(f"   📈 Plotting distribution of {len(predicted_labels)} validation samples across {len(unique_clusters)} clusters")
        
        # Create histogram plot
        plt.figure(figsize=(12, 6))
        plt.bar(unique_clusters, cluster_counts, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Validation Samples')
        plt.title(f'Distribution of Validation Samples Across Clusters\n({len(predicted_labels):,} samples, {len(unique_clusters)} clusters)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_samples = cluster_counts.mean()
        std_samples = cluster_counts.std()
        max_samples = cluster_counts.max()
        min_samples = cluster_counts.min()
        
        stats_text = f'Stats: μ={mean_samples:.1f}, σ={std_samples:.1f}, max={max_samples}, min={min_samples}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Log to wandb if available
        if wandb_run:
            wandb_run.log({"validation_cluster_distribution": wandb.Image(plt)})
            print(f"      ✅ Logged cluster distribution histogram to wandb")
        
        # Save locally as well
        hist_path = f"cluster_distribution_validation.png"
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved histogram to: {hist_path}")
        
        # Print summary statistics
        print(f"   📈 Cluster Distribution Statistics:")
        print(f"      Total clusters with samples: {len(unique_clusters)}")
        print(f"      Average samples per cluster: {mean_samples:.1f}")
        print(f"      Standard deviation: {std_samples:.1f}")
        print(f"      Most populated cluster: {max_samples} samples")
        print(f"      Least populated cluster: {min_samples} samples")
        print(f"      Coverage: {len(unique_clusters)}/{len(np.unique(cluster_labels))} training clusters have validation samples")
        
        # Step 8: Create cluster visualizations
        print("\n🎨 Step 8: Creating cluster visualizations")
        
        # Update cluster metadata for validation
        validation_metadata = cluster_metadata.copy()
        validation_metadata.update({
            'source': 'validation_prediction',
            'training_clusters': len(np.unique(cluster_labels)),
            'validation_clusters': len(np.unique(predicted_labels)),
            'cluster_distribution_stats': {
                'mean_samples_per_cluster': float(mean_samples),
                'std_samples_per_cluster': float(std_samples),
                'max_samples_per_cluster': int(max_samples),
                'min_samples_per_cluster': int(min_samples),
                'cluster_coverage': f"{len(unique_clusters)}/{len(np.unique(cluster_labels))}"
            }
        })
        
        # Create custom visualization function that works with existing wandb run
        results = visualize_validation_clusters(
            images=images,
            cluster_labels=predicted_labels,
            cluster_metadata=validation_metadata,
            max_clusters=max_clusters,
            samples_per_cluster=samples_per_cluster,
            wandb_run=wandb_run
        )
        
        # Add validation-specific metadata
        results.update({
            'validation_samples': len(images),
            'model_path': model_path,
            'cluster_method': cluster_method,
            'prediction_method': 'cluster_centers' if cluster_centers is not None else 'random',
            'status': 'completed'
        })
        
        print(f"\n🎉 CLUSTER VALIDATION COMPLETED!")
        print(f"📊 Validated {len(images)} samples across {results['visualized_clusters']} clusters")
        if wandb_run is not None:
            print(f"🎨 Logged visualizations to existing wandb run with prefix 'validation_cluster_'")
        else:
            print(f"🎨 Visualizations created (no wandb logging)")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error in cluster validation: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'reason': str(e)}


def main():
    """CLI function for cluster visualization of validation images."""
    parser = argparse.ArgumentParser(
        description="Visualize validation image clusters using GLW model and PID analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--glw-checkpoint", type=str, required=True,
                       help="Path to GLW model checkpoint file")
    
    # Data paths  
    parser.add_argument("--val-images-path", type=str, 
                       default="/home/janerik/shimmer-ssd/simple_shapes_dataset/val",
                       help="Path to validation images directory")
    parser.add_argument("--dataset-path", type=str,
                       default="/home/janerik/shimmer-ssd/simple_shapes_dataset", 
                       help="Path to dataset root directory")
    parser.add_argument("--pid-results-dir", type=str, 
                       default="shimmer_ssd/pid_analysis/pid_results",
                       help="Directory containing PID analysis results")
    
    # Data parameters
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of samples to process")
    parser.add_argument("--text-latent-filename", type=str, 
                       default="bert-base-uncased",
                       help="Text latent filename identifier")
    
    # Cluster parameters
    parser.add_argument("--cluster-file", type=str, default=None,
                       help="Specific cluster file to load from PID analysis")
    parser.add_argument("--max-clusters", type=int, default=20,
                       help="Maximum number of clusters to visualize")
    parser.add_argument("--samples-per-cluster", type=int, default=100,
                       help="Number of samples to show per cluster")
    
    # Visualization parameters
    parser.add_argument("--wandb-project", type=str, default="cluster-visualization-validation",
                       help="Wandb project name for logging")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging completely")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not os.path.exists(args.glw_checkpoint):
        print(f"❌ GLW checkpoint not found: {args.glw_checkpoint}")
        return 1
    
    if not os.path.exists(args.val_images_path):
        print(f"❌ Validation images directory not found: {args.val_images_path}")
        return 1
    
    print("🚀 Starting Cluster Visualization for Validation Split")
    print("=" * 60)
    print(f"GLW Checkpoint: {args.glw_checkpoint}")
    print(f"Images Path: {args.val_images_path}")
    print(f"Samples: {args.n_samples}")
    print(f"Max Clusters: {args.max_clusters}")
    print(f"Samples per Cluster: {args.samples_per_cluster}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        # Step 1: Load validation images
        print("\n📸 Step 1: Loading validation images")
        images = load_validation_images(
            val_images_path=args.val_images_path,
            n_samples=args.n_samples,
            deterministic=True
        )
        
        # Step 2: Load VAE latents
        print("\n🧠 Step 2: Loading VAE latents")
        vae_latents = load_vae_latents(
            dataset_path=args.dataset_path,
            split="val",
            n_samples=args.n_samples
        )
        
        # Step 3: Load text latents
        print("\n📝 Step 3: Loading text latents")
        text_latents = load_text_latents(
            dataset_path=args.dataset_path,
            split="val",
            latent_filename=args.text_latent_filename,
            n_samples=args.n_samples
        )
        
        # Verify correspondence between all data types
        data_lengths = [len(images), len(vae_latents), len(text_latents)]
        if not all(length == data_lengths[0] for length in data_lengths):
            print(f"⚠️  Data length mismatch:")
            print(f"   Images: {len(images)}")
            print(f"   VAE latents: {len(vae_latents)}")
            print(f"   Text latents: {len(text_latents)}")
            
            min_len = min(data_lengths)
            images = images[:min_len]
            vae_latents = vae_latents[:min_len]
            text_latents = text_latents[:min_len]
            print(f"   Truncated all to {min_len} samples for correspondence")
        else:
            print(f"✅ All data types have {len(images)} samples - correspondence verified")
        
        # Step 4: Load GLW model
        print("\n🔄 Step 4: Loading GLW model")
        glw_model = load_glw_model(args.glw_checkpoint, device)
        
        # Step 5: Encode to global workspace
        print("\n🧮 Step 5: Encoding to global workspace")
        gw_representations = encode_to_global_workspace(
            vae_latents=vae_latents,
            text_latents=text_latents,
            glw_model=glw_model,
            device=device
        )
        
        
        # Step 7: Create visualizations
        print("\n🎨 Step 7: Creating cluster visualizations")
        results = visualize_all_clusters(
            images=images,
            cluster_labels=cluster_labels,
            cluster_metadata=cluster_metadata,
            max_clusters=args.max_clusters,
            samples_per_cluster=args.samples_per_cluster,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project
        )
        
        # Finish wandb run
        if not args.no_wandb and HAS_WANDB and wandb.run is not None:
            wandb.finish()
        
        print("\n✅ Cluster visualization completed successfully!")
        print(f"📊 Summary: {results['visualized_clusters']}/{results['total_clusters']} clusters visualized")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error in cluster visualization pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Finish wandb run even on error
        if not args.no_wandb and HAS_WANDB and wandb.run is not None:
            wandb.finish()
        
        return 1


if __name__ == "__main__":
    exit(main()) 