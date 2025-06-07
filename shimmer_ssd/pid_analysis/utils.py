import gc
import itertools
import json
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.utils.checkpoint
from pathlib import Path
import re
from typing import Dict, List, Any, Optional, Mapping, Union, Callable, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import for new metrics
from sklearn.metrics import (
    precision_recall_fscore_support,
    jaccard_score,
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

# Add the parent directory to the path for imports (like analyze_pid_new.py does)
import sys
import os
# Add root directory to path so we can import the shimmer files
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Try to import shimmer modules with proper path handling
import sys
import os
# Add root directory to path so we can import the shimmer files
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Try to import the critical shimmer modules from root directory
try:
    # Import the GWModuleConfigurableFusion from the local file
    from gw_module_configurable_fusion import GWModuleConfigurableFusion
    # Import the necessary functions
    from minimal_script_with_validation import load_domain_modules as shimmer_load_domain_modules
    print("✅ Successfully imported REAL shimmer modules from root directory")
    SHIMMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: module not found ({e}). Running in limited mode (can show help but not execute PID analysis).")
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

    # Function to load domain modules that fails hard
    def shimmer_load_domain_modules(configs): 
        raise RuntimeError(
            "❌ CRITICAL ERROR: load_domain_modules not available from minimal_script_with_validation!\n"
            "This system requires REAL shimmer modules to function.\n"
            "NEVER use synthetic/dummy data - shimmer or break!"
        )

# Create minimal DomainModule class for type hints (like analyze_pid_new.py does)
class DomainModule:
    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim

# Try to import shimmer domain modules for encoder/decoder
try:
    from shimmer.modules.gw_module import GWEncoder, GWDecoder
    SHIMMER_ENCODERS_AVAILABLE = True
    print("✅ Successfully imported GW encoders/decoders from shimmer!")
except ImportError as e:
    print(f"⚠️  Warning: Could not import shimmer GW modules: {e}")
    SHIMMER_ENCODERS_AVAILABLE = False
    
    # Create dummy encoder/decoder classes
    class GWEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
            super().__init__()
            self.net = nn.Linear(in_dim, out_dim)  # Simple linear layer
        def forward(self, x):
            return self.net(x)
            
    class GWDecoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
            super().__init__()
            self.net = nn.Linear(in_dim, out_dim)  # Simple linear layer  
        def forward(self, x):
            return self.net(x)

# Global performance configuration
CHUNK_SIZE = 128  # Size of chunks for processing large matrices sequentially
MEMORY_CLEANUP_INTERVAL = 10  # Number of iterations after which to force memory cleanup
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean up memory between operations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add professional plotting imports and settings at the top
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Any
import pathlib

# Professional color palette and styling
PROFESSIONAL_COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Deep pink/purple  
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'success': '#5D737E',      # Steel blue-gray
    'neutral': '#7F8C8D',      # Gray
    'background': '#F8F9FA',   # Light gray
    'text': '#2C3E50'          # Dark blue-gray
}

# Professional matplotlib styling
def setup_professional_plot_style():
    """Setup professional matplotlib styling with consistent theme"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    })

def create_professional_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", 
                                       wandb_key=None, save_path=None):
    """Create a professional confusion matrix plot"""
    setup_professional_plot_style()
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with professional styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'shrink': 0.8},
                square=True, linewidths=0.5,
                ax=ax)
    
    # Professional styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Set class names if provided
    if class_names:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    # Log to wandb if key provided
    if wandb_key and 'wandb' in globals() and wandb.run is not None:
        wandb.log({wandb_key: wandb.Image(fig)})
    
    plt.close()
    return fig

def create_professional_training_curves(metrics_history: Dict[str, list], title="Training Curves", 
                                       wandb_key=None, save_path=None):
    """Create professional training curves plot"""
    setup_professional_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = [PROFESSIONAL_COLORS['primary'], PROFESSIONAL_COLORS['secondary'], 
              PROFESSIONAL_COLORS['tertiary'], PROFESSIONAL_COLORS['quaternary']]
    
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        if idx >= 4:  # Limit to 4 subplots
            break
            
        ax = axes[idx]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, color=colors[idx % len(colors)], 
                linewidth=2.5, marker='o', markersize=4,
                label=metric_name.replace('_', ' ').title())
        
        ax.set_title(f"{metric_name.replace('_', ' ').title()}", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Remove empty subplots
    for idx in range(len(metrics_history), 4):
        fig.delaxes(axes[idx])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if wandb_key and 'wandb' in globals() and wandb.run is not None:
        wandb.log({wandb_key: wandb.Image(fig)})
    
    plt.close()
    return fig

def create_professional_pid_comparison(pid_results: Dict[str, float], title="PID Components Analysis",
                                     wandb_key=None, save_path=None):
    """Create professional PID components visualization"""
    setup_professional_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract PID components
    components = ['Redundancy', 'Unique 1', 'Unique 2', 'Synergy']
    values = [pid_results.get(comp.lower().replace(' ', '_'), 0) for comp in components]
    colors = [PROFESSIONAL_COLORS['primary'], PROFESSIONAL_COLORS['secondary'], 
              PROFESSIONAL_COLORS['tertiary'], PROFESSIONAL_COLORS['quaternary']]
    
    # Bar chart
    bars = ax1.bar(components, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_title('PID Components (Bar Chart)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Information (nats)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(values, labels=components, colors=colors, autopct='%1.1f%%',
                                      startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    ax2.set_title('PID Components (Distribution)', fontsize=14, fontweight='bold')
    
    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if wandb_key and 'wandb' in globals() and wandb.run is not None:
        wandb.log({wandb_key: wandb.Image(fig)})
    
    plt.close()
    return fig

def prepare_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.complexfloating)):
        return obj.item()  # Convert NumPy scalars to Python primitives
    elif isinstance(obj, (list, tuple)):
        return [prepare_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, dict):
        return {key: prepare_for_json(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {'__class__': obj.__class__.__name__, 
                '__data__': prepare_for_json(obj.__dict__)}
    else:
        return obj

def load_checkpoint(
    checkpoint_path: str,
    domain_modules: Optional[Mapping[str, DomainModule]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> GWModuleConfigurableFusion:
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict, hidden_dim, n_layers, workspace_dim, fusion_weights = None, 32, 3, 12, None

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        hidden_dim = checkpoint.get("hidden_dim", 32)
        n_layers = checkpoint.get("n_layers", 3)
        workspace_dim = checkpoint.get("workspace_dim", 12)
        fusion_weights = checkpoint.get("fusion_weights", None)
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            hidden_dim = hparams.get("hidden_dim", 32)
            n_layers = hparams.get("n_layers", 3)
            workspace_dim = hparams.get("workspace_dim", 12)
            fusion_weights = hparams.get("fusion_weights", None)
    else:
        state_dict = checkpoint
    
    if domain_modules is None:
        raise ValueError("domain_modules must be provided to load_checkpoint")

    gw_encoders = {}
    gw_decoders = {}
    for domain_name, domain_module in domain_modules.items():
        # Use the exact same approach as analyze_pid_new.py
        latent_dim = domain_module.latent_dim  # Direct access like in analyze_pid_new.py
        print(f"Domain '{domain_name}' latent dimension: {latent_dim}")
        
        # Import exactly like analyze_pid_new.py does it
        from shimmer.modules.gw_module import GWEncoder, GWDecoder
        
        encoder = GWEncoder(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=workspace_dim,
            n_layers=n_layers,
        )
        
        decoder = GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=n_layers,
        )
        
        gw_encoders[domain_name] = encoder
        gw_decoders[domain_name] = decoder
    
    if fusion_weights is None:
        weight_value = 1.0 / len(domain_modules) if domain_modules else 0.0
        fusion_weights = {name: weight_value for name in domain_modules}
    
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
    dataset_split: str = "test",
    use_real_data_files: bool = True  # New parameter to enable real data file loading
) -> Dict[str, torch.Tensor]:
    if data_module is not None:
        print(f"Using real data from {dataset_split} split for sample generation")
        return generate_samples_from_dataset_fixed(
            model=model, data_module=data_module, domain_names=domain_names,
            split=dataset_split, n_samples=n_samples, batch_size=batch_size,
            device=device, use_gw_encoded=use_gw_encoded)
    
    # Try to use real data files if available and requested
    if use_real_data_files:
        try:
            print("🔍 Attempting to use real data files from pid_analysis_data/...")
            real_data = load_real_data_files(domain_names, n_samples, device)
            if real_data is not None:
                print("✅ Successfully loaded REAL data from pid_analysis_data/ files!")
                return real_data
        except Exception as e:
            print(f"⚠️  Warning: Could not load real data files: {e}")
    
    # HARD ERROR: Never use synthetic data - always require real shimmer data
    raise ValueError(
        "❌ CRITICAL ERROR: No data_module provided for sample generation!\n"
        "This system is configured to ONLY use real shimmer data, never synthetic.\n"
        "Please ensure:\n"
        "1. --use-dataset flag is provided\n"
        "2. --dataset-path points to a valid checkpoint with data_module\n"
        "3. The checkpoint contains proper shimmer domain modules and data\n"
        "OR real data files are available in pid_analysis_data/\n"
        "NEVER use synthetic/dummy data - shimmer or break!"
    )

def generate_samples_from_dataset_fixed(
    model: GWModuleConfigurableFusion,
    data_module,
    domain_names: List[str],
    split: str = "test",
    n_samples: int = 1000,
    batch_size: int = 64, # This batch_size is for the dataloader, not internal processing.
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_gw_encoded: bool = False,
) -> Dict[str, torch.Tensor]:
    if len(domain_names) != 2: raise ValueError(f"Requires 2 domains, got {len(domain_names)}")
    model = model.to(device); model.eval()
    domain_a, domain_b = domain_names
    
    reps = {f"{dn}_{rep_type}": [] for dn in domain_names for rep_type in ["orig", "gw_encoded", "decoded"]}
    reps["gw_reps"] = []

    dl_map = {"train": data_module.train_dataloader, "val": data_module.val_dataloader, "test": data_module.test_dataloader}
    if split not in dl_map: raise ValueError(f"Unknown split: {split}")
    dataloader = dl_map[split]()
    
    total_samples = 0
    with torch.no_grad():
        for batch_data in dataloader:
            if total_samples >= n_samples: break
            processed_batch = process_batch(batch_data, device)
            if not (domain_a in processed_batch and domain_b in processed_batch): continue

            latents = {dn: processed_batch[dn] for dn in domain_names}
            current_bs = latents[domain_a].size(0)
            
            # Apply projectors if they exist (e.g., for text)
            for dn in domain_names:
                if hasattr(model.domain_mods.get(dn), 'projector'):
                    latents[dn] = model.domain_mods[dn].projector(latents[dn])
            
            gw_states = {dn: model.gw_encoders[dn](latents[dn]) for dn in domain_names}
            for dn in domain_names: # Handle v_latents special case
                if dn == 'v_latents' and gw_states[dn].dim() > 2: gw_states[dn] = gw_states[dn][:, 0, :]
            
            fused_gw = model.fuse(gw_states, None)
            decoded_states = {dn: model.gw_decoders[dn](fused_gw) for dn in domain_names}

            reps[f"{domain_a}_orig"].append(gw_states[domain_a].cpu() if use_gw_encoded else latents[domain_a].cpu())
            reps[f"{domain_b}_orig"].append(gw_states[domain_b].cpu() if use_gw_encoded else latents[domain_b].cpu())
            for dn in domain_names:
                reps[f"{dn}_gw_encoded"].append(gw_states[dn].cpu())
                reps[f"{dn}_decoded"].append(decoded_states[dn].cpu())
            reps["gw_reps"].append(fused_gw.cpu())
            total_samples += current_bs
            if total_samples % 1000 < current_bs : print(f"Processed {total_samples} samples") # Print roughly every 1000

    if total_samples == 0: raise ValueError(f"Could not generate samples from {split} dataset")
    print(f"Generated {total_samples} samples from {split} dataset")
    source_prefix = "gw_encoded" if use_gw_encoded else "latent"
    final_result = {}
    final_result[f"{domain_a}_{source_prefix}"] = torch.cat(reps[f"{domain_a}_orig"])[:n_samples]
    final_result[f"{domain_b}_{source_prefix}"] = torch.cat(reps[f"{domain_b}_orig"])[:n_samples]
    for dn in domain_names:
        final_result[f"{dn}_gw_encoded"] = torch.cat(reps[f"{dn}_gw_encoded"])[:n_samples]
        final_result[f"{dn}_decoded"] = torch.cat(reps[f"{dn}_decoded"])[:n_samples]
    final_result["gw_rep"] = torch.cat(reps["gw_reps"])[:n_samples]
    return final_result

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
                        
                        # Extract tensor from complex objects if needed
                        if hasattr(v, 'bert'):
                            processed_batch[domain_name] = v.bert.to(device)
                        elif isinstance(v, dict) and domain_name in v:
                            value = v[domain_name]
                            if hasattr(value, 'bert'):
                                processed_batch[domain_name] = value.bert.to(device)
                            elif hasattr(value, 'to') and callable(getattr(value, 'to')):
                                processed_batch[domain_name] = value.to(device)
                            else:
                                processed_batch[domain_name] = value
                        else:
                            # Move to device if it's a tensor, otherwise keep as-is
                            if hasattr(v, 'to') and callable(getattr(v, 'to')):
                                processed_batch[domain_name] = v.to(device)
                            else:
                                processed_batch[domain_name] = v
                    # Handle DomainDesc keys directly (new format)
                    elif hasattr(k, 'base') and hasattr(k, 'kind'):
                        # This is a DomainDesc object
                        if k.base == "v" and k.kind == "v":
                            # Raw images - use key "v" for backward compatibility
                            domain_key = "v"
                        elif k.base == "v" and k.kind == "v_latents":
                            # Latent representations - use key "v_latents"
                            domain_key = "v_latents"
                        elif k.base == "t" and k.kind == "t":
                            # Text data - use key "t"
                            domain_key = "t"
                        else:
                            # Other domains - use base_kind format
                            domain_key = f"{k.base}_{k.kind}" if k.kind != k.base else k.base
                        
                        # Extract tensor from complex objects if needed
                        if hasattr(v, 'bert'):
                            processed_batch[domain_key] = v.bert.to(device)
                        elif isinstance(v, dict) and k.base in v:
                            value = v[k.base]
                            if hasattr(value, 'bert'):
                                processed_batch[domain_key] = value.bert.to(device)
                            elif hasattr(value, 'to') and callable(getattr(value, 'to')):
                                processed_batch[domain_key] = value.to(device)
                            else:
                                processed_batch[domain_key] = value
                        else:
                            # Move to device if it's a tensor, otherwise keep as-is
                            if hasattr(v, 'to') and callable(getattr(v, 'to')):
                                processed_batch[domain_key] = v.to(device)
                            else:
                                processed_batch[domain_key] = v
                    else:
                        # Regular key, just move to device if it's a tensor
                        if hasattr(v, 'to') and callable(getattr(v, 'to')):
                            processed_batch[k] = v.to(device)
                        else:
                            # Handle non-tensor objects (like lists)
                            processed_batch[k] = v
    # Handle different batch formats
    elif isinstance(batch, list) and len(batch) > 0:
        # Handle list batches
        for item in batch:
            if isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(k, frozenset):
                        domain_name = next(iter(k))
                    elif hasattr(k, 'base') and hasattr(k, 'kind'):
                        # Handle DomainDesc objects in list format
                        if k.base == "v" and k.kind == "v":
                            domain_name = "v"
                        elif k.base == "v" and k.kind == "v_latents":
                            domain_name = "v_latents"
                        elif k.base == "t" and k.kind == "t":
                            domain_name = "t"
                        else:
                            domain_name = f"{k.base}_{k.kind}" if k.kind != k.base else k.base
                    else:
                        domain_name = k
                        
                    # Extract tensor from complex objects if needed
                    if hasattr(v, 'bert'):
                        processed_batch[domain_name] = v.bert.to(device)
                    elif isinstance(v, dict) and domain_name in v:
                        value = v[domain_name]
                        if hasattr(value, 'bert'):
                            processed_batch[domain_name] = value.bert.to(device)
                        elif hasattr(value, 'to') and callable(getattr(value, 'to')):
                            processed_batch[domain_name] = value.to(device)
                        else:
                            processed_batch[domain_name] = value
                    else:
                        # Move to device if it's a tensor, otherwise keep as-is
                        if hasattr(v, 'to') and callable(getattr(v, 'to')):
                            processed_batch[domain_name] = v.to(device)
                        else:
                            processed_batch[domain_name] = v
    elif isinstance(batch, dict):
        # Handle dictionary batches
        for k, v in batch.items():
            if isinstance(k, frozenset):
                domain_name = next(iter(k))
            elif hasattr(k, 'base') and hasattr(k, 'kind'):
                # Handle DomainDesc objects in dict format
                if k.base == "v" and k.kind == "v":
                    domain_name = "v"
                elif k.base == "v" and k.kind == "v_latents":
                    domain_name = "v_latents"
                elif k.base == "t" and k.kind == "t":
                    domain_name = "t"
                else:
                    domain_name = f"{k.base}_{k.kind}" if k.kind != k.base else k.base
            else:
                domain_name = k
                
            if hasattr(v, 'bert'):
                processed_batch[domain_name] = v.bert.to(device)
            elif isinstance(v, dict) and domain_name in v:
                value = v[domain_name]
                if hasattr(value, 'bert'):
                    processed_batch[domain_name] = value.bert.to(device)
                elif hasattr(value, 'to') and callable(getattr(value, 'to')):
                    processed_batch[domain_name] = value.to(device)
                else:
                    processed_batch[domain_name] = value
            else:
                # Move to device if it's a tensor, otherwise keep as-is
                if hasattr(v, 'to') and callable(getattr(v, 'to')):
                    processed_batch[domain_name] = v.to(device)
                else:
                    processed_batch[domain_name] = v
    
    # Apply domain-specific processing
    processed_result = processed_batch.copy()
    for domain_name, domain_input in processed_batch.items():
        # Fix shape for v_latents domain (common issue with extra dimensions)
        if domain_name == 'v_latents' and domain_input.dim() > 2:
            # Take only the first element along dimension 1 (mean vector)
            processed_result[domain_name] = domain_input[:, 0, :]
    
    return processed_result

def find_latest_model_checkpoints(base_dir: str, max_configs: Optional[int] = None) -> List[str]:
    epoch_pattern = re.compile(r'model_epoch_(\d+)')
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory {base_dir} does not exist"); return []
    
    target_config_dir = None
    for config_dir_path in Path(base_dir).glob("config_*_v_latents_0.4_t_0.6"): # Corrected glob pattern
        if config_dir_path.is_dir():
            target_config_dir = config_dir_path
            break
    if not target_config_dir: print("Warning: Could not find config for v_latents_0.4_t_0.6"); return []
    
    checkpoints = list(target_config_dir.glob("model_epoch_*.pt"))
    if not checkpoints: print(f"Warning: No checkpoints in {target_config_dir}"); return []
    
    checkpoint_epochs = []
    for cp in checkpoints:
        match = epoch_pattern.search(cp.name)
        if match: checkpoint_epochs.append((int(match.group(1)), cp))
    
    if checkpoint_epochs:
        latest_epoch, latest_checkpoint = max(checkpoint_epochs, key=lambda x: x[0])
        print(f"Found latest checkpoint for v_latents_0.4_t_0.6: {latest_checkpoint.name} (epoch {latest_epoch})")
        return [str(latest_checkpoint)]
    return []

def get_allowed_sources_and_targets(domain_name: str) -> Dict[str, List[str]]:
    allowed = {"sources": [], "targets": []}
    if domain_name == "v_latents":
        allowed["sources"] = ["v_latents_latent", "v_latents_encoded"]
        allowed["targets"] = ["gw_latent", "gw_decoded"]
    elif domain_name == "t":
        allowed["sources"] = ["t_latent", "t_encoded"]
        allowed["targets"] = ["gw_latent", "gw_decoded"]
    return allowed

def validate_source_target_config(
    source_config: Dict[str, str],
    target_config: str,
    domain_names: List[str]
) -> None:
    if len(domain_names) != 2: raise ValueError(f"Exactly 2 domains required, got {len(domain_names)}")
    if len(source_config) != 2: raise ValueError(f"Exactly 2 sources required, got {len(source_config)}")
    
    for domain, source in source_config.items():
        if domain not in domain_names: raise ValueError(f"Invalid domain in source config: {domain}")
        allowed_s = get_allowed_sources_and_targets(domain)['sources']
        if source not in allowed_s: raise ValueError(f"Invalid source for {domain}: {source}. Allowed: {allowed_s}")
    
    allowed_t = []
    for domain in domain_names: allowed_t.extend(get_allowed_sources_and_targets(domain)['targets'])
    if target_config not in allowed_t: raise ValueError(f"Invalid target: {target_config}. Allowed: {allowed_t}")

def load_domain_modules(configs: List[Dict[str, Any]], eval_mode: bool = True, device: Optional[str] = None) -> Dict[str, DomainModule]:
    """Load domain modules from configuration using shimmer_ssd."""
    try:
        # Try to import shimmer_ssd modules
        from shimmer_ssd.config import LoadedDomainConfig, DomainModuleVariant
        from shimmer_ssd.modules.domains.pretrained import load_pretrained_module
        HAS_SHIMMER_SSD = True
    except ImportError:
        HAS_SHIMMER_SSD = False
        print("Warning: shimmer_ssd not found. Using fallback domain modules.")
        return load_domain_modules_fallback(configs, eval_mode, device)
    
    if not HAS_SHIMMER_SSD:
        return load_domain_modules_fallback(configs, eval_mode, device)
    
    print("🔧 Loading domain modules using shimmer_ssd...")
    domain_modules = {}
    
    # Convert dict format to list format if needed
    configs_list = []
    if isinstance(configs, dict):
        for domain_name, config in configs.items():
            if "name" not in config:
                config = config.copy()  # Create a copy to avoid modifying the original
                config["name"] = domain_name
            configs_list.append(config)
    else:
        configs_list = configs
    
    for config in configs_list:
        # Handle domain type
        domain_type = config["domain_type"]
        domain_type_str = str(domain_type)
        
        if "." in domain_type_str:
            domain_type_str = domain_type_str.split(".")[-1]
        
        # Map domain type to variant
        if hasattr(domain_type, "value") and hasattr(domain_type.value, "kind"):
            domain_variant = domain_type
        else:
            if domain_type_str in ["v", "v_latents"]:
                domain_variant = DomainModuleVariant.v_latents
            elif domain_type_str == "t":
                domain_variant = DomainModuleVariant.t
            elif domain_type_str == "attr":
                domain_variant = DomainModuleVariant.attr
            else:
                raise ValueError(f"Unsupported domain type: {domain_type_str}")
        
        # Prepare domain config
        domain_config = LoadedDomainConfig(
            domain_type=domain_variant,
            checkpoint_path=config["checkpoint_path"],
            args=config.get("args", {})
        )
        
        # Load module
        domain_module = load_pretrained_module(domain_config)
        
        # Determine domain name
        if "name" in config:
            domain_name = config["name"]
        elif hasattr(domain_config.domain_type, "kind") and hasattr(domain_config.domain_type.kind.value, "kind"):
            domain_name = domain_config.domain_type.kind.value.kind
        else:
            domain_name = domain_type_str
        
        # Verify that the text domain has a projector
        if domain_name == 't':
            if hasattr(domain_module, 'projector'):
                print(f"Loaded domain module '{domain_name}' with pretrained projector")
            else:
                print(f"Warning: Text domain module does not have a projector!")
        else:
            print(f"Loaded domain module: {domain_name}")
        
        domain_modules[domain_name] = domain_module
    
    return domain_modules

def load_domain_modules_fallback(configs: List[Dict[str, Any]], eval_mode: bool = True, device: Optional[str] = None) -> Dict[str, DomainModule]:
    """
    THIS FUNCTION SHOULD NEVER BE CALLED - we want REAL modules only.
    """
    raise RuntimeError("❌ CRITICAL ERROR: Attempted to use fallback domain modules! This is not allowed - real shimmer modules required.")

class LRFinder:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: Callable, device: Optional[torch.device] = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_state = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
        self.optimizer_state = self.optimizer.state_dict()
        self.history = {'lr': [], 'loss': []}
        self.best_lr = None

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

    def range_test(self, train_loader: torch.utils.data.DataLoader, start_lr: float = 1e-7, end_lr: float = 10, num_iter: int = 1, step_mode: str = "exp", smooth_f: float = 0.05, diverge_th: float = 5.0):
        self.model.train()
        self.history = {'lr': [], 'loss': []}
        current_lr = start_lr
        for param_group in self.optimizer.param_groups: param_group['lr'] = current_lr
        avg_loss, best_loss = 0.0, float('inf')
        lr_step = (end_lr / start_lr) ** (1 / num_iter) if step_mode == "exp" else (end_lr - start_lr) / num_iter
        iterator = iter(train_loader)

        for iteration in range(num_iter):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(train_loader); batch = next(iterator)
            
            x1_batch, x2_batch, y_batch = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].to(self.device)
            
            # Handle different target formats
            if y_batch.dtype == torch.float and y_batch.dim() > 1 and y_batch.size(1) > 1:
                # One-hot encoded targets - convert to class indices
                y_batch = torch.argmax(y_batch, dim=1).long()
            elif y_batch.dim() > 1:
                # Multi-dimensional but not one-hot - squeeze and convert
                y_batch = y_batch.squeeze().long()
            else:
                # Already class indices - just ensure long type
                y_batch = y_batch.long()
            
            self.optimizer.zero_grad()
            
            # Model specific call for CEAlignmentInformation
            try:
                loss, _, _ = self.model(x1_batch, x2_batch, y_batch)
                loss.backward()
                self.optimizer.step()
            except Exception as e:
                print(f"Error in LR finder iteration {iteration}: {e}")
                print(f"Batch shapes: x1={x1_batch.shape}, x2={x2_batch.shape}, y={y_batch.shape}")
                print(f"y_batch dtype: {y_batch.dtype}, y_batch range: [{y_batch.min()}, {y_batch.max()}]")
                print(f"y_batch unique values: {torch.unique(y_batch)}")
                raise
            
            loss_value = loss.item()
            avg_loss = loss_value if iteration == 0 else smooth_f * loss_value + (1 - smooth_f) * avg_loss
            if iteration > 0 and avg_loss > diverge_th * best_loss: print(f"Loss diverged at lr={current_lr:.6f}"); break
            if avg_loss < best_loss: best_loss, self.best_lr = avg_loss, current_lr
            
            self.history['lr'].append(current_lr)
            self.history['loss'].append(avg_loss)
            current_lr = current_lr * lr_step if step_mode == "exp" else current_lr + lr_step
            for param_group in self.optimizer.param_groups: param_group['lr'] = current_lr
            if (iteration + 1) % 10 == 0: print(f"Iter {iteration+1}/{num_iter}: lr={current_lr:.2e}, loss={avg_loss:.4f}")

    def plot(self, skip_start=10, skip_end=5, log_lr=True, return_fig=False):
        import matplotlib.pyplot as plt # Import locally for plotting
        import numpy as np # Import locally
        lrs, losses = self.history["lr"], self.history["loss"]
        lrs, losses = np.array(lrs[skip_start:-skip_end if skip_end > 0 else len(lrs)]), np.array(losses[skip_start:-skip_end if skip_end > 0 else len(losses)])
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        axs[0].plot(lrs, losses, label='Loss', linewidth=2)
        if self.best_lr: 
            for ax_i in axs: ax_i.axvline(self.best_lr, linestyle='--', color='r', label=f'Best LR = {self.best_lr:.2e}')

        with np.errstate(divide='ignore', invalid='ignore'):
            xs_grad = np.log(lrs) if log_lr else lrs
            derivatives = np.gradient(losses, xs_grad)
            derivatives = np.where(np.isfinite(derivatives), derivatives, np.nan)
            from scipy.ndimage import gaussian_filter1d
            smooth_derivatives = gaussian_filter1d(derivatives, sigma=min(len(derivatives)//10+1,10)/4)
        
        axs[1].plot(lrs, smooth_derivatives, label='d(Loss)/d(log(LR))' if log_lr else 'd(Loss)/d(LR)', color='green', linewidth=2)
        axs[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        if len(smooth_derivatives) > 0:
            min_idx = np.nanargmin(smooth_derivatives)
            if min_idx < len(lrs):
                steepest_lr = lrs[min_idx]
                axs[1].plot(steepest_lr, smooth_derivatives[min_idx], 'ro', markersize=8, label=f'Steepest: {steepest_lr:.2e}')
                axs[0].axvline(steepest_lr, linestyle=':', color='orange', alpha=0.7, label=f'Steepest: {steepest_lr:.2e}')
        
        axs[0].set_title("LR Finder Results", fontsize=16); axs[0].set_ylabel("Loss", fontsize=14); axs[0].grid(True, alpha=0.3); axs[0].legend(fontsize=12)
        axs[1].set_title("Loss Change Rate", fontsize=16); axs[1].set_ylabel("d(Loss)/d(log(LR))" if log_lr else "d(Loss)/d(LR)", fontsize=14)
        axs[1].set_xlabel("Learning Rate" + (" (log scale)" if log_lr else ""), fontsize=14); axs[1].grid(True, alpha=0.3); axs[1].legend(fontsize=12)
        if log_lr: 
            for ax_i in axs: ax_i.set_xscale("log")
        plt.tight_layout()
        if self.best_lr: axs[0].text(0.02, 0.02, f"Suggested LR: {self.best_lr/10.0:.2e}\n(div by 10)", transform=axs[0].transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=12)
        if return_fig: return fig
        plt.show(); plt.close()

    def get_best_lr(self, factor=10.0): return self.best_lr / factor if self.best_lr else None

def find_optimal_lr(model, train_ds, batch_size: int = 256, start_lr: float = 1e-7, end_lr: float = 1.0, num_iter: int = 10, skip_start: int = 10, skip_end: int = 5, factor: float = 10.0, log_to_wandb: bool = False, seed: int = 42, return_finder: bool = False) -> Union[float, Tuple[float, LRFinder]]:
    import matplotlib.pyplot as plt # Local import for when plot is generated
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    
    # Assuming MultimodalDataset is defined in .data or imported appropriately
    # from .data import MultimodalDataset 
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0, pin_memory=torch.cuda.is_available(), drop_last=False)
    optimizer = torch.optim.Adam(model.align.parameters(), lr=start_lr) # Model specific: model.align.parameters()
    model_device = next(model.parameters()).device
    
    # Criterion for CEAlignmentInformation model (not actually used by LRFinder.range_test)
    def criterion_fn(x1, x2, y): return model(x1, x2, y)[0] 

    lr_finder_instance = LRFinder(model=model, optimizer=optimizer, criterion=criterion_fn, device=model_device)
    print(f"Running LR finder from {start_lr:.2e} to {end_lr:.2e} over {num_iter} iterations")
    
    try:
        lr_finder_instance.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
    except Exception as e:
        print(f"❌ Error in LR finder range_test: {e}")
        # Check the first batch to debug the issue
        try:
            first_batch = next(iter(train_loader))
            print(f"First batch info:")
            print(f"  batch[0] (x1): shape={first_batch[0].shape}, dtype={first_batch[0].dtype}")
            print(f"  batch[1] (x2): shape={first_batch[1].shape}, dtype={first_batch[1].dtype}")
            print(f"  batch[2] (y): shape={first_batch[2].shape}, dtype={first_batch[2].dtype}")
            if first_batch[2].dim() > 1:
                print(f"  y unique values: {torch.unique(first_batch[2])}")
            else:
                print(f"  y range: [{first_batch[2].min()}, {first_batch[2].max()}]")
        except Exception as debug_e:
            print(f"Could not debug first batch: {debug_e}")
        raise
    fig = lr_finder_instance.plot(skip_start=skip_start, skip_end=skip_end, return_fig=True)
    
    # Assuming HAS_WANDB and wandb are defined globally
    global HAS_WANDB, wandb
    if log_to_wandb and HAS_WANDB and wandb.run is not None:
        try: wandb.log({"lr_finder/loss_vs_lr": wandb.Image(fig)})
        except Exception as e: print(f"Warning: Could not log LR finder plot to wandb: {e}")
    plt.close(fig)
    
    best_lr_val = lr_finder_instance.get_best_lr(factor=factor)
    print(f"Suggested learning rate: {best_lr_val:.2e}")
    lr_finder_instance.reset()
    # Update optimizer with new learning rate is usually done by the caller, not here.
    # for pg in optimizer.param_groups: pg["lr"] = best_lr_val 
    
    return (best_lr_val, lr_finder_instance) if return_finder else best_lr_val

# Additional imports for plotting
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d

# Try to import wandb, but make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

def generate_samples_from_dataset(
    model,  # Changed from GWModuleConfigurableFusion to avoid import issues
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
    
    try:
        import matplotlib.pyplot as plt
        import scipy.stats as stats
        from scipy.ndimage import gaussian_filter1d
    except ImportError as e:
        print(f"Warning: Could not import required plotting libraries: {e}")
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
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
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
            try:
                import wandb
                wandb.log({"plots/pid_vs_weight_ratio": wandb.Image(ratio_plot_path)})
            except:
                pass
        
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
            try:
                import wandb
                wandb.log({"plots/pid_stacked_bars": wandb.Image(stacked_plot_path)})
            except:
                pass
        
        plt.close()
        
        # 3. Create correlation plot of weight ratio vs. PID measures
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation coefficients
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
            try:
                import wandb
                wandb.log({"plots/pid_correlation": wandb.Image(corr_plot_path)})
            except:
                pass
        
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
            try:
                import wandb
                wandb.log({"plots/total_information": wandb.Image(total_info_path)})
            except:
                pass
        
        plt.close()
        
        print(f"Plots saved to {plots_dir}")

# Stub for the missing plot_stacked_pid function that was called but not defined
def plot_stacked_pid(results: List[Dict], output_dir: str, wandb_run=None):
    """
    Stub for plot_stacked_pid function that was called but not defined in original code.
    This function was referenced in analyze_multiple_models_from_list but never implemented.
    """
    print("Warning: plot_stacked_pid called but not implemented in original code. Skipping.")
    pass 

def log_extended_clustering_metrics(predictions, labels, cluster_method, wandb_prefix=None):
    """
    Log extended clustering metrics for wandb visualization.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    import numpy as np
    
    metrics = {}
    
    try:
        # Only compute if we have actual clustering data
        if len(np.unique(predictions)) > 1 and len(np.unique(labels)) > 1:
            # Clustering quality metrics
            ari = adjusted_rand_score(labels, predictions)
            nmi = normalized_mutual_info_score(labels, predictions)
            
            metrics.update({
                'adjusted_rand_index': ari,
                'normalized_mutual_info': nmi,
                'num_predicted_clusters': len(np.unique(predictions)),
                'num_true_clusters': len(np.unique(labels)),
            })
            
            # Log to wandb if available
            if wandb_prefix and HAS_WANDB:
                try:
                    import wandb
                    if wandb.run:
                        wandb_logs = {f"{wandb_prefix}/clustering/{k}": v for k, v in metrics.items()}
                        wandb.run.log(wandb_logs)
                except:
                    pass
    
    except Exception as e:
        print(f"Warning: Could not compute clustering metrics: {e}")
    
    return metrics 

def load_real_data_files(domain_names: List[str], n_samples: int, device: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load real data from pid_analysis_data/ directory files.
    
    Args:
        domain_names: List of domain names (should be ['v_latents', 't'])
        n_samples: Number of samples to load
        device: Device to load tensors to
        
    Returns:
        Dictionary with real data or None if not available
    """
    if len(domain_names) != 2:
        print(f"Warning: Expected 2 domains, got {len(domain_names)}")
        return None
        
    # Expected domain names
    expected_domains = set(['v_latents', 't'])
    if set(domain_names) != expected_domains:
        print(f"Warning: Expected domains {expected_domains}, got {set(domain_names)}")
        return None
    
    # Data file paths - look in root directory (2 levels up from this file)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(root_dir, "pid_analysis_data")
    vision_file = os.path.join(data_dir, "pre_fusion_vision_latents.pt")
    text_file = os.path.join(data_dir, "pre_fusion_text_latents.pt") 
    fused_file = os.path.join(data_dir, "fused_latents.pt")
    
    print(f"🔍 Looking for data files in: {data_dir}")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [vision_file, text_file, fused_file]):
        missing = [f for f in [vision_file, text_file, fused_file] if not os.path.exists(f)]
        print(f"Missing real data files: {missing}")
        return None
    
    try:
        # Load the data files
        print(f"📂 Loading real data from {data_dir}/...")
        vision_data = torch.load(vision_file, map_location=device)
        text_data = torch.load(text_file, map_location=device)  
        fused_data = torch.load(fused_file, map_location=device)
        
        # Ensure we have enough samples
        min_samples = min(len(vision_data), len(text_data), len(fused_data))
        
        if min_samples < n_samples:
            print(f"⚠️  Warning: Requested {n_samples} samples, but only {min_samples} available in cached files")
            print(f"🔄 Cached files insufficient - will use data_module to generate fresh samples")
            return None
        
        actual_samples = n_samples
        
        # Create the result dictionary in the expected format
        result = {}
        
        # Map domain names to the correct data
        for domain_name in domain_names:
            if domain_name == 'v_latents':
                # Use vision latent data as source
                result[f"{domain_name}_latent"] = vision_data[:actual_samples]
                print(f"✅ Loaded {actual_samples} vision latent samples (shape: {vision_data[:actual_samples].shape})")
            elif domain_name == 't':
                # Use text latent data as source  
                result[f"{domain_name}_latent"] = text_data[:actual_samples]
                print(f"✅ Loaded {actual_samples} text latent samples (shape: {text_data[:actual_samples].shape})")
        
        # Add the fused representation as the target
        result["gw_rep"] = fused_data[:actual_samples]
        print(f"✅ Loaded {actual_samples} fused latent samples (shape: {fused_data[:actual_samples].shape})")
        
        print(f"🎯 Final data keys: {list(result.keys())}")
        return result
        
    except Exception as e:
        print(f"❌ Error loading real data files: {e}")
        import traceback
        traceback.print_exc()
        return None 