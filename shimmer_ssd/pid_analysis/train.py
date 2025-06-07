import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import TensorDataset
import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
import json
from datetime import datetime

# Imports for metrics
from sklearn.metrics import precision_recall_fscore_support, jaccard_score, precision_recall_curve
from scipy.stats import spearmanr, kendalltau

try:
    from .utils import find_optimal_lr
    print("✅ Successfully imported find_optimal_lr from utils")
except ImportError as e:
    print(f"❌ Failed to import find_optimal_lr from utils: {e}")
    print("This is a critical error - LR finder functionality will not work!")
    # Create a dummy function to prevent crashes
    def find_optimal_lr(*args, **kwargs):
        print("⚠️  find_optimal_lr not available - using default learning rate")
        return 1e-3

try:
    import wandb
    HAS_WANDB = True
    print("✅ wandb available")
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb not available")

# Global device and compile settings
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {global_device}")

from .models import (
    Discrim, 
    CEAlignment, 
    CEAlignmentInformation, 
    PretrainedDiscrim, 
    PretrainedJointDiscrim,
    global_device
)

from .data import MultimodalDataset # Assuming data.py is in the same directory

# Caching functions for discriminators
def save_discriminator_with_metadata(discriminator, cache_path: Path, discrim_type: str, input_dim: int, hidden_dim: int, layers: int, 
                                   model_path: str, model_name: str, domain_name: str, num_clusters: int, 
                                   discrim_epochs: int, n_samples: int, cluster_method_discrim: str, 
                                   use_compile_torch: bool,
                                   joint_discrim_hidden_dim: int = None, joint_discrim_layers: int = None,
                                   x1_dim: int = None, x2_dim: int = None):
    """Save discriminator with comprehensive metadata and validation."""
    try:
        # Save the discriminator model
        torch.save(discriminator, cache_path)
        print(f"💾 Cached {discrim_type} to: {cache_path.name}")
        
        # Save comprehensive metadata
        metadata = {
            "discriminator_type": discrim_type,
            "model_path": str(model_path),
            "model_name": model_name,
            "domain_name": domain_name,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "num_clusters": num_clusters,
            "discrim_epochs": discrim_epochs,
            "n_samples": n_samples,
            "cluster_method": cluster_method_discrim,
            "use_compile": use_compile_torch,
            "generated_timestamp": str(datetime.now()),
            "architecture_summary": f"input({input_dim}) -> hidden({hidden_dim}) -> output({num_clusters})",
            "activation": "relu"
        }
        
        # Add joint discriminator specific metadata if provided
        if joint_discrim_hidden_dim is not None:
            metadata["joint_discrim_hidden_dim"] = joint_discrim_hidden_dim
        if joint_discrim_layers is not None:
            metadata["joint_discrim_layers"] = joint_discrim_layers
        if x1_dim is not None:
            metadata["x1_dim"] = x1_dim
        if x2_dim is not None:
            metadata["x2_dim"] = x2_dim
        
        # Save metadata
        metadata_path = cache_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Saved {discrim_type} metadata to: {metadata_path.name}")
        
    except Exception as e:
        print(f"⚠️  Failed to cache {discrim_type}: {e}")

def load_discriminator_with_validation(cache_path: Path, discrim_type: str, expected_input_dim: int, expected_hidden_dim: int, expected_layers: int):
    """Load discriminator with validation of architecture parameters."""
    try:
        # Check if metadata exists and validate
        metadata_path = cache_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate critical parameters
            validation_errors = []
            if metadata.get('input_dim') != expected_input_dim:
                validation_errors.append(f"input_dim mismatch: cached={metadata.get('input_dim')}, expected={expected_input_dim}")
            if metadata.get('hidden_dim') != expected_hidden_dim:
                validation_errors.append(f"hidden_dim mismatch: cached={metadata.get('hidden_dim')}, expected={expected_hidden_dim}")
            if metadata.get('layers') != expected_layers:
                validation_errors.append(f"layers mismatch: cached={metadata.get('layers')}, expected={expected_layers}")
            
            if validation_errors:
                print(f"⚠️  {discrim_type} validation failed: {'; '.join(validation_errors)}")
                print(f"   🔄 Will retrain {discrim_type} with new parameters")
                return None
            else:
                print(f"✅ {discrim_type} metadata validation passed")
        
        # Load the discriminator
        discriminator = torch.load(cache_path, map_location=str(global_device), weights_only=False)
        discriminator.to(global_device)
        print(f"✅ Successfully loaded cached {discrim_type}")
        return discriminator
        
    except Exception as e:
        print(f"⚠️  Failed to load cached {discrim_type}: {e}, training new one...")
        return None

def train_discrim(model, loader, optimizer, data_type, num_epoch=40, wandb_prefix=None, use_compile=True, cluster_method='gmm', enable_extended_metrics=False):
    """Train a Discrim on (X → Y). data_type tells which fields of the batch are features/labels."""
    model.train()
    model.to(global_device) # Ensure model is on the correct device

    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("🚀 Applied torch.compile optimization to discriminator")
        except Exception as e:
            print(f"Warning: torch.compile failed for discriminator: {e}")

    prev_probs_epoch = None #TODO what is this?

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        all_logits_epoch = []
        all_targets_epoch = []
        
        # Initialize metrics to NaN or empty
        ce_loss_metric, kl_div_metric, jaccard_metric = np.nan, np.nan, np.nan
        precision_micro, recall_micro, f1_micro = np.nan, np.nan, np.nan
        entropy_mean_metric, rho, tau, brier_metric, ece_metric = np.nan, np.nan, np.nan, np.nan, np.nan
        top_k_accuracy_epoch = np.nan
        one_hot_epoch = np.array([])
        entropies_epoch = np.array([])
        rel_diag_mean_predicted_value = np.array([])
        rel_diag_fraction_of_positives = np.array([])
        #TODO find out what these are for and if there is any pruporse in them
        
        for batch in loader:
            # Use the data_type pattern to extract features and labels
            xs = [batch[i].float().to(global_device) for i in data_type[0]]
            y_batch_original = batch[data_type[1][0]].to(global_device)
            #TODO check if this is correct
            
            # Forward pass
            logits = model(*xs)
            
            # Detect label type from the actual data, not cluster_method parameter
            #TODO check if this is correct, i.e. is y_batch_original post-clustering?
            if y_batch_original.dim() > 1 and y_batch_original.size(1) > 1:
                # Soft labels (GMM) - use KL divergence #TODO check if this makes sense, the use of KL and CE for dist and discrete
                y_for_loss = y_batch_original # Soft labels
                log_q = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_q, y_for_loss, reduction='batchmean')
                all_targets_epoch.append(y_for_loss.argmax(dim=1).cpu())
            else:
                # Hard labels (k-means) - use CrossEntropy
                y_for_loss = y_batch_original.long()
                if y_for_loss.dim() > 1: y_for_loss = y_for_loss.squeeze(-1) if y_for_loss.size(1) == 1 else y_for_loss[:,0]
                
                if hasattr(model, 'module') and hasattr(model.module, 'mlp') and model.module.mlp: num_classes = model.module.mlp[-1].out_features
                elif hasattr(model, 'mlp') and model.mlp: num_classes = model.mlp[-1].out_features
                else: num_classes = logits.shape[-1]

                if y_for_loss.min() < 0 or y_for_loss.max() >= num_classes: 
                    y_for_loss = torch.clamp(y_for_loss, 0, num_classes-1)
                loss = F.cross_entropy(logits, y_for_loss)
                all_targets_epoch.append(y_for_loss.cpu())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store batch results
            epoch_loss += loss.item()
            all_logits_epoch.append(logits.detach().cpu())
        
        # Compute epoch metrics
        avg_loss = epoch_loss / len(loader) if len(loader) > 0 else np.nan
        all_logits_epoch = torch.cat(all_logits_epoch, dim=0)
        all_targets_epoch = torch.cat(all_targets_epoch, dim=0)
        all_probs_epoch = F.softmax(all_logits_epoch, dim=1)
        
        # Basic accuracy computation (always calculated)
        probs_np = all_probs_epoch.numpy()
        targets_np = all_targets_epoch.numpy()
        preds_np = probs_np.argmax(axis=1)
        accuracy_epoch = np.mean(preds_np == targets_np) if len(preds_np) > 0 else np.nan
        
        # Basic metrics that are always calculated
        metrics = {
            "loss": avg_loss,
            "epoch": epoch,
            "accuracy": accuracy_epoch
        }
        
        # Extended metrics if enabled
        if enable_extended_metrics:
            # Compute extended metrics
            try:
                # Convert to numpy for sklearn metrics
                probs_np = all_probs_epoch.numpy()
                targets_np = all_targets_epoch.numpy()
                preds_np = probs_np.argmax(axis=1)
                
                # Multi-class metrics
                precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                    targets_np, preds_np, average='micro'
                )
                
                # Jaccard score
                jaccard_metric = jaccard_score(targets_np, preds_np, average='micro')
                
                # Entropy and calibration
                entropies = -(probs_np * np.log(probs_np + 1e-8)).sum(axis=1)
                entropy_mean = entropies.mean()
                
                # Spearman and Kendall correlations between entropy and accuracy
                accuracies = (preds_np == targets_np).astype(float)
                rho, _ = spearmanr(entropies, accuracies)
                tau, _ = kendalltau(entropies, accuracies)
                
                # Brier score
                n_classes = probs_np.shape[1]
                one_hot = np.zeros((len(targets_np), n_classes))
                one_hot[np.arange(len(targets_np)), targets_np] = 1
                brier_metric = np.mean((probs_np - one_hot) ** 2)
                
                # Expected Calibration Error
                confidences = probs_np.max(axis=1)
                ece_metric = np.abs(confidences.mean() - accuracies.mean())
                
                # Top-k accuracy
                k = min(5, n_classes)
                top_k_preds = all_logits_epoch.topk(k, dim=1)[1]
                top_k_correct = (top_k_preds == all_targets_epoch.unsqueeze(1)).any(dim=1)
                top_k_accuracy = top_k_correct.float().mean().item()
                
                # Reliability diagram data
                n_bins = 10
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                rel_diag_accuracies = []
                rel_diag_confidences = []
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                    if any(in_bin):
                        rel_diag_accuracies.append(accuracies[in_bin].mean())
                        rel_diag_confidences.append(confidences[in_bin].mean())
                
                rel_diag_mean_predicted_value = np.array(rel_diag_confidences)
                rel_diag_fraction_of_positives = np.array(rel_diag_accuracies)
                
                # Update metrics dictionary with extended metrics
                metrics.update({
                    "precision_micro": precision_micro,
                    "recall_micro": recall_micro,
                    "f1_micro": f1_micro,
                    "jaccard": jaccard_metric,
                    "entropy_mean": entropy_mean,
                    "spearman_rho": rho,
                    "kendall_tau": tau,
                    "brier_score": brier_metric,
                    "ece": ece_metric,
                    "top_k_accuracy": top_k_accuracy
                })
                
                # Add reliability diagram data if available
                if len(rel_diag_mean_predicted_value) > 0 and len(rel_diag_fraction_of_positives) > 0:
                    metrics.update({
                        "rel_diag_mean_predicted_value": rel_diag_mean_predicted_value.tolist(),
                        "rel_diag_fraction_of_positives": rel_diag_fraction_of_positives.tolist()
                    })
            
            except Exception as e:
                print(f"Warning: Error computing extended metrics: {e}")
                # Keep the basic metrics even if extended metrics fail
        
        # Log metrics
        if HAS_WANDB and wandb_prefix and wandb.run:
            wandb.log({f"{wandb_prefix}/{k}": v for k, v in metrics.items()})
        
        # Print progress
        print(f"Epoch {epoch:3d}/{num_epoch:3d} - Loss: {avg_loss:.4f} - Acc: {accuracy_epoch:.4f}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return model

def train_ce_alignment(model: CEAlignmentInformation, loader: DataLoader, optimizer_class: Callable[..., torch.optim.Optimizer], 
                       num_epoch=10, wandb_prefix: Optional[str]=None, step_offset=0, use_compile=True, 
                       test_mode=False, max_test_examples=3000, auto_find_lr=False, 
                       lr_finder_steps=200, lr_start=1e-7, lr_end=1.0):
    
    print(f"🔮 train_ce_alignment: Starting with wandb_prefix='{wandb_prefix}', step_offset={step_offset}")
    print(f"   HAS_WANDB: {HAS_WANDB}")
    if HAS_WANDB:
        print(f"   wandb.run: {wandb.run}")
        print(f"   wandb.run is not None: {wandb.run is not None}")
    
    # Define custom metrics for CE alignment to allow out-of-order logging
    if HAS_WANDB and wandb_prefix is not None and wandb.run is not None:
        print(f"🎯 Defining custom wandb metrics for CE alignment to allow out-of-order logging...")
        
        # Use glob pattern to set all CE alignment metrics to use custom step - cleaner approach
        # This sets all metrics with the wandb_prefix to use the custom step metric
        wandb.define_metric(f"{wandb_prefix}/*", step_metric=f"{wandb_prefix}/ce_step")
        
        print(f"   ✅ Defined glob pattern '{wandb_prefix}/*' to use custom step metric '{wandb_prefix}/ce_step'")
        print(f"   📖 Reference: https://wandb.me/define-metric")
    
    # model is already on global_device (expected to be moved by caller, e.g. critic_ce_alignment)
    # For safety, can assert: assert next(model.parameters()).device == global_device
    model.train()

    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("🚀 Applied torch.compile optimization to CE alignment model")
        except Exception as e:
            print(f"Warning: torch.compile failed for CE alignment model: {e}")
    
    # Limit dataset size if in test mode #TODO there is a test mode???
    if test_mode and max_test_examples > 0:
        print(f"🔬 Test mode: Using only {max_test_examples} examples")
        n_total = len(loader.dataset)
        if max_test_examples < n_total:
            indices = torch.randperm(n_total)[:max_test_examples]
            subset_dataset = Subset(loader.dataset, indices)
            loader = DataLoader(
                subset_dataset,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=loader.num_workers,
                pin_memory=loader.pin_memory
            )
    
    # Find optimal learning rate if requested
    if auto_find_lr:
        print("🔍 Finding optimal learning rate...")
        try:
            # Create a new optimizer instance for LR finder
            optimizer = optimizer_class(model.parameters(), lr=lr_start)
            
            # Run LR finder
            optimal_lr = find_optimal_lr(
                model=model,
                train_ds=loader.dataset,
                batch_size=loader.batch_size,
                start_lr=lr_start,
                end_lr=lr_end,
                num_iter=lr_finder_steps,
                log_to_wandb=HAS_WANDB and wandb.run is not None,
                seed=42
            )
            print(f"✨ Found optimal learning rate: {optimal_lr:.2e}")
            
        except Exception as e:
            print(f"❌ Error finding optimal learning rate: {e}")
            print("⚠️  Using default learning rate of 1e-3")
            optimal_lr = 1e-3
    else:
        optimal_lr = 1e-3
        print(f"Using default learning rate: {optimal_lr}")
    
    # Create optimizer with found/default learning rate
    optimizer = optimizer_class(model.parameters(), lr=optimal_lr)
    
    # Training loop
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        epoch_pid_vals = []
        
        for batch_idx, batch_data in enumerate(loader):
            x1_batch, x2_batch, y_batch_orig = batch_data
            x1_batch, x2_batch, y_batch_orig = x1_batch.to(global_device), x2_batch.to(global_device), y_batch_orig.to(global_device)
            
            # Forward pass
            loss, pid_vals, _ = model(x1_batch, x2_batch, y_batch_orig)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store batch results
            epoch_loss += loss.item()
            epoch_pid_vals.append(pid_vals.detach().cpu().numpy())
            
            # Log step metrics if wandb is enabled
            if HAS_WANDB and wandb_prefix and wandb.run:
                ce_step = step_offset + epoch * len(loader) + batch_idx
                wandb.log({
                    f"{wandb_prefix}/ce_step": ce_step,
                    f"{wandb_prefix}/batch_loss": loss.item(),
                    f"{wandb_prefix}/batch_redundancy": pid_vals[0].item(),
                    f"{wandb_prefix}/batch_unique1": pid_vals[1].item(),
                    f"{wandb_prefix}/batch_unique2": pid_vals[2].item(),
                    f"{wandb_prefix}/batch_synergy": pid_vals[3].item()
                })
        
        # Compute epoch metrics
        avg_loss = epoch_loss / len(loader) if len(loader) > 0 else np.nan
        avg_pid_vals = np.mean(epoch_pid_vals, axis=0) if epoch_pid_vals else np.array([np.nan] * 4)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{num_epoch:3d} - Loss: {avg_loss:.4f}, "
              f"PID [R={avg_pid_vals[0]:.4f}, U1={avg_pid_vals[1]:.4f}, "
              f"U2={avg_pid_vals[2]:.4f}, S={avg_pid_vals[3]:.4f}]")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return model

def create_pretrained_discriminators(
    x1, x2, labels, num_labels,
    model, domain_names,
    discrim_hidden_dim=64, discrim_layers=2, 
    joint_discrim_hidden_dim=None, joint_discrim_layers=None,
    activation='relu'
):
    """
    Create discriminators using pretrained frozen encoders with additional softmax layers.
    
    Args:
        x1: First domain features, shape [batch_size, x1_dim]
        x2: Second domain features, shape [batch_size, x2_dim]
        labels: Integer labels, shape [batch_size]
        num_labels: Number of possible label values
        model: The loaded model with gw_encoders
        domain_names: List of domain names for the two modalities
        discrim_hidden_dim: Hidden dimension for individual classifier networks
        discrim_layers: Number of layers in individual classifier networks (typically 2)
        joint_discrim_hidden_dim: Hidden dimension for joint classifier network (defaults to discrim_hidden_dim)
        joint_discrim_layers: Number of layers in joint classifier network (defaults to discrim_layers)
        activation: Activation function to use
    
    Returns:
        Tuple of (d1, d2, d12) discriminators
    """
    from .models import PretrainedDiscrim, PretrainedJointDiscrim
    
    # Validate inputs
    if len(domain_names) < 2:
        raise ValueError(f"Need at least 2 domain names, got {len(domain_names)}")
    
    # Get domain names for vision and text
    domain1_name, domain2_name = domain_names[0], domain_names[1]
    
    # Set default values for joint discriminator parameters if None
    if joint_discrim_hidden_dim is None:
        joint_discrim_hidden_dim = discrim_hidden_dim
    if joint_discrim_layers is None:
        joint_discrim_layers = discrim_layers
    
    # Access the pretrained encoders from the loaded model
    if not hasattr(model, 'gw_encoders'):
        raise ValueError("Model does not have gw_encoders attribute")
    
    available_domains = list(model.gw_encoders.keys())
    print(f"🔍 Available domains in model.gw_encoders: {available_domains}")
    
    # Security check: ensure we're not using the same encoder twice
    if domain1_name == domain2_name:
        raise ValueError(f"Cannot use the same domain '{domain1_name}' for both modalities")
    
    # Validate domain availability
    if domain1_name not in model.gw_encoders:
        raise ValueError(f"Domain '{domain1_name}' not found in model.gw_encoders. Available: {available_domains}")
    
    if domain2_name not in model.gw_encoders:
        raise ValueError(f"Domain '{domain2_name}' not found in model.gw_encoders. Available: {available_domains}")
    
    # Get the actual pretrained encoders from the loaded model checkpoint
    encoder1 = model.gw_encoders[domain1_name]
    encoder2 = model.gw_encoders[domain2_name]
    
    # Additional security check: ensure we have different encoder instances
    if encoder1 is encoder2:
        raise ValueError(f"Security Error: encoder1 and encoder2 are the same instance! Domain names: {domain1_name}, {domain2_name}")
    
    print(f"🔧 Creating pretrained discriminators with frozen GW encoders...")
    print(f"   Domain 1 ({domain1_name}): {type(encoder1).__name__} with {sum(p.numel() for p in encoder1.parameters())} parameters")
    print(f"   Domain 2 ({domain2_name}): {type(encoder2).__name__} with {sum(p.numel() for p in encoder2.parameters())} parameters")
    print(f"   Individual classifiers: {discrim_hidden_dim} hidden, {discrim_layers} layers → {num_labels} classes")
    print(f"   Joint classifier: {joint_discrim_hidden_dim} hidden, {joint_discrim_layers} layers → {num_labels} classes")
    
    # Validate that encoders are different by checking their parameters
    encoder1_params = torch.cat([p.flatten() for p in encoder1.parameters()])
    encoder2_params = torch.cat([p.flatten() for p in encoder2.parameters()])
    
    if encoder1_params.shape == encoder2_params.shape:
        # Only compare if they have the same structure
        param_comparison = torch.allclose(encoder1_params, encoder2_params, atol=1e-6) #TODO check that this works with allclose
        if param_comparison:
            print(f"⚠️  Warning: Encoders for {domain1_name} and {domain2_name} have identical parameters!")
        else:
            print(f"✅ Confirmed: Encoders for {domain1_name} and {domain2_name} have different parameters")
    else:
        # Different architectures, so definitely different
        print(f"✅ Confirmed: Encoders for {domain1_name} and {domain2_name} have different architectures ({len(encoder1_params)} vs {len(encoder2_params)} parameters)")
    
    # Create individual discriminators with pretrained encoders
    d1 = PretrainedDiscrim(
        pretrained_encoder=encoder1,
        num_labels=num_labels,
        hidden_dim=discrim_hidden_dim,
        layers=discrim_layers,
        activation=activation
    ).to(global_device)
    
    d2 = PretrainedDiscrim(
        pretrained_encoder=encoder2,
        num_labels=num_labels,
        hidden_dim=discrim_hidden_dim,
        layers=discrim_layers,
        activation=activation
    ).to(global_device)
    
    # Create joint discriminator with separate parameters
    d12 = PretrainedJointDiscrim(
        pretrained_encoder1=encoder1,
        pretrained_encoder2=encoder2,
        num_labels=num_labels,
        hidden_dim=joint_discrim_hidden_dim,  # Use joint-specific hidden dim
        layers=joint_discrim_layers,          # Use joint-specific layers
        activation=activation
    ).to(global_device)
    
    print(f"✅ Created pretrained discriminators:")
    print(f"   d1 ({domain1_name}): PretrainedDiscrim with {discrim_layers}-layer classifier")
    print(f"   d2 ({domain2_name}): PretrainedDiscrim with {discrim_layers}-layer classifier")
    print(f"   d12 (joint): PretrainedJointDiscrim with {joint_discrim_layers}-layer classifier")
    
    return d1, d2, d12

def critic_ce_alignment(
    x1, x2, labels, num_labels,
    train_ds, test_ds,
    discrim_1=None, discrim_2=None, discrim_12=None,
    learned_discrim=True, shuffle=True,
    discrim_epochs=40, ce_epochs=10, 
    wandb_enabled=False, model_name=None,
    discrim_hidden_dim=64, discrim_layers=5, joint_discrim_layers=None, joint_discrim_hidden_dim=None,
    use_compile=True, test_mode=False, max_test_examples=3000, 
    auto_find_lr=False, lr_finder_steps=200, 
    lr_start=1e-7, lr_end=1.0,
    enable_extended_metrics=True,
    run_critic_ce_direct=False,
    force_retrain_discriminators=False,
    model_type="complete_MLP",
    model=None,
    domain_names=None
):
    """
    Core function for Partial Information Decomposition via Conditional Entropy alignment.
    #TODO update naming here too, and everywhere in train.
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
        joint_discrim_layers: Number of layers in joint discriminator networks
        use_compile: Whether to use torch.compile for model optimization
        test_mode: Whether to run in test mode with limited examples
        max_test_examples: Maximum number of examples to process in test mode
        auto_find_lr: Whether to automatically find the optimal learning rate
        lr_finder_steps: Number of iterations for the learning rate finder
        lr_start: Start learning rate for the finder
        lr_end: End learning rate for the finder
        enable_extended_metrics: Whether to enable extended metrics for training
        run_critic_ce_direct: Whether to run the critic_ce_direct function
        force_retrain_discriminators: Whether to force retraining of discriminators
        model_type: Type of model to use for training
        model: The loaded model with gw_encoders
        domain_names: List of domain names for the two modalities
    
    Returns:
        Tuple of (PID components, alignments, models)
    """
    
    # ========================================
    # 🎯 SET DETERMINISTIC SEEDS FOR CACHING
    # ========================================
    # CRITICAL: Set all random seeds at the very beginning to ensure
    # deterministic data generation and consistent cache hashes
    print("🎯 Setting deterministic seeds for reproducible caching...")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import numpy as np
    np.random.seed(42)
    import random
    random.seed(42)
    # Also set Python's hash seed for string hashing (affects dict/set ordering)
    os.environ['PYTHONHASHSEED'] = '42'
    print(f"   ✅ Set all seeds to 42 for reproducible data generation")
    
    # Ensure input tensors are on the global_device
    x1 = x1.to(global_device)
    x2 = x2.to(global_device)
    labels = labels.to(global_device)
    
    print(f"critic_ce_alignment: x1.shape={x1.shape}, x2.shape={x2.shape}, labels.shape={labels.shape}")
    print(f"Using device: {global_device}")
    
    # Create train/test datasets from the full data
    #TODO check that we are not shuffling the visual and text data out of their synchrony
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=shuffle, 
                         num_workers=0, pin_memory=False)  # Reduced complexity
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, #TODO we are never testing the networks on test data
                        num_workers=0, pin_memory=False)
    
    # Step 1: Train or use discriminators
    if learned_discrim:
        # Check model type and create discriminators accordingly
        if model_type == "pretrained_encoders":
            print(f"🧠 Using pretrained encoders for discriminators")
            
            # Validate required parameters for pretrained encoders
            if model is None:
                raise ValueError("model parameter is required when model_type='pretrained_encoders'")
            if domain_names is None:
                raise ValueError("domain_names parameter is required when model_type='pretrained_encoders'")
            if len(domain_names) < 2:
                raise ValueError("domain_names must contain at least 2 domain names")
            
            # Create pretrained discriminators (no training needed, just classifier heads)
            d1, d2, d12 = create_pretrained_discriminators(
                x1=x1, x2=x2, labels=labels, num_labels=num_labels,
                model=model, domain_names=domain_names,
                discrim_hidden_dim=discrim_hidden_dim, 
                discrim_layers=discrim_layers,  # For pretrained, this will be 2-layer MLP
                joint_discrim_hidden_dim=joint_discrim_hidden_dim,  # Pass joint parameters
                joint_discrim_layers=joint_discrim_layers,          # Pass joint parameters
                activation='relu'
            )
            
            # Train only the classifier heads
            print(f"🔄 Training classifier heads for pretrained discriminators...")
            
            # Train individual discriminators' classifier heads
            # Note: classifier is initialized lazily in the first forward pass, so we don't check for it here
            print(f"\n🧠 TRAINING PRETRAINED DISCRIMINATOR 1 ({domain_names[0]}) - {discrim_epochs} epochs")
            print(f"   Input: {domain_names[0]} features → Labels (via pretrained encoder)")
            print(f"   Architecture: Frozen Encoder → {discrim_hidden_dim} → {num_labels}")
            
            # For pretrained discriminators, we need to initialize the classifier first
            # by doing a dummy forward pass, then create optimizer with only classifier parameters
            dummy_batch = next(iter(train_dl))
            dummy_x1, dummy_x2, dummy_y = dummy_batch[0][:1], dummy_batch[1][:1], dummy_batch[2][:1] #TODO wait didnt we access data above differently?!!
            dummy_x1, dummy_x2, dummy_y = dummy_x1.to(global_device), dummy_x2.to(global_device), dummy_y.to(global_device)
            
            # Initialize classifiers with dummy forward passes
            _ = d1(dummy_x1)  # This will trigger classifier initialization
            _ = d2(dummy_x2)  # This will trigger classifier initialization  
            _ = d12(dummy_x1, dummy_x2)  # This will trigger classifier initialization
            
            # Now create optimizers with only the trainable (classifier) parameters
            opt1 = torch.optim.Adam(d1.trainable_parameters(), lr=1e-3)
            wandb_prefix = f"pretrained_discriminator_1/{model_name}" if wandb_enabled and model_name else None
            d1 = train_discrim(
                model=d1,
                loader=train_dl,
                optimizer=opt1,
                data_type=([0], [2]),
                num_epoch=discrim_epochs,
                wandb_prefix=wandb_prefix,
                use_compile=use_compile,
                cluster_method='gmm',
                enable_extended_metrics=enable_extended_metrics
            )
            
            print(f"\n🧠 TRAINING PRETRAINED DISCRIMINATOR 2 ({domain_names[1]}) - {discrim_epochs} epochs")
            print(f"   Input: {domain_names[1]} features → Labels (via pretrained encoder)")
            print(f"   Architecture: Frozen Encoder → {discrim_hidden_dim} → {num_labels}")
            
            opt2 = torch.optim.Adam(d2.trainable_parameters(), lr=1e-3)
            wandb_prefix = f"pretrained_discriminator_2/{model_name}" if wandb_enabled and model_name else None
            d2 = train_discrim(
                model=d2,
                loader=train_dl,
                optimizer=opt2,
                data_type=([1], [2]),
                num_epoch=discrim_epochs,
                wandb_prefix=wandb_prefix,
                use_compile=use_compile,
                cluster_method='gmm', #TODO does this mean we train with KL loss in any case? even if k means?
                enable_extended_metrics=enable_extended_metrics
            )
            
            print(f"\n🧠 TRAINING PRETRAINED JOINT DISCRIMINATOR ({domain_names[0]}+{domain_names[1]}) - {discrim_epochs} epochs")
            print(f"   Input: Combined features → Labels (via pretrained encoders)")
            print(f"   Architecture: Frozen Encoders → {joint_discrim_hidden_dim} → {num_labels}")
            
            opt12 = torch.optim.Adam(d12.trainable_parameters(), lr=1e-3)
            wandb_prefix = f"pretrained_discriminator_joint/{model_name}" if wandb_enabled and model_name else None
            d12 = train_discrim(
                model=d12,
                loader=train_dl,
                optimizer=opt12,
                data_type=([0,1], [2]),
                num_epoch=discrim_epochs,
                wandb_prefix=wandb_prefix,
                use_compile=use_compile,
                cluster_method='gmm',
                enable_extended_metrics=enable_extended_metrics
            )
        
        elif model_type == "complete_MLP":
            print("🧠 Using complete MLP discriminators (original approach)")
            print("🔧 Setting up discriminator caching...")
            
            # Create cache directory and compute cache parameters
            model_dir = Path.cwd() / "discriminator_cache"  # Use current working directory if no specific model path
            model_name = model_name if model_name else "critic_ce_alignment"
            discrim_cache_dir = model_dir / "discrim_cache" 
            discrim_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute a data hash for cache differentiation
            train_data_sample = []
            for batch_idx, batch in enumerate(train_dl):
                train_data_sample.append(torch.cat([batch[0], batch[1], batch[2]], dim=1))
                if len(train_data_sample) >= 5:  # Use first 5 batches for hash
                    break
            
            # Determine domain names (fallback to generic names if not available)
            if domain_names is None or len(domain_names) < 2:
                domain1_name = "domain1"
                domain2_name = "domain2"
            else:
                domain1_name, domain2_name = domain_names[0], domain_names[1]
            
            # Cache filename template
            def get_discrim_cache_path(discrim_type: str) -> Path:
                # Include domain names to distinguish multimodal setups
                domain_suffix = f"_{domain1_name}_{domain2_name}"
                
                if discrim_type == "d12":
                    # Use joint_discrim_layers and joint_discrim_hidden_dim for joint discriminator
                    cache_filename = f"{model_name}_{discrim_type}{domain_suffix}_h{joint_discrim_hidden_dim}_l{joint_discrim_layers}_e{discrim_epochs}_s{train_data_sample[0].size(0) if train_data_sample else 0}_c{num_labels}_gmm_d{x1.size(1)}x{x2.size(1)}_comp{int(use_compile)}.pt"
                else:
                    # Use regular discrim_layers and discrim_hidden_dim for individual discriminators
                    input_dim = x1.size(1) if discrim_type == "d1" else x2.size(1)
                    domain_name = domain1_name if discrim_type == "d1" else domain2_name
                    cache_filename = f"{model_name}_{discrim_type}_{domain_name}_h{discrim_hidden_dim}_l{discrim_layers}_e{discrim_epochs}_s{train_data_sample[0].size(0) if train_data_sample else 0}_c{num_labels}_gmm_d{input_dim}_comp{int(use_compile)}.pt"
                return discrim_cache_dir / cache_filename
            
            # ========================================
            # 🔥 DISCRIMINATOR 1 TRAINING 
            # ========================================
            print(f"\n🧠 TRAINING/LOADING DISCRIMINATOR 1 ({domain1_name}) - {discrim_epochs} epochs")
            print(f"   Input: {domain1_name} features → Labels")
            print(f"   Architecture: {x1.size(1)} → {discrim_hidden_dim} → {num_labels}")
            
            discrim_1_cache_path = get_discrim_cache_path("d1")
            d1 = load_discriminator_with_validation(
                cache_path=discrim_1_cache_path,
                discrim_type="d1", 
                expected_input_dim=x1.size(1),
                expected_hidden_dim=discrim_hidden_dim,
                expected_layers=discrim_layers
            )
            
            if d1 is None or force_retrain_discriminators:
                print(f"🔄 Training new discriminator 1...")
                d1 = Discrim(x1.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(global_device)
                opt1 = torch.optim.Adam(d1.parameters(), lr=1e-3)
                wandb_prefix = f"discriminator_1/{model_name}" if wandb_enabled and model_name else None
                
                # Use the proper train_discrim function with all advanced metrics
                d1 = train_discrim(
                    model=d1,
                    loader=train_dl,
                    optimizer=opt1,
                    data_type=([0], [2]),
                    num_epoch=discrim_epochs,
                    wandb_prefix=wandb_prefix,
                    use_compile=use_compile,
                    cluster_method='gmm',  # Use GMM as default, will auto-detect from data
                    enable_extended_metrics=enable_extended_metrics  # Use the parameter value
                )
                
                # Save to cache with enhanced metadata
                save_discriminator_with_metadata(
                    discriminator=d1,
                    cache_path=discrim_1_cache_path,
                    discrim_type="d1",
                    input_dim=x1.size(1),
                    hidden_dim=discrim_hidden_dim,
                    layers=discrim_layers,
                    model_path="critic_ce_alignment",
                    model_name=model_name,
                    domain_name=domain1_name,
                    num_clusters=num_labels,
                    discrim_epochs=discrim_epochs,
                    n_samples=train_data_sample[0].size(0) if train_data_sample else 0,
                    cluster_method_discrim='gmm',
                    use_compile_torch=use_compile
                )
            
            # ========================================
            # 🔥 DISCRIMINATOR 2 TRAINING
            # ========================================
            print(f"\n🧠 TRAINING/LOADING DISCRIMINATOR 2 ({domain2_name}) - {discrim_epochs} epochs")
            print(f"   Input: {domain2_name} features → Labels")
            print(f"   Architecture: {x2.size(1)} → {discrim_hidden_dim} → {num_labels}")
            
            discrim_2_cache_path = get_discrim_cache_path("d2")
            d2 = load_discriminator_with_validation(
                cache_path=discrim_2_cache_path,
                discrim_type="d2",
                expected_input_dim=x2.size(1),
                expected_hidden_dim=discrim_hidden_dim,
                expected_layers=discrim_layers
            )
            
            if d2 is None or force_retrain_discriminators:
                print(f"🔄 Training new discriminator 2...")
                d2 = Discrim(x2.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(global_device)
                opt2 = torch.optim.Adam(d2.parameters(), lr=1e-3)
                wandb_prefix = f"discriminator_2/{model_name}" if wandb_enabled and model_name else None
                
                d2 = train_discrim(
                    model=d2,
                    loader=train_dl,
                    optimizer=opt2,
                    data_type=([1], [2]),
                    num_epoch=discrim_epochs,
                    wandb_prefix=wandb_prefix,
                    use_compile=use_compile,
                    cluster_method='gmm',
                    enable_extended_metrics=enable_extended_metrics
                )
                
                # Save to cache with enhanced metadata
                save_discriminator_with_metadata(
                    discriminator=d2,
                    cache_path=discrim_2_cache_path,
                    discrim_type="d2",
                    input_dim=x2.size(1),
                    hidden_dim=discrim_hidden_dim,
                    layers=discrim_layers,
                    model_path="critic_ce_alignment",
                    model_name=model_name,
                    domain_name=domain2_name,
                    num_clusters=num_labels,
                    discrim_epochs=discrim_epochs,
                    n_samples=train_data_sample[0].size(0) if train_data_sample else 0,
                    cluster_method_discrim='gmm',
                    use_compile_torch=use_compile
                )

            # ========================================
            # 🔥 JOINT DISCRIMINATOR TRAINING
            # ========================================
            print(f"\n🧠 TRAINING/LOADING JOINT DISCRIMINATOR ({domain1_name}+{domain2_name}) - {discrim_epochs} epochs")
            print(f"   Input: Combined features → Labels")
            print(f"   Architecture: {x1.size(1) + x2.size(1)} → {joint_discrim_hidden_dim} → {num_labels}")

            discrim_12_cache_path = get_discrim_cache_path("d12")
            d12 = load_discriminator_with_validation(
                cache_path=discrim_12_cache_path,
                discrim_type="d12",
                expected_input_dim=x1.size(1) + x2.size(1),
                expected_hidden_dim=joint_discrim_hidden_dim,
                expected_layers=joint_discrim_layers
            )
            
            if d12 is None or force_retrain_discriminators:
                print(f"🔄 Training new discriminator 12...")
                d12 = Discrim(x1.size(1) + x2.size(1), joint_discrim_hidden_dim, num_labels, layers=joint_discrim_layers, activation='relu').to(global_device)
                opt12 = torch.optim.Adam(d12.parameters(), lr=1e-3)
                wandb_prefix = f"discriminator_joint/{model_name}" if wandb_enabled and model_name else None
                
                d12 = train_discrim(
                    model=d12,
                    loader=train_dl,
                    optimizer=opt12,
                    data_type=([0,1], [2]),
                    num_epoch=discrim_epochs,
                    wandb_prefix=wandb_prefix,
                    use_compile=use_compile,
                    cluster_method='gmm',
                    enable_extended_metrics=enable_extended_metrics
                )
                
                # Save to cache with enhanced metadata including joint parameters
                save_discriminator_with_metadata(
                    discriminator=d12,
                    cache_path=discrim_12_cache_path,
                    discrim_type="d12",
                    input_dim=x1.size(1) + x2.size(1),
                    hidden_dim=joint_discrim_hidden_dim,
                    layers=joint_discrim_layers,
                    model_path="critic_ce_alignment",
                    model_name=model_name,
                    domain_name=f"{domain1_name}+{domain2_name}",
                    num_clusters=num_labels,
                    discrim_epochs=discrim_epochs,
                    n_samples=train_data_sample[0].size(0) if train_data_sample else 0,
                    cluster_method_discrim='gmm',
                    use_compile_torch=use_compile,
                    # Joint discriminator specific metadata
                    joint_discrim_hidden_dim=joint_discrim_hidden_dim,
                    joint_discrim_layers=joint_discrim_layers,
                    x1_dim=x1.size(1),
                    x2_dim=x2.size(1)
                )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'complete_MLP' or 'pretrained_encoders'")
    
    
    
    # Step 2: Create and train CE alignment model
    print("Creating CE alignment model...")
    
    #TODO make the visualization function for the distribution of labels
    # Create p_y distribution - using the correct method from analyze_pid_new.py
    print("🔄 Computing label distribution...")
    with torch.no_grad():
        # Ensure labels are 1D for one-hot encoding
        labels_flat = labels.view(-1)
        if labels_flat.dtype in (torch.int64, torch.int32):
            # k-means: hard labels → one-hot encode
            one_hot = F.one_hot(labels_flat.long(), num_labels).float()
        else:
            # GMM: soft labels → already probabilities
            # ensure shape [N, num_clusters]
            one_hot = labels.view(-1, num_labels).float()
        # compute p_y
        p_y = one_hot.sum(dim=0) / one_hot.size(0)
        p_y = p_y.to(global_device)  # Ensure on correct device
        del one_hot, labels_flat
    
    # Create CE alignment model
    ce_model = CEAlignmentInformation(
        x1_dim=x1.size(1), 
        x2_dim=x2.size(1), 
        hidden_dim=discrim_hidden_dim, 
        embed_dim=discrim_hidden_dim,  # Use same as hidden_dim for embed_dim
        num_labels=num_labels, 
        layers=discrim_layers, 
        activation='relu',
        discrim_1=d1,
        discrim_2=d2,
        discrim_12=d12,
        p_y=p_y
    ).to(global_device)
    
    # Train CE alignment using the proper train_ce_alignment function
    if ce_epochs > 0:
        print("Training CE alignment...")
        wandb_prefix = f"ce_alignment/{model_name}" if wandb_enabled and model_name else "ce_alignment" if wandb_enabled else None
        
        # Use the proper train_ce_alignment function with all features
        ce_model = train_ce_alignment(
            model=ce_model,
            loader=train_dl,
            optimizer_class=torch.optim.Adam,
            num_epoch=ce_epochs,
            wandb_prefix=wandb_prefix,
            use_compile=use_compile,
            test_mode=test_mode,
            max_test_examples=max_test_examples,
            auto_find_lr=auto_find_lr,
            lr_finder_steps=lr_finder_steps,
            lr_start=lr_start,
            lr_end=lr_end
        )
    
    # Step 3: Compute PID values using the trained CEAlignmentInformation model
    print("Computing PID values...")
    ce_model.eval()
    
    # Use the EXACT same evaluation logic as the original
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loss = 0
    all_pids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                x1_batch = batch[0].float().to(global_device)
                x2_batch = batch[1].float().to(global_device)
                y_batch = batch[2].to(global_device)
                
                # Handle y shape - ensure it's 1D if needed
                if y_batch.dim() > 1 and y_batch.size(1) == 1:
                    y_batch = y_batch.squeeze()
                
                # Validate shapes before processing
                if x1_batch.size(0) != y_batch.size(0) or x2_batch.size(0) != y_batch.size(0):
                    print("⚠️  Batch size mismatch detected, adjusting...")
                    # Make sure all have the same batch size by trimming
                    min_batch = min(x1_batch.size(0), x2_batch.size(0), y_batch.size(0))
                    x1_batch = x1_batch[:min_batch]
                    x2_batch = x2_batch[:min_batch]
                    y_batch = y_batch[:min_batch]
                
                # Forward pass - ce_model returns (loss, pid_vals, P)
                loss, pid_values, P = ce_model(x1_batch, x2_batch, y_batch)
                
                test_loss += loss.item()
                all_pids.append(pid_values.cpu())
                
            except Exception as e:
                print(f"❌ Error in eval batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next batch
                continue
                
            # Clear memory
            del x1_batch, x2_batch, y_batch
    
    if all_pids:
        # Compute average test loss (exactly like original)
        avg_test_loss = test_loss / len(test_loader)
        
        # Average PID values across ALL batches (exactly like original)
        all_pids = torch.stack(all_pids).mean(dim=0)
        
        # Extract the individual components
        redundancy = all_pids[0].item()
        unique1 = all_pids[1].item()
        unique2 = all_pids[2].item()
        synergy = all_pids[3].item()
        
        print(f"✨ CE Alignment Results:")
        print(f"├─ Loss: {avg_test_loss:.4f}")
        print(f"└─ PID Components:")
        print(f"   ├─ Redundancy: {redundancy:.4f}")
        print(f"   ├─ Unique1: {unique1:.4f}")
        print(f"   ├─ Unique2: {unique2:.4f}")
        print(f"   └─ Synergy: {synergy:.4f}")
        
        if HAS_WANDB and wandb_enabled and wandb.run is not None:
            # Use consistent prefix with CE training
            final_wandb_prefix = f"ce_alignment/{model_name}" if model_name else "ce_alignment"
            
            # Create professional PID visualization
            try:
                from .utils import create_professional_pid_comparison
                pid_dict = {
                    'redundancy': redundancy,
                    'unique_1': unique1, 
                    'unique_2': unique2,
                    'synergy': synergy
                }
                create_professional_pid_comparison(
                    pid_dict,
                    title=f"PID Analysis Results - {model_name}" if model_name else "PID Analysis Results",
                    wandb_key=f"{final_wandb_prefix}/pid_analysis_visualization"
                )
            except Exception as e:
                print(f"Warning: Could not create professional PID plots: {e}")
            
            wandb.log({
                f"{final_wandb_prefix}/pid_redundancy": redundancy,
                f"{final_wandb_prefix}/pid_unique1": unique1,
                f"{final_wandb_prefix}/pid_unique2": unique2,
                f"{final_wandb_prefix}/pid_synergy": synergy,
                f"{final_wandb_prefix}/test_loss": avg_test_loss,
                # Also log with the standard pid/ prefix for consistency with analyze_pid_new
                "pid/redundancy": redundancy,
                "pid/unique1": unique1,
                "pid/unique2": unique2,
                "pid/synergy": synergy,
                "pid/test_loss": avg_test_loss
            })
    else:
        print("\n⚠️  No valid evaluation batches processed. Using default values.")
        all_pids = torch.zeros(4)  # Default to zeros if no valid batches
    
    return all_pids, None, (ce_model, d1, d2, d12, p_y) 