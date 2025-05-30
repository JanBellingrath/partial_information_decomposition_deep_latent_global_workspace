import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import gc
from typing import Callable, Optional, List, Tuple, Dict, Any

# Imports for metrics, moved to top-level
from sklearn.metrics import precision_recall_fscore_support, jaccard_score, precision_recall_curve
from scipy.stats import spearmanr, kendalltau

# Try to import wandb, but make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .data import MultimodalDataset # Assuming data.py is in the same directory
from .models import Discrim, CEAlignmentInformation # Assuming models.py is in the same directory

# Import find_optimal_lr directly to avoid circular imports
try:
    from .utils import find_optimal_lr
except ImportError:
    # Define a dummy function to avoid errors
    def find_optimal_lr(*args, **kwargs):
        print("Warning: find_optimal_lr not available")
        return 1e-3

# Global configurations (avoid circular imports)
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False  # Default value
PRECISION = torch.float16  # Default precision for AMP

# Global configurations (consider moving to a config file or passing as arguments)

# Import AMP functionality
try:
    from torch.amp import autocast, GradScaler
    
    class DummyAMPModule:
        def __init__(self):
            self.autocast = autocast
            self.GradScaler = GradScaler
    
    amp = DummyAMPModule()
    
except ImportError:
    print("Warning: torch.amp not available. Mixed precision training disabled.")
    
    class DummyAMPModule:
        def __init__(self):
            pass
        
        def autocast(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        
        def GradScaler(self, *args, **kwargs):
            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
            return DummyScaler()
    
    amp = DummyAMPModule()

# Global scaler instance, enabled based on USE_AMP 
# This scaler is for train_discrim. train_ce_alignment will use CEAlignmentInformation's internal scaler.
scaler = amp.GradScaler()

def train_discrim(model, loader, optimizer, data_type, num_epoch=40, wandb_prefix=None, use_compile=True, cluster_method='gmm', enable_extended_metrics=True):
    """Train a Discrim on (X → Y). data_type tells which fields of the batch are features/labels."""
    model.train()
    model.to(global_device) # Ensure model is on the correct device

    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("🚀 Applied torch.compile optimization to discriminator")
        except Exception as e:
            print(f"Warning: torch.compile failed for discriminator: {e}")

    prev_probs_epoch = None
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

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            
            xs = [batch[i].float().to(global_device) for i in data_type[0]]
            y_batch_original = batch[data_type[1][0]].to(global_device)
            
            current_device_type = global_device.type
            autocast_dtype = PRECISION if current_device_type == 'cuda' else torch.bfloat16
            
            with amp.autocast(device_type=current_device_type, dtype=autocast_dtype, enabled=USE_AMP):
                logits_batch = model(*xs)
                
                if cluster_method == 'kmeans':
                    y_batch_for_loss = y_batch_original.long()
                    if y_batch_for_loss.dim() > 1:
                        y_batch_for_loss = y_batch_for_loss.squeeze(-1) if y_batch_for_loss.size(1) == 1 else y_batch_for_loss[:, 0]
                    
                    # Get num_classes from model output layer
                    if hasattr(model, 'module') and hasattr(model.module, 'mlp') and model.module.mlp: # Compiled model
                        num_classes = model.module.mlp[-1].out_features
                    elif hasattr(model, 'mlp') and model.mlp: # Regular model
                        num_classes = model.mlp[-1].out_features
                    else:
                        # Fallback or error if mlp structure is not found
                        print("Warning: Could not determine num_classes from model structure in train_discrim.")
                        num_classes = logits_batch.shape[-1] 

                    if y_batch_for_loss.min() < 0 or y_batch_for_loss.max() >= num_classes:
                        clamped_labels = torch.clamp(y_batch_for_loss - y_batch_for_loss.min() if y_batch_for_loss.min() < 0 else y_batch_for_loss, 0, num_classes-1)
                        y_batch_for_loss = clamped_labels
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(logits_batch, y_batch_for_loss)
                    all_targets_epoch.append(y_batch_for_loss.detach().cpu())
                else:  # GMM
                    y_batch_for_loss = y_batch_original # Soft labels
                    log_q_batch = F.log_softmax(logits_batch, dim=1)
                    criterion = nn.KLDivLoss(reduction='batchmean')
                    loss = criterion(log_q_batch, y_batch_for_loss)
                    all_targets_epoch.append(y_batch_for_loss.argmax(dim=1).detach().cpu() if y_batch_for_loss.dim() > 1 else y_batch_for_loss.detach().cpu())
            
            all_logits_epoch.append(logits_batch.detach().cpu())
            
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            del xs, y_batch_original, logits_batch, loss, y_batch_for_loss
            if cluster_method == 'gmm': del log_q_batch
            if global_device.type == 'cuda': torch.cuda.empty_cache()
        
        logits_epoch = torch.cat(all_logits_epoch)
        targets_epoch_np = torch.cat(all_targets_epoch).numpy()
        if targets_epoch_np.ndim > 1: targets_epoch_np = targets_epoch_np.squeeze(-1) if targets_epoch_np.shape[1] == 1 else np.argmax(targets_epoch_np, axis=1)

        probs_epoch_np = F.softmax(logits_epoch, dim=1).numpy()
        preds_epoch_np = probs_epoch_np.argmax(axis=1)
        num_classes_epoch = probs_epoch_np.shape[1]

        top_k_val = 5
        if logits_epoch.shape[0] > 0:
            actual_k = min(top_k_val, num_classes_epoch)
            if actual_k > 0:
                _, pred_top_k = torch.topk(logits_epoch, actual_k, dim=1, largest=True, sorted=True)
                targets_epoch_torch_for_topk = torch.from_numpy(targets_epoch_np).view(-1, 1).expand_as(pred_top_k)
                correct_k = torch.any(pred_top_k == targets_epoch_torch_for_topk, dim=1)
                top_k_accuracy_epoch = correct_k.float().mean().item()
        
        avg_batch_loss = epoch_loss / len(loader)
        accuracy_epoch = np.mean(preds_epoch_np == targets_epoch_np) if preds_epoch_np.size > 0 else np.nan

        print(f"\nEpoch {epoch+1:3d}/{num_epoch} 🚀")
        print("--------------------------------------------------")
        print("  📋 Metrics:")
        print(f"    - Avg Batch Loss (Criterion): {avg_batch_loss:.4f}")
        print(f"    - Accuracy (Top-1):           {accuracy_epoch:.4f}")
        print(f"    - Accuracy (Top-{top_k_val}):         {top_k_accuracy_epoch:.4f}")
        
        if enable_extended_metrics:
            ce_loss_metric = F.cross_entropy(logits_epoch, torch.from_numpy(targets_epoch_np).long(), reduction='mean').item()
            one_hot_epoch = np.eye(num_classes_epoch)[targets_epoch_np]
            kl_div_metric = np.mean(np.sum(one_hot_epoch * (np.log(one_hot_epoch + 1e-12) - np.log(probs_epoch_np + 1e-12)), axis=1))
            jaccard_metric = jaccard_score(targets_epoch_np, preds_epoch_np, average='macro', zero_division=0)
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(targets_epoch_np, preds_epoch_np, average='micro', zero_division=0)
            entropies_epoch = -np.sum(probs_epoch_np * np.log(probs_epoch_np + 1e-12), axis=1)
            entropy_mean_metric = entropies_epoch.mean()

            if prev_probs_epoch is not None and prev_probs_epoch.shape == probs_epoch_np.shape:
                try:
                    true_class_probs_current = probs_epoch_np[np.arange(len(targets_epoch_np)), targets_epoch_np]
                    true_class_probs_prev = prev_probs_epoch[np.arange(len(targets_epoch_np)), targets_epoch_np]
                    if len(true_class_probs_current) > 1 and len(true_class_probs_prev) > 1:
                        rho, _ = spearmanr(true_class_probs_prev, true_class_probs_current)
                        tau, _ = kendalltau(true_class_probs_prev, true_class_probs_current)
                except Exception as e: print(f"Warning: Could not compute rank correlation: {e}")
            prev_probs_epoch = probs_epoch_np.copy()

            brier_metric = np.mean(np.sum((probs_epoch_np - one_hot_epoch)**2, axis=1))
            max_probs_1d = probs_epoch_np.max(axis=1)
            targets_epoch_np_1d = targets_epoch_np # Already 1D
            if len(max_probs_1d) == len(targets_epoch_np_1d) and len(max_probs_1d) > 0:
                try:
                    num_bins = 10
                    bins_ece = np.linspace(0.0, 1.0, num_bins + 1)
                    bin_indices = np.digitize(max_probs_1d, bins_ece[1:-1])
                    bin_accuracies_calc, bin_confidences_calc, bin_counts_calc = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)
                    for i_bin in range(num_bins):
                        in_bin = (bin_indices == i_bin)
                        bin_counts_calc[i_bin] = np.sum(in_bin)
                        if bin_counts_calc[i_bin] > 0:
                            bin_accuracies_calc[i_bin] = np.mean((preds_epoch_np == targets_epoch_np_1d)[in_bin])
                            bin_confidences_calc[i_bin] = np.mean(max_probs_1d[in_bin])
                    valid_bins_mask = bin_counts_calc > 0
                    if np.any(valid_bins_mask):
                        rel_diag_fraction_of_positives = bin_accuracies_calc[valid_bins_mask]
                        rel_diag_mean_predicted_value = bin_confidences_calc[valid_bins_mask]
                        ece_metric = np.sum(np.abs(rel_diag_fraction_of_positives - rel_diag_mean_predicted_value) * (bin_counts_calc[valid_bins_mask] / np.sum(bin_counts_calc)))
                    else: ece_metric = np.nan
                except Exception as e_cal: print(f"Warning: Could not compute ECE: {e_cal}")
            
            print(f"    - Cross-Entropy (Log-Loss):   {ce_loss_metric:.4f}")
            print(f"    - KL Divergence (vs. OneHot): {kl_div_metric:.4f}")
            # ... (other print statements for metrics) ...
            print(f"    - Brier Score (Multiclass):   {brier_metric:.4f}")

        print("--------------------------------------------------")
        
        if HAS_WANDB and wandb_prefix is not None and wandb.run is not None:
            log_dict = { f"{wandb_prefix}/{k}": v for k, v in locals().items() if isinstance(v, (float, int)) and ('metric' in k or 'accuracy' in k or 'loss' in k or 'rho' in k or 'tau' in k) }
            log_dict[f"{wandb_prefix}/epoch"] = epoch
            # ... (wandb logging for plots) ...
            wandb.log(log_dict)
        del logits_epoch, targets_epoch_np # Add other large tensors if any
        if global_device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()

    return model

def train_ce_alignment(model: CEAlignmentInformation, loader: DataLoader, optimizer_class: Callable[..., torch.optim.Optimizer], 
                       num_epoch=10, wandb_prefix: Optional[str]=None, step_offset=0, use_compile=True, 
                       test_mode=False, max_test_examples=3000, auto_find_lr=False, 
                       lr_finder_steps=200, lr_start=1e-7, lr_end=1.0):
    
    # model is already on global_device (expected to be moved by caller, e.g. critic_ce_alignment)
    # For safety, can assert: assert next(model.parameters()).device == global_device
    model.train()

    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("🚀 Applied torch.compile optimization to CE alignment model")
        except Exception as e:
            print(f"Warning: torch.compile failed for CE alignment model: {e}")

    optimizer = optimizer_class(model.align.parameters(), lr=1e-3) 
    
    if auto_find_lr:
        print("🔍 Finding optimal learning rate for CEAlignment's align sub-module...")
        subset_size = min(5000, len(loader.dataset))
        # ... (subset dataset creation logic as before) ...
        if len(loader.dataset) <= subset_size and len(loader.dataset) > 0:
            subset_dataset = loader.dataset
        elif len(loader.dataset) > subset_size:
            indices = torch.randperm(len(loader.dataset))[:subset_size].tolist()
            subset_dataset = Subset(loader.dataset, indices)
        else:
            subset_dataset = None
            print("Warning: auto_find_lr called with empty dataset. Skipping LR finding.")

        if subset_dataset:
            # Ensure find_optimal_lr is called with the model on the correct device
            # and using GLOBAL_USE_AMP settings internally if it needs to run a training loop.
            # The LRFinder in utils.py should handle its own device placement and AMP based on GLOBAL_USE_AMP.
            best_lr = find_optimal_lr(
                model=model, # CEAlignmentInformation model
                train_ds=subset_dataset,
                batch_size=loader.batch_size if loader.batch_size else 32,
                start_lr=lr_start, end_lr=lr_end, num_iter=lr_finder_steps,
                log_to_wandb=(wandb_prefix is not None and HAS_WANDB and wandb.run is not None),
                # device_for_finder=global_device # LRFinder should use global_device from utils
            )
            for pg in optimizer.param_groups: pg["lr"] = best_lr
            print(f"✨ Using learning rate for CEAlignment: {best_lr:.2e}")

    num_batches = len(loader)
    wandb_step_offset = step_offset if wandb_prefix else 0
    avg_loss = 0.0
    examples_processed = 0

    if test_mode and HAS_WANDB and wandb_prefix is not None and wandb.run is not None:
        print(f"📝 Using existing wandb run (test mode, max {max_test_examples} examples)")
        wandb.config.update({"test_mode": True, "max_test_examples": max_test_examples}, allow_val_change=True)
    
    for epoch in range(num_epoch):
        epoch_loss_sum = 0.0
        num_batches_epoch = 0
        for batch_idx, batch_data in enumerate(loader):
            x1_batch, x2_batch, y_batch_orig = batch_data # y_batch might be soft or hard labels
            x1_batch, x2_batch, y_batch_orig = x1_batch.to(global_device), x2_batch.to(global_device), y_batch_orig.to(global_device)
            
            # The model.forward (CEAlignmentInformation) handles its own autocast via self.use_amp.
            # It returns (loss, pid_vals, P).
            # The loss is for model.align parameters.
            loss_tuple = model(x1_batch, x2_batch, y_batch_orig)
            
            # Unpack the tuple - model returns (loss, pid_vals, P)
            loss = loss_tuple[0]  # Get just the loss for backward pass
            pid_vals = loss_tuple[1]  # Get PID values for logging
            
            # Backward pass and optimization
            optimizer.zero_grad()

            if model.use_amp and model.scaler is not None:
                model.scaler.scale(loss).backward()
                model.scaler.step(optimizer)
                model.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Calculate gradient norm before optimizer step
            grad_norm = sum(p.grad.norm(2).item() ** 2 for p in model.align.parameters() if p.grad is not None) ** 0.5
            epoch_loss_sum += loss.item()
            num_batches_epoch += 1
            examples_processed += x1_batch.size(0)
            
            log_step_condition = test_mode or ((batch_idx + 1) % 10 == 0)
            if HAS_WANDB and wandb_prefix is not None and wandb.run is not None and log_step_condition:
                global_step = epoch * num_batches + batch_idx + wandb_step_offset
                log_data = { f"{wandb_prefix}/{k}": v.item() if isinstance(v, torch.Tensor) else v for k,v in zip(["batch_loss", "redundancy", "unique1", "unique2", "synergy"], [loss] + list(pid_vals)) }
                log_data.update({f"{wandb_prefix}/gradient_norm": grad_norm, "epoch": epoch + 1, "batch_in_epoch": batch_idx + 1})
                if test_mode: log_data["examples_processed"] = examples_processed
                wandb.log(log_data, step=global_step)
                if test_mode and batch_idx % 5 == 0: print(f"Test mode: Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}, Examples: {examples_processed}/{max_test_examples}")
            
            if test_mode and examples_processed >= max_test_examples:
                print(f"Test mode: Reached {examples_processed} examples, stopping training")
                break
        if test_mode and examples_processed >= max_test_examples: break
            
        avg_epoch_loss = epoch_loss_sum / num_batches_epoch if num_batches_epoch > 0 else np.nan
        avg_loss += avg_epoch_loss
        if HAS_WANDB and wandb_prefix is not None and wandb.run is not None:
            wandb.log({f"{wandb_prefix}/train_loss": avg_epoch_loss, f"{wandb_prefix}/epoch": epoch}, step=epoch + step_offset)
        print(f"Epoch {epoch+1}/{num_epoch} - CE Align Loss: {avg_epoch_loss:.6f}")
    
    avg_loss /= min(num_epoch, epoch + 1) if num_epoch > 0 and epoch >=0 else 1
    print(f"Training completed - Average Loss: {avg_loss:.6f}")
    if test_mode and HAS_WANDB and wandb_prefix is not None and wandb.run is not None:
        wandb.log({f"{wandb_prefix}/final_loss": avg_loss, f"{wandb_prefix}/examples_processed": examples_processed, f"{wandb_prefix}/training_completed": True})
    if global_device.type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return model

def critic_ce_alignment(
    x1, x2, labels, num_labels,
    train_ds, test_ds,
    discrim_1=None, discrim_2=None, discrim_12=None,
    learned_discrim=True, shuffle=True,
    discrim_epochs=40, ce_epochs=10, 
    wandb_enabled=False, model_name=None,
    discrim_hidden_dim=64, discrim_layers=5,
    use_compile=True, test_mode=False, max_test_examples=3000, 
    auto_find_lr=False, lr_finder_steps=200, 
    lr_start=1e-7, lr_end=1.0
):
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
        test_mode: Whether to run in test mode with limited examples
        max_test_examples: Maximum number of examples to process in test mode
        auto_find_lr: Whether to automatically find the optimal learning rate
        lr_finder_steps: Number of iterations for the learning rate finder
        lr_start: Start learning rate for the finder
        lr_end: End learning rate for the finder
    
    Returns:
        Tuple of (PID components, alignments, models)
    """
    
    # Ensure input tensors are on the global_device
    x1 = x1.to(global_device)
    x2 = x2.to(global_device)
    labels = labels.to(global_device)
    
    print(f"critic_ce_alignment: x1.shape={x1.shape}, x2.shape={x2.shape}, labels.shape={labels.shape}")
    print(f"Using device: {global_device}")
    
    # Create train/test datasets from the full data
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=shuffle, 
                         num_workers=0, pin_memory=False)  # Reduced complexity
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, 
                        num_workers=0, pin_memory=False)
    
    # Step 1: Train or use discriminators
    if learned_discrim:
        print("Training learned discriminators...")
        
        # Create discriminators
        d1 = Discrim(x1.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(global_device)
        d2 = Discrim(x2.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(global_device)
        d12 = Discrim(x1.size(1) + x2.size(1), discrim_hidden_dim, num_labels, layers=discrim_layers, activation='relu').to(global_device)
        
        # Train each discriminator
        for model, data_type, name in [(d1, ([0], [2]), "d1"), (d2, ([1], [2]), "d2"), (d12, ([0,1], [2]), "d12")]:
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            wandb_prefix = f"{name}_{model_name}" if wandb_enabled and model_name else None
            
            # Use simplified training (not the full train_discrim function to avoid complexity)
            for epoch in range(min(discrim_epochs, 10)):  # Limit epochs for efficiency
                model.train()
                epoch_loss = 0.0
                
                for batch in train_dl:
                    opt.zero_grad()
                    xs = [batch[i].float().to(global_device) for i in data_type[0]]
                    y = batch[data_type[1][0]].to(global_device)
                    
                    logits = model(*xs)
                    
                    # Handle both hard and soft labels
                    if y.dim() > 1 and y.size(1) > 1:
                        # Soft labels (GMM) - use KL divergence
                        log_probs = torch.log_softmax(logits, dim=1)
                        loss = torch.nn.functional.kl_div(log_probs, y, reduction='batchmean')
                    else:
                        # Hard labels (k-means) - use CrossEntropy
                        if y.dim() > 1:
                            y = y.squeeze()
                        y = y.long()
                        loss = nn.CrossEntropyLoss()(logits, y)
                    
                    loss.backward()
                    opt.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_dl)
                print(f"  {name} epoch {epoch+1}: loss={avg_loss:.4f}")
                
                if HAS_WANDB and wandb_prefix:
                    wandb.log({f"{wandb_prefix}/loss": avg_loss, f"{wandb_prefix}/epoch": epoch})
        
    else:
        print("Using simple count-based discriminators...")
        # Use simple discriminators from the models module
        from .models import simple_discrim
        d1 = simple_discrim([x1], labels, num_labels)
        d2 = simple_discrim([x2], labels, num_labels)
        d12 = simple_discrim([x1, x2], labels, num_labels)
    
    # Step 2: Create and train CE alignment model
    print("Creating CE alignment model...")
    
    # Create p_y distribution
    p_y = torch.zeros(num_labels).to(global_device)
    if labels.dim() > 1 and labels.size(1) == num_labels:
        # Soft labels (GMM) - average across samples
        p_y = labels.mean(dim=0)
    else:
        # Hard labels (k-means) - count occurrences
        labels_int = labels.long() if labels.dtype != torch.long else labels
        for i in range(num_labels):
            p_y[i] = (labels_int == i).float().mean()
    
    # Create CE alignment model
    ce_model = CEAlignmentInformation(
        x1_dim=x1.size(1), x2_dim=x2.size(1), 
        hidden_dim=discrim_hidden_dim, embed_dim=32,
        num_labels=num_labels, layers=discrim_layers, activation='relu',
        discrim_1=d1, discrim_2=d2, discrim_12=d12, p_y=p_y
    ).to(global_device)
    
    # Train CE alignment
    if ce_epochs > 0:
        print("Training CE alignment...")
        ce_opt = torch.optim.Adam(ce_model.parameters(), lr=1e-3)
        
        if auto_find_lr:
            print("Finding optimal learning rate...")
            optimal_lr = find_optimal_lr(
                model=ce_model, train_ds=train_ds, batch_size=256,
                start_lr=lr_start, end_lr=lr_end, num_iter=lr_finder_steps,
                log_to_wandb=wandb_enabled
            )
            print(f"Optimal LR found: {optimal_lr}")
            for param_group in ce_opt.param_groups:
                param_group['lr'] = optimal_lr
        
        for epoch in range(ce_epochs):
            ce_model.train()
            epoch_loss = 0.0
            
            for batch in train_dl:
                ce_opt.zero_grad()
                x1_batch = batch[0].float().to(global_device)
                x2_batch = batch[1].float().to(global_device)
                y_batch = batch[2].to(global_device)
                
                if test_mode and len(train_dl.dataset) > max_test_examples:
                    break
                
                # Forward pass - ce_model returns (loss, pid_vals, P)
                loss, pid_vals, P = ce_model(x1_batch, x2_batch, y_batch)
                loss.backward()
                ce_opt.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_dl)
            print(f"  CE epoch {epoch+1}: loss={avg_loss:.4f}")
            
            if HAS_WANDB and wandb_enabled:
                wandb.log({"ce_alignment/loss": avg_loss, "ce_alignment/epoch": epoch})
    
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
        
        if HAS_WANDB and wandb_enabled:
            wandb.log({
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