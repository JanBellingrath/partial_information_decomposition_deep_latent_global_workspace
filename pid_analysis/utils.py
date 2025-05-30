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

# Try to import shimmer specific modules, but make them optional
try:
    from shimmer.modules.domain import DomainModule
    from shimmer.modules.gw_module import GWModule, GWEncoder, GWDecoder
    from gw_module_configurable_fusion import GWModuleConfigurableFusion
    # Try to import the shimmer version of load_domain_modules
    from shimmer.utils.utils import load_domain_modules as shimmer_load_domain_modules
    SHIMMER_AVAILABLE = True
    SHIMMER_UTILS_AVAILABLE = True
except ImportError:
    # This block catches if shimmer itself or GWModuleConfigurableFusion is missing
    # Or if shimmer.utils.utils.load_domain_modules is missing
    SHIMMER_AVAILABLE = False
    SHIMMER_UTILS_AVAILABLE = False # Assume utils are not available if base shimmer fails
    print("Warning: Shimmer modules or shimmer.utils.utils.load_domain_modules not found. Some functionalities might be limited or use fallbacks.")
    # Define dummy classes if shimmer is not available to prevent import errors
    class DomainModule:
        def __init__(self, latent_dim=64): # Add latent_dim to dummy
            self.latent_dim = latent_dim
    class GWModule: pass
    class GWEncoder(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
    class GWDecoder(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
    class GWModuleConfigurableFusion(torch.nn.Module):
        def __init__(self, *args, **kwargs): 
            super().__init__()
            self.domain_mods = kwargs.get('domain_modules', {})
            self.gw_encoders = kwargs.get('gw_encoders', {})
            self.gw_decoders = kwargs.get('gw_decoders', {})
            self.workspace_dim = kwargs.get('workspace_dim', 12)
        def fuse(self, x, selection_scores=None):
            if not self.gw_encoders: return torch.zeros(list(x.values())[0].shape[0], self.workspace_dim)
            return torch.zeros(list(x.values())[0].shape[0], self.workspace_dim, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')

    # Fallback shimmer_load_domain_modules if not imported
    def shimmer_load_domain_modules(configs, eval_mode=True, device=None):
        print("Using fallback shimmer_load_domain_modules (dummy implementation).")
        return {cfg.get('name', f'dummy_domain_{i}'): DomainModule() for i, cfg in enumerate(configs)}

# Global performance configuration
CHUNK_SIZE = 128  # Size of chunks for processing large matrices sequentially
MEMORY_CLEANUP_INTERVAL = 10  # Number of iterations after which to force memory cleanup
USE_AMP = False  # Whether to use automatic mixed precision
PRECISION = torch.float16  # Precision to use with AMP
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean up memory between operations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
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
        latent_dim = getattr(domain_module, 'latent_dim', 64) # Default if not present
        print(f"Domain '{domain_name}' latent dimension: {latent_dim}")
        gw_encoders[domain_name] = GWEncoder(in_dim=latent_dim, hidden_dim=hidden_dim, out_dim=workspace_dim, n_layers=n_layers)
        gw_decoders[domain_name] = GWDecoder(in_dim=workspace_dim, hidden_dim=hidden_dim, out_dim=latent_dim, n_layers=n_layers)
    
    if fusion_weights is None:
        weight_value = 1.0 / len(domain_modules) if domain_modules else 0.0
        fusion_weights = {name: weight_value for name in domain_modules}
    
    gw_module = GWModuleConfigurableFusion(
        domain_modules=domain_modules, workspace_dim=workspace_dim,
        gw_encoders=gw_encoders, gw_decoders=gw_decoders, fusion_weights=fusion_weights)
    
    setattr(gw_module, 'hidden_dim', hidden_dim) # Use setattr for robustness
    setattr(gw_module, 'n_layers', n_layers)
    
    if state_dict:
        try:
            if any(k.startswith('gw_module.') for k in state_dict.keys()):
                new_state_dict = {k[len('gw_module.'):]: v for k, v in state_dict.items() if k.startswith('gw_module.')}
                # Keep non-gw_module keys if they don't start with domain_mods either
                new_state_dict.update({k:v for k,v in state_dict.items() if not k.startswith('gw_module.') and not k.startswith('domain_mods.')})
                state_dict = new_state_dict
            missing_keys, unexpected_keys = gw_module.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
            if missing_keys: print(f"Missing keys: {missing_keys}")
            if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print(f"Warning: Error loading state dict: {e}. Proceeding with newly initialized model.")
    
    return gw_module.to(device)

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
    if data_module is not None:
        print(f"Using real data from {dataset_split} split for sample generation")
        return generate_samples_from_dataset_fixed(
            model=model, data_module=data_module, domain_names=domain_names,
            split=dataset_split, n_samples=n_samples, batch_size=batch_size,
            device=device, use_gw_encoded=use_gw_encoded)
    
    print(f"WARNING: No data_module provided. Using synthetic random latent vectors.")
    if not domain_names: domain_names = ["v_latents", "t"]
    if len(domain_names) != 2: raise ValueError(f"Requires 2 domains, got {len(domain_names)}")
    
    model = model.to(device); model.eval()
    domain_a, domain_b = domain_names
    reps = {f"{dn}_{rep_type}": [] for dn in domain_names for rep_type in ["orig", "gw_encoded", "decoded"]} 
    reps["gw_reps"] = []

    num_batches = (n_samples + batch_size - 1) // batch_size
    total_samples = 0
    with torch.no_grad():
        for _ in range(num_batches):
            if total_samples >= n_samples: break
            curr_batch_size = min(batch_size, n_samples - total_samples)
            
            latent_dims = {dn: getattr(model.domain_mods.get(dn, {}), 'latent_dim', 64) for dn in domain_names}
            latents = {dn: torch.randn(curr_batch_size, latent_dims[dn], device=device) for dn in domain_names}
            gw_states, decoded_states = {}, {}

            try:
                for dn in domain_names:
                    if not hasattr(model, 'gw_encoders') or dn not in model.gw_encoders:
                        print(f"Warning: Using dummy encoder for {dn}")
                        gw_states[dn] = torch.randn(curr_batch_size, getattr(model, 'workspace_dim', 12), device=device)
                    else:
                        gw_states[dn] = model.gw_encoders[dn](latents[dn])
                        if dn == 'v_latents' and gw_states[dn].dim() > 2: gw_states[dn] = gw_states[dn][:, 0, :]
                
                fused_gw = model.fuse(gw_states, None)

                for dn in domain_names:
                    if not hasattr(model, 'gw_decoders') or dn not in model.gw_decoders:
                        print(f"Warning: Using dummy decoder for {dn}")
                        decoded_states[dn] = torch.randn(curr_batch_size, latent_dims[dn], device=device)
                    else:
                        decoded_states[dn] = model.gw_decoders[dn](fused_gw)

                reps[f"{domain_a}_orig"].append(gw_states[domain_a].cpu() if use_gw_encoded else latents[domain_a].cpu())
                reps[f"{domain_b}_orig"].append(gw_states[domain_b].cpu() if use_gw_encoded else latents[domain_b].cpu())
                reps[f"{domain_a}_gw_encoded"].append(gw_states[domain_a].cpu())
                reps[f"{domain_b}_gw_encoded"].append(gw_states[domain_b].cpu())
                reps[f"{domain_a}_decoded"].append(decoded_states[domain_a].cpu())
                reps[f"{domain_b}_decoded"].append(decoded_states[domain_b].cpu())
                reps["gw_reps"].append(fused_gw.cpu())
                total_samples += curr_batch_size
            except Exception as e: print(f"Error processing batch: {e}"); continue
    
    if total_samples == 0: raise ValueError("Could not generate any samples")
    print(f"Generated {total_samples} samples")
    source_prefix = "gw_encoded" if use_gw_encoded else "latent"
    final_result = {}
    final_result[f"{domain_a}_{source_prefix}"] = torch.cat(reps[f"{domain_a}_orig"])[:n_samples]
    final_result[f"{domain_b}_{source_prefix}"] = torch.cat(reps[f"{domain_b}_orig"])[:n_samples]
    for dn in domain_names:
        final_result[f"{dn}_gw_encoded"] = torch.cat(reps[f"{dn}_gw_encoded"])[:n_samples]
        final_result[f"{dn}_decoded"] = torch.cat(reps[f"{dn}_decoded"])[:n_samples]
    final_result["gw_rep"] = torch.cat(reps["gw_reps"])[:n_samples]
    return final_result

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
    processed = {}
    # Handles tuple of dicts (from CombinedLoader)
    if isinstance(batch, tuple) and all(isinstance(item, dict) for item in batch):
        for item_dict in batch:
            for k, v in item_dict.items():
                domain_name = next(iter(k)) if isinstance(k, frozenset) else k 
                # Ensure v is a tensor before moving to device
                if isinstance(v, torch.Tensor):
                    processed[domain_name] = v.to(device)
                elif hasattr(v, 'bert') and isinstance(v.bert, torch.Tensor): # For Text objects
                     processed[domain_name] = v.bert.to(device)
                # Add more specific handling if other complex types are expected
    # Handles single dict (common case)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            domain_name = next(iter(k)) if isinstance(k, frozenset) else k
            if isinstance(v, torch.Tensor):
                processed[domain_name] = v.to(device)
            elif hasattr(v, 'bert') and isinstance(v.bert, torch.Tensor):
                 processed[domain_name] = v.bert.to(device)
    else:
        # Fallback or error for unhandled batch types
        print(f"Warning: Unhandled batch type: {type(batch)}. Attempting direct processing or skipping.")
        # Potentially, try to iterate if it's a list of tensors, or raise error
        if isinstance(batch, list) and all(isinstance(item, torch.Tensor) for item in batch):
             # This case might represent a list of tensors for a single domain if not named
             # Needs clearer specification on how to map to domain names if this occurs.
             pass # Or assign to a default domain if applicable

    # Post-process v_latents specifically if present
    if 'v_latents' in processed and processed['v_latents'].dim() > 2:
        processed['v_latents'] = processed['v_latents'][:, 0, :]
    return processed

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

# Placeholder for shimmer_utilities.load_domain_modules if not available from shimmer
# This would typically load DomainModule instances based on configuration.
def load_domain_modules_fallback(configs: List[Dict[str, Any]], eval_mode: bool = True, device: Optional[str] = None) -> Dict[str, DomainModule]:
    """
    Fallback function to load domain modules if shimmer.utils is not available.
    This creates dummy DomainModule instances.
    """
    loaded_modules = {}
    print(f"Using fallback 'load_domain_modules_fallback'. Creating dummy DomainModule instances.")
    for i, config in enumerate(configs):
        name = config.get("name", f"dummy_domain_{i}")
        # Create a basic dummy DomainModule. 
        # You might need to add more attributes if your code relies on them.
        module = DomainModule()
        setattr(module, 'latent_dim', config.get('latent_dim', 64)) # Example: set latent_dim
        setattr(module, 'name', name)
        loaded_modules[name] = module
    return loaded_modules

# Use shimmer_load_domain_modules if available, otherwise use fallback
# load_domain_modules = shimmer_load_domain_modules if SHIMMER_AVAILABLE else load_domain_modules_fallback
# For now, to avoid potential import issues if shimmer_utilities isn't structured as expected:
def load_domain_modules(configs: List[Dict[str, Any]], eval_mode: bool = True, device: Optional[str] = None) -> Dict[str, DomainModule]:
    """
    Load domain modules. Tries to use shimmer.utils.utils.load_domain_modules first,
    then falls back to a local dummy implementation.
    """
    if SHIMMER_UTILS_AVAILABLE and callable(shimmer_load_domain_modules):
        try:
            print("Attempting to load domain modules using shimmer.utils.utils.load_domain_modules...")
            # Ensure device is passed if the shimmer function expects it.
            # The signature of the actual shimmer_load_domain_modules might vary.
            # This is a guess based on common patterns.
            # Adjust if the actual shimmer function has a different signature.
            if device:
                 return shimmer_load_domain_modules(configs, eval_mode=eval_mode, device=device)
            else:
                 return shimmer_load_domain_modules(configs, eval_mode=eval_mode)
        except Exception as e:
            print(f"Error using shimmer.utils.utils.load_domain_modules: {e}. Falling back.")
            # Fall through to local fallback
    
    # Fallback to local implementation
    print("Using local fallback for load_domain_modules.")
    return load_domain_modules_fallback(configs, eval_mode=eval_mode, device=device)

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
        # Assuming USE_AMP is defined globally or passed appropriately
        global USE_AMP 
        self.use_amp = getattr(model, 'use_amp', USE_AMP) 
        # Assuming amp is defined globally (from torch.amp import autocast, GradScaler)
        global amp 
        self.scaler = amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None # GradScaler doesn't take device as parameter

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

    def range_test(self, train_loader: torch.utils.data.DataLoader, start_lr: float = 1e-7, end_lr: float = 10, num_iter: int = 100, step_mode: str = "exp", smooth_f: float = 0.05, diverge_th: float = 5.0):
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
            
            x1_batch, x2_batch, y_batch = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].long().to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                # Assuming amp.autocast is available globally
                with amp.autocast(device_type='cuda' if str(self.device) == 'cuda' else 'cpu'):
                    # Model specific call for CEAlignmentInformation, adapt if LRFinder is used for other models
                    loss, _, _ = self.model(x1_batch, x2_batch, y_batch) 
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, _, _ = self.model(x1_batch, x2_batch, y_batch)
                loss.backward()
                self.optimizer.step()
            
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

def find_optimal_lr(model, train_ds, batch_size: int = 256, start_lr: float = 1e-7, end_lr: float = 1.0, num_iter: int = 200, skip_start: int = 10, skip_end: int = 5, factor: float = 10.0, log_to_wandb: bool = False, seed: int = 42, return_finder: bool = False) -> Union[float, Tuple[float, LRFinder]]:
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
    lr_finder_instance.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
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

# Additional imports for plotting and amp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d

# Try to import amp, but make it optional
try:
    from torch.amp import autocast, GradScaler
    
    class DummyAMPModule:
        def __init__(self):
            self.autocast = autocast
            self.GradScaler = GradScaler
    
    amp = DummyAMPModule()
    
except ImportError:
    print("Warning: torch.amp not available. Some functionality may be limited.")
    
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