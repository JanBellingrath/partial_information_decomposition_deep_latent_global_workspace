import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable
import os
import json
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime

# Global configurations (consider moving to a config file or passing as arguments)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Imports from the pid_analysis package (avoid circular imports)
from .utils import (
    load_checkpoint, 
    generate_samples_from_model, 
    prepare_for_json,
    USE_AMP, CHUNK_SIZE, MEMORY_CLEANUP_INTERVAL, AGGRESSIVE_CLEANUP
)

# Global configurations (avoid circular imports)
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_USE_AMP = USE_AMP
GLOBAL_PRECISION = torch.float16 if torch.cuda.is_available() else torch.bfloat16
from .data import prepare_pid_data, MultimodalDataset
from .models import Discrim, CEAlignmentInformation # Models used in analyze_model
from .train import train_discrim, train_ce_alignment, critic_ce_alignment # Training functions used in analyze_model
from .data_interface import GeneralizedDataInterface, create_synthetic_interface

# Removed local device, USE_AMP, DummyAMPModule, and amp_eval. Using globals from utils.

# analyze_model function (moved from analyze_pid_new.py)
def analyze_model(
    model_path: str,
    domain_modules: Dict[str, Any], 
    output_dir: str,
    source_config: Dict[str, str],
    target_config: str,
    synthetic_labels: Optional[torch.Tensor] = None, 
    n_samples: int = 10000,
    batch_size: int = 128,
    num_clusters: int = 10, 
    discrim_epochs: int = 40,
    ce_epochs: int = 10,
    discrim_hidden_dim: int = 64,
    discrim_layers: int = 5,
    use_wandb: bool = True,
    wandb_project: str = "pid-analysis",
    wandb_entity: Optional[str] = None,
    data_module=None, 
    dataset_split: str = "test",
    use_gw_encoded: bool = False, 
    use_compile_torch: bool = True, # Renamed for clarity, passed to train functions
    ce_test_mode_run: bool = False, # Renamed for clarity
    max_test_examples_run: int = 3000, # Renamed for clarity
    auto_find_lr_run: bool = False, # Renamed for clarity
    lr_finder_steps_run: int = 200,
    lr_start_run: float = 1e-7,
    lr_end_run: float = 1.0,
    cluster_method_discrim: str = 'gmm', 
    enable_extended_metrics_discrim: bool = True,
    run_critic_ce_direct: bool = False # New flag to run critic_ce_alignment directly
) -> Dict[str, Any]:
    """
    Analyze a single model checkpoint.
    Can run either the original separate DINO + CE training, 
    or the combined critic_ce_alignment.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    wandb_run = None
    if use_wandb and HAS_WANDB:
        run_name = f"analyze_{Path(model_path).stem}"
        if run_critic_ce_direct: run_name += "_criticCE"
        wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity, name=run_name,
            config={ "model_path": model_path, "n_samples": n_samples, "num_clusters": num_clusters,
                     "discrim_epochs": discrim_epochs, "ce_epochs": ce_epochs,
                     "source_config": source_config, "target_config": target_config,
                     "use_compile_torch": use_compile_torch, "ce_test_mode_run": ce_test_mode_run,
                     "max_test_examples_run": max_test_examples_run if ce_test_mode_run else None,
                     "cluster_method_discrim": cluster_method_discrim, 
                     "enable_extended_metrics_discrim": enable_extended_metrics_discrim,
                     "auto_find_lr_run": auto_find_lr_run, "run_critic_ce_direct": run_critic_ce_direct
                   }
        )
    
    print(f"Loading model from {model_path}")
    model_gw = load_checkpoint(
        checkpoint_path=model_path, domain_modules=domain_modules, # type: ignore
        device=str(global_device) # Use global_device
    )
    
    domain_names = list(model_gw.domain_mods.keys()) if hasattr(model_gw, 'domain_mods') and model_gw.domain_mods else []
    print(f"Domain names from model: {domain_names}")
    if not domain_names or len(domain_names) < 2:
        print("Error: Could not extract at least two domain names from the model.")
        if wandb_run: wandb_run.finish()
        return {"error": "Insufficient domain names in model"}

    generated_data = generate_samples_from_model(
        model=model_gw, domain_names=domain_names, n_samples=n_samples, batch_size=batch_size,
        device=str(global_device), use_gw_encoded=use_gw_encoded, data_module=data_module, dataset_split=dataset_split
    )
    
    if synthetic_labels is None:
        raise ValueError(
            "synthetic_labels is required but was not provided. "
            "The create_synthetic_labels function has been removed. "
            "Please provide pre-computed synthetic labels."
        )

    # synthetic_labels are now ensured to be present (either passed or correctly generated)
    train_ds, test_ds, x1_data, x2_data, labels_data = prepare_pid_data(
        generated_data=generated_data, domain_names=domain_names,
        source_config=source_config, target_config=target_config, 
        synthetic_labels=synthetic_labels
    )
    # Move x1_data, x2_data, labels_data to global_device as they are used for CE model init / p_y calc
    x1_data, x2_data, labels_data = x1_data.to(global_device), x2_data.to(global_device), labels_data.to(global_device)

    pid_results: Dict[str, Any] = {}
    final_models: Tuple[Any, Any, Any, Any, Any] = (None, None, None, None, None)

    if run_critic_ce_direct:
        print("\nRunning PID analysis using critic_ce_alignment directly...")
        # critic_ce_alignment handles its own discriminator training and CE alignment training.
        # It also needs the raw tensors x1, x2, labels for initial p(y) and model sizing.
        avg_pid_vals, _, models_tuple = critic_ce_alignment(
            x1=x1_data, x2=x2_data, labels=labels_data, num_labels=num_clusters,
            train_ds=train_ds, test_ds=test_ds,
            # Pass Optional pre-trained discriminators if available (None for now)
            discrim_1=None, discrim_2=None, discrim_12=None, 
            learned_discrim=True, # True to train them inside critic_ce_alignment
            shuffle=True, discrim_epochs=discrim_epochs, ce_epochs=ce_epochs,
            wandb_enabled=(use_wandb and HAS_WANDB and wandb_run is not None), model_name=Path(model_path).stem,
            discrim_hidden_dim=discrim_hidden_dim, discrim_layers=discrim_layers,
            use_compile=use_compile_torch, 
            test_mode=ce_test_mode_run, max_test_examples=max_test_examples_run,
            auto_find_lr=auto_find_lr_run, lr_finder_steps=lr_finder_steps_run,
            lr_start=lr_start_run, lr_end=lr_end_run
        )
        pid_results = {
            "redundancy": avg_pid_vals[0].item(), "unique1": avg_pid_vals[1].item(),
            "unique2": avg_pid_vals[2].item(), "synergy": avg_pid_vals[3].item()
        }
        final_models = models_tuple # (ce_model, d1, d2, d12, p_y_calc)
        # No separate eval functions needed here as critic_ce_alignment includes an eval step.

    else: # Original separate training and evaluation path
        print("\nRunning PID analysis with separate DINO training and CE alignment training...")
        # Train discriminators (from .train)
        d1 = Discrim(x1_data.size(1), discrim_hidden_dim, num_clusters, layers=discrim_layers, activation="relu").to(global_device)
        opt1 = torch.optim.Adam(d1.parameters(), lr=1e-3)
        discrim_1 = train_discrim(
            d1, DataLoader(train_ds, batch_size=batch_size, shuffle=True), opt1, ([0], [2]),
            num_epoch=discrim_epochs, wandb_prefix=f"{Path(model_path).stem}/discrim_1" if use_wandb and HAS_WANDB else None, 
            use_compile=use_compile_torch, cluster_method=cluster_method_discrim, 
            enable_extended_metrics=enable_extended_metrics_discrim
        )

        d2 = Discrim(x2_data.size(1), discrim_hidden_dim, num_clusters, layers=discrim_layers, activation="relu").to(global_device)
        opt2 = torch.optim.Adam(d2.parameters(), lr=1e-3)
        discrim_2 = train_discrim(
            d2, DataLoader(train_ds, batch_size=batch_size, shuffle=True), opt2, ([1], [2]),
            num_epoch=discrim_epochs, wandb_prefix=f"{Path(model_path).stem}/discrim_2" if use_wandb and HAS_WANDB else None,
            use_compile=use_compile_torch, cluster_method=cluster_method_discrim, 
            enable_extended_metrics=enable_extended_metrics_discrim
        )

        d12 = Discrim(x1_data.size(1) + x2_data.size(1), discrim_hidden_dim, num_clusters, layers=discrim_layers, activation="relu").to(global_device)
        opt12 = torch.optim.Adam(d12.parameters(), lr=1e-3)
        discrim_12 = train_discrim(
            d12, DataLoader(train_ds, batch_size=batch_size, shuffle=True), opt12, ([0, 1], [2]),
            num_epoch=discrim_epochs, wandb_prefix=f"{Path(model_path).stem}/discrim_12" if use_wandb and HAS_WANDB else None,
            use_compile=use_compile_torch, cluster_method=cluster_method_discrim, 
            enable_extended_metrics=enable_extended_metrics_discrim
        )

        with torch.no_grad():
            labels_for_py = labels_data.view(-1).to(global_device)
            if cluster_method_discrim == 'kmeans':
                if labels_for_py.dim() > 1 and labels_for_py.shape[1] == 1: labels_for_py = labels_for_py.squeeze(-1)
                if labels_for_py.max() >= num_clusters: 
                    labels_for_py = torch.clamp(labels_for_py, 0, num_clusters - 1)
                one_hot_py = F.one_hot(labels_for_py.long(), num_clusters).float()
            else: # GMM
                one_hot_py = labels_for_py.view(-1, num_clusters).float() if labels_for_py.dim() > 1 and labels_for_py.shape[1] == num_clusters else F.one_hot(labels_for_py.long(), num_clusters).float()
            p_y_calc = one_hot_py.sum(dim=0) / (one_hot_py.size(0) + 1e-9)
            p_y_calc = p_y_calc.to(global_device)

        ce_model = CEAlignmentInformation(
            x1_data.size(1), x2_data.size(1), discrim_hidden_dim, discrim_hidden_dim,
            num_clusters, discrim_layers, "relu",
            discrim_1, discrim_2, discrim_12, p_y_calc
        ).to(global_device)
        
        train_ce_alignment(
            ce_model, DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            torch.optim.Adam, 
            num_epoch=ce_epochs, wandb_prefix=f"{Path(model_path).stem}/ce" if use_wandb and HAS_WANDB else None,
            use_compile=use_compile_torch, test_mode=ce_test_mode_run, max_test_examples=max_test_examples_run,
            auto_find_lr=auto_find_lr_run, lr_finder_steps=lr_finder_steps_run, 
            lr_start=lr_start_run, lr_end=lr_end_run
        )
        
        print("\n📊 Evaluating models...")
        # Eval functions (eval_discrim, eval_ce_alignment) are defined later in this file.
        # They should use global_device and global AMP settings implicitly or explicitly.
        discrim_1_results = eval_discrim(
            discrim_1, DataLoader(test_ds, batch_size=batch_size), ([0], [2]),
            wandb_prefix=f"{Path(model_path).stem}/discrim_1_eval" if use_wandb and HAS_WANDB else None,
            cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim
        )
        
        discrim_2_results = eval_discrim(
            discrim_2, DataLoader(test_ds, batch_size=batch_size), ([1], [2]),
            wandb_prefix=f"{Path(model_path).stem}/discrim_2_eval" if use_wandb and HAS_WANDB else None,
            cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim
        )

        discrim_12_results_eval = eval_discrim( 
            discrim_12, DataLoader(test_ds, batch_size=batch_size), ([0,1], [2]),
            wandb_prefix=f"{Path(model_path).stem}/discrim_12_eval" if use_wandb and HAS_WANDB else None,
            cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim
        )

        pid_results = eval_ce_alignment( 
            ce_model, DataLoader(test_ds, batch_size=batch_size),
            wandb_prefix=f"{Path(model_path).stem}/ce_eval" if use_wandb and HAS_WANDB else None
        )
        final_models = (ce_model, discrim_1, discrim_2, discrim_12, p_y_calc)

    # Common result preparation and saving logic
    output_results = {
        "model_path": model_path,
        "pid_results": pid_results, # This will be populated by one of the branches
        "domain_names": domain_names,
        "source_config": source_config,
        "target_config": target_config,
        "n_samples": n_samples,
        "num_clusters": num_clusters,
        "discrim_epochs": discrim_epochs,
        "ce_epochs": ce_epochs,
        # Add other relevant parameters to results
    }

    # Saving results to JSON
    results_path = Path(output_dir) / f"{Path(model_path).stem}_pid_results.json"
    with open(results_path, 'w') as f:
        json.dump(prepare_for_json(output_results), f, indent=4)
    print(f"Saved PID results to {results_path}")

    if wandb_run:
        wandb_run.log(prepare_for_json(pid_results))
        wandb_run.finish()
    
    # To keep analyze_multiple_models compatible, it expects dict with specific keys.
    # We return the main pid_results and also a fuller dict if needed by caller.
    return output_results 

def eval_discrim(model, loader, data_type, wandb_prefix=None, cluster_method='gmm', enable_extended_metrics=True):
    """Evaluate a trained discriminator. Uses global_device and GLOBAL_USE_AMP/GLOBAL_PRECISION."""
    model.eval()
    model.to(global_device) # Ensure model on correct device
    all_logits = []
    all_targets = []
    epoch_loss = 0.0 # Using epoch_loss to mirror train_discrim structure for loss calc

    current_device_type = global_device.type
    autocast_dtype = GLOBAL_PRECISION if current_device_type == 'cuda' else torch.bfloat16

    with torch.no_grad():
        for batch in loader:
            # Use the same data_type pattern as train_discrim for consistency
            xs = [batch[i].float().to(global_device) for i in data_type[0]]
            y_batch_original = batch[data_type[1][0]].to(global_device)

            with torch.amp.autocast(device_type=current_device_type, dtype=autocast_dtype, enabled=GLOBAL_USE_AMP):
                logits = model(*xs)
                
                # Detect label type from the actual data, not cluster_method parameter
                if y_batch_original.dim() > 1 and y_batch_original.size(1) > 1:
                    # Soft labels (GMM) - use KL divergence
                    y_for_loss = y_batch_original # Soft labels
                    log_q = F.log_softmax(logits, dim=1)
                    loss = F.kl_div(log_q, y_for_loss, reduction='batchmean')
                    all_targets.append(y_for_loss.argmax(dim=1).cpu())
                else:
                    # Hard labels (k-means) - use CrossEntropy
                    y_for_loss = y_batch_original.long()
                    if y_for_loss.dim() > 1: y_for_loss = y_for_loss.squeeze(-1) if y_for_loss.size(1) == 1 else y_for_loss[:,0]
                    
                    if hasattr(model, 'module') and hasattr(model.module, 'mlp') and model.module.mlp: num_classes_eval = model.module.mlp[-1].out_features
                    elif hasattr(model, 'mlp') and model.mlp: num_classes_eval = model.mlp[-1].out_features
                    else: num_classes_eval = logits.shape[-1]

                    if y_for_loss.min() < 0 or y_for_loss.max() >= num_classes_eval: 
                        y_for_loss = torch.clamp(y_for_loss, 0, num_classes_eval-1)
                    loss = F.cross_entropy(logits, y_for_loss)
                    all_targets.append(y_for_loss.cpu())
            all_logits.append(logits.cpu())
            epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader) if len(loader) > 0 else np.nan
    results = {"eval_loss": avg_loss}
    # ... (Full metric calculation as in train_discrim's epoch end, using all_logits, all_targets)
    # This part should be comprehensive, similar to train_discrim but on test data and without backprop.
    # For brevity, detailed metrics like accuracy, F1, ECE, etc. are omitted here but should be implemented.

    # Placeholder for comprehensive metrics based on all_logits and all_targets
    if all_logits and all_targets:
        logits_all_eval = torch.cat(all_logits)
        targets_all_eval_np = torch.cat(all_targets).numpy()
        probs_all_eval_np = F.softmax(logits_all_eval, dim=1).numpy()
        preds_all_eval_np = probs_all_eval_np.argmax(axis=1)
        
        accuracy = np.mean(preds_all_eval_np == targets_all_eval_np) if targets_all_eval_np.size > 0 else np.nan
        results["accuracy"] = accuracy
        # Add more metrics here (precision, recall, F1, ECE, etc.)
        print(f"Eval Discrim ({wandb_prefix if wandb_prefix else 'Unknown'}) - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    if HAS_WANDB and wandb_prefix and wandb.run:
        wandb.log({f"{wandb_prefix}/{k}": v for k,v in results.items()})
    return results

def eval_ce_alignment(model: CEAlignmentInformation, loader: DataLoader, wandb_prefix: Optional[str]=None, step_offset=0):
    """Evaluate a trained CEAlignmentInformation model. Uses global_device and model's internal AMP."""
    model.eval()
    model.to(global_device) # Ensure model on correct device
    all_losses = []
    all_pid_vals = []

    # model.use_amp and model.scaler are handled internally by CEAlignmentInformation
    # based on GLOBAL_USE_AMP during its initialization.
    # The forward pass model(x1, x2, y) uses internal autocast.
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            x1_batch, x2_batch, y_batch_orig = batch_data
            x1_batch, x2_batch, y_batch_orig = x1_batch.to(global_device), x2_batch.to(global_device), y_batch_orig.to(global_device)
            
            loss, pid_vals, _ = model(x1_batch, x2_batch, y_batch_orig)
            
            all_losses.append(loss.item())
            all_pid_vals.append(pid_vals.cpu().numpy()) # Store PID components as numpy array
    
    avg_loss = np.mean(all_losses) if all_losses else np.nan
    avg_pid_vals = np.mean(np.array(all_pid_vals), axis=0) if all_pid_vals else np.zeros(4)
    
    results = {
        "eval_loss": avg_loss,
        "redundancy": avg_pid_vals[0],
        "unique1": avg_pid_vals[1],
        "unique2": avg_pid_vals[2],
        "synergy": avg_pid_vals[3]
    }
    print(f"Eval CE Alignment ({wandb_prefix if wandb_prefix else 'Unknown'}): Loss: {avg_loss:.4f}, Red: {avg_pid_vals[0]:.4f}, U1: {avg_pid_vals[1]:.4f}, U2: {avg_pid_vals[2]:.4f}, Syn: {avg_pid_vals[3]:.4f}")
    
    if HAS_WANDB and wandb_prefix and wandb.run:
        # Log individual PID components and the loss
        wandb.log({f"{wandb_prefix}/{k}": v for k, v in results.items()})
        # wandb.log({f"{wandb_prefix}/avg_pid_redundancy": avg_pid_vals[0], ...etc }, step=step_offset)
        # wandb.log({f"{wandb_prefix}/eval_loss": avg_loss}, step=step_offset) # Example for step based logging

    return results # Return dictionary of results

# analyze_multiple_models function (moved from analyze_pid_new.py)
def analyze_multiple_models(
    checkpoint_dir: str,
    domain_modules: Dict[str, Any], 
    output_dir: str,
    source_config: Dict[str, str],
    target_config: str,
    synthetic_labels: Optional[torch.Tensor] = None, 
    n_samples: int = 10000,
    batch_size: int = 128,
    num_clusters: int = 10,
    discrim_epochs: int = 40,
    ce_epochs: int = 10,
    discrim_hidden_dim: int = 64,
    discrim_layers: int = 5,
    use_wandb: bool = True,
    wandb_project: str = "pid-analysis-multiple",
    wandb_entity: Optional[str] = None,
    data_module=None,
    dataset_split: str = "test",
    use_gw_encoded: bool = False,
    use_compile_torch: bool = True,
    ce_test_mode_run: bool = False,
    max_test_examples_run: int = 3000,
    auto_find_lr_run: bool = False,
    lr_finder_steps_run: int = 200,
    lr_start_run: float = 1e-7,
    lr_end_run: float = 1.0,
    cluster_method_discrim: str = 'gmm',
    enable_extended_metrics_discrim: bool = True,
    run_critic_ce_direct_multi: bool = False, # Flag for analyze_multiple_models
    max_models_to_analyze: Optional[int] = None, # New param to limit number of models
    find_latest_checkpoints_func: Optional[Callable[[str, Optional[int]], List[str]]] = None
) -> List[Dict[str, Any]]:
    
    if find_latest_checkpoints_func is None:
        from .utils import find_latest_model_checkpoints # Default implementation
        find_latest_checkpoints_func = find_latest_model_checkpoints

    checkpoint_paths = find_latest_checkpoints_func(checkpoint_dir, max_models_to_analyze)
    if not checkpoint_paths:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    all_results = []
    for model_idx, model_path in enumerate(checkpoint_paths):
        print(f"\n--- Analyzing model {model_idx + 1}/{len(checkpoint_paths)}: {model_path} ---")
        
        # Ensure a new W&B run for each model if wandb is enabled
        # Wandb run handling is now inside analyze_model
        
        # If synthetic_labels are generated once, they should be passed to each analyze_model call.
        # If they depend on the model, generation logic would be inside analyze_model or its data prep steps.
        # For now, assuming synthetic_labels are global if provided, or handled by analyze_model if None.

        current_model_output_dir = Path(output_dir) / Path(model_path).stem
        current_model_output_dir.mkdir(parents=True, exist_ok=True)

        analysis_result = analyze_model(
            model_path=str(model_path),
            domain_modules=domain_modules,
            output_dir=str(current_model_output_dir),
            source_config=source_config,
            target_config=target_config,
            synthetic_labels=synthetic_labels.clone() if synthetic_labels is not None else None, # Pass a clone if tensor
            n_samples=n_samples,
            batch_size=batch_size,
            num_clusters=num_clusters,
            discrim_epochs=discrim_epochs,
            ce_epochs=ce_epochs,
            discrim_hidden_dim=discrim_hidden_dim,
            discrim_layers=discrim_layers,
            use_wandb=use_wandb, # This will init a new run per model inside analyze_model
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            data_module=data_module,
            dataset_split=dataset_split,
            use_gw_encoded=use_gw_encoded,
            use_compile_torch=use_compile_torch,
            ce_test_mode_run=ce_test_mode_run,
            max_test_examples_run=max_test_examples_run,
            auto_find_lr_run=auto_find_lr_run,
            lr_finder_steps_run=lr_finder_steps_run,
            lr_start_run=lr_start_run,
            lr_end_run=lr_end_run,
            cluster_method_discrim=cluster_method_discrim,
            enable_extended_metrics_discrim=enable_extended_metrics_discrim,
            run_critic_ce_direct=run_critic_ce_direct_multi # Use the flag for multiple models
        )
        all_results.append(analysis_result)
    
    return all_results 

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
    from datetime import datetime
    
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
    from .utils import load_domain_modules
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
            from .utils import load_checkpoint, generate_samples_from_model
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
            
            # This function requires synthetic_labels to be provided externally
            # as we've removed synthetic label generation functionality per user requirements
            print("ERROR: analyze_multiple_models_from_list requires external synthetic labels")
            print("This function cannot work without synthetic label generation which was removed per user requirements")
            
            # Create a placeholder result to maintain consistency
            model_result = {
                "model_index": i,
                "model_path": checkpoint_path,
                "domain_names": domain_names,
                "source_config": source_config,
                "target_config": target_config,
                "error": "synthetic_labels_required",
                "pid_values": [0.0, 0.0, 0.0, 0.0]  # Placeholder values
            }
            results.append(model_result)
            continue
            
            # This code block was removed since we can't generate synthetic labels
            # The function will return results with error flags instead
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error analyzing model {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to file
    results_file = os.path.join(output_dir, "multiple_models_results.json")
    try:
        from .utils import prepare_for_json
        with open(results_file, 'w') as f:
            json.dump(prepare_for_json(results), f, indent=2)
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    # Finish wandb run
    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Failed to finish W&B run: {e}")
    
    return results 

def analyze_with_data_interface(
    data_interface: GeneralizedDataInterface,
    source_config: Dict[str, str],
    target_config: str,
    output_dir: str,
    n_samples: int = 10000,
    batch_size: int = 128,
    num_clusters: int = 10,
    discrim_epochs: int = 40,
    ce_epochs: int = 10,
    discrim_hidden_dim: int = 64,
    discrim_layers: int = 5,
    use_wandb: bool = True,
    wandb_project: str = "pid-analysis",
    wandb_entity: Optional[str] = None,
    synthetic_labels: Optional[torch.Tensor] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze PID using a generalized data interface.
    
    Args:
        data_interface: GeneralizedDataInterface instance
        source_config: Configuration for source domains
        target_config: Target domain name
        output_dir: Output directory for results
        n_samples: Number of samples to use
        batch_size: Batch size for training
        num_clusters: Number of clusters for synthetic labels
        discrim_epochs: Epochs for discriminator training
        ce_epochs: Epochs for CE alignment training
        discrim_hidden_dim: Hidden dimension for discriminator
        discrim_layers: Number of layers for discriminator
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        synthetic_labels: Pre-computed synthetic labels
        **kwargs: Additional arguments
    
    Returns:
        Dictionary with PID analysis results
    """
    import os
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data using the interface
    train_ds, test_ds, x1, x2, labels = data_interface.prepare_pid_data(
        source_config=source_config,
        target_config=target_config,
        n_samples=n_samples,
        synthetic_labels=synthetic_labels,
        num_clusters=num_clusters,
        batch_size=batch_size  # Pass to provider if needed
    )
    
    # Initialize wandb if requested
    wandb_run = None
    if use_wandb and HAS_WANDB:
        metadata = data_interface.get_metadata()
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                'n_samples': n_samples,
                'batch_size': batch_size,
                'num_clusters': num_clusters,
                'discrim_epochs': discrim_epochs,
                'ce_epochs': ce_epochs,
                'discrim_hidden_dim': discrim_hidden_dim,
                'discrim_layers': discrim_layers,
                'source_config': source_config,
                'target_config': target_config,
                'data_metadata': metadata
            }
        )
    
    # Run PID analysis using existing critic_ce_alignment function
    from .train import critic_ce_alignment
    
    pid_results = critic_ce_alignment(
        x1=x1,
        x2=x2,
        labels=labels,
        num_labels=num_clusters,
        train_ds=train_ds,
        test_ds=test_ds,
        discrim_epochs=discrim_epochs,
        ce_epochs=ce_epochs,
        wandb_enabled=use_wandb and HAS_WANDB,
        model_name=f"data_interface_{target_config}",
        discrim_hidden_dim=discrim_hidden_dim,
        discrim_layers=discrim_layers,
        **{k: v for k, v in kwargs.items() if k in [
            'use_compile', 'test_mode', 'max_test_examples', 
            'auto_find_lr', 'lr_finder_steps', 'lr_start', 'lr_end'
        ]}
    )
    
    # Add metadata to results
    result = {
        'pid_results': pid_results,
        'metadata': data_interface.get_metadata(),
        'config': {
            'source_config': source_config,
            'target_config': target_config,
            'n_samples': n_samples,
            'num_clusters': num_clusters,
            'discrim_epochs': discrim_epochs,
            'ce_epochs': ce_epochs
        }
    }
    
    # Save results
    results_file = output_path / f"pid_results_{target_config}.json"
    with open(results_file, 'w') as f:
        json.dump(prepare_for_json(result), f, indent=2)
    
    if wandb_run:
        wandb_run.finish()
    
    return result 