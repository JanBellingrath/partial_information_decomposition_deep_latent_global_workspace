import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable
import os
import json
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
import hashlib

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
from .train import train_discrim, train_ce_alignment, critic_ce_alignment  # Use the newer implementations from local train.py
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
    joint_discrim_layers: int = None,
    joint_discrim_hidden_dim: int = None,
    use_wandb: bool = True,
    wandb_project: str = "pid-analysis",
    wandb_entity: Optional[str] = None,
    data_module=None, 
    dataset_split: str = "train",  # Changed default from "test" to "train" for main analysis
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
    run_critic_ce_direct: bool = False, # New flag to run critic_ce_alignment directly
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a single model checkpoint.
    Can run either the original separate DINO + CE training, 
    or the combined critic_ce_alignment.
    """
    # Set default for joint_discrim_layers if not specified
    if joint_discrim_layers is None:
        joint_discrim_layers = discrim_layers
    
    # Set default for joint_discrim_hidden_dim if not specified
    if joint_discrim_hidden_dim is None:
        joint_discrim_hidden_dim = discrim_hidden_dim
    
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
    
    # Auto-generate synthetic labels if not provided
    if synthetic_labels is None:
        print("🧮 Generating synthetic labels from model data...")
        
        # Use the enhanced caching function from main.py
        from .main import load_or_generate_synthetic_labels
        
        synthetic_labels = load_or_generate_synthetic_labels(
            model_path=model_path,
            generated_data=generated_data,
            target_config=target_config,
            num_clusters=num_clusters,
            cluster_method=cluster_method_discrim,
            n_samples=n_samples,
            source_config=source_config,
            dataset_split=dataset_split,
            use_gw_encoded=use_gw_encoded,
            synthetic_labels_path=None,  # No specific path, use caching system
            force_regenerate=kwargs.get('force_regenerate_labels', False) or kwargs.get('force_retrain', False)
        )
    else:
        # synthetic_labels is provided as a string path
        print(f"🧮 Loading or generating synthetic labels from: {synthetic_labels}")
        
        # Use the enhanced caching function from main.py
        from .main import load_or_generate_synthetic_labels
        
        synthetic_labels = load_or_generate_synthetic_labels(
            model_path=model_path,
            generated_data=generated_data,
            target_config=target_config,
            num_clusters=num_clusters,
            cluster_method=cluster_method_discrim,
            n_samples=n_samples,
            source_config=source_config,
            dataset_split=dataset_split,
            use_gw_encoded=use_gw_encoded,
            synthetic_labels_path=synthetic_labels,  # Pass the user-provided path
            force_regenerate=kwargs.get('force_regenerate_labels', False) or kwargs.get('force_retrain', False)
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
            joint_discrim_layers=joint_discrim_layers, joint_discrim_hidden_dim=joint_discrim_hidden_dim,
            use_compile=use_compile_torch, 
            test_mode=ce_test_mode_run, max_test_examples=max_test_examples_run,
            auto_find_lr=auto_find_lr_run, lr_finder_steps=lr_finder_steps_run,
            lr_start=lr_start_run, lr_end=lr_end_run,
            enable_extended_metrics=enable_extended_metrics_discrim
        )
        pid_results = {
            "redundancy": avg_pid_vals[0].item(), "unique1": avg_pid_vals[1].item(),
            "unique2": avg_pid_vals[2].item(), "synergy": avg_pid_vals[3].item()
        }
        final_models = models_tuple # (ce_model, d1, d2, d12, p_y_calc)
        # No separate eval functions needed here as critic_ce_alignment includes an eval step.

    else: # Original separate training and evaluation path
        print("\nRunning PID analysis with separate DINO training and CE alignment training...")
        
        # Create discriminator cache directory and compute cache parameters
        model_dir = Path(model_path).parent
        model_name = Path(model_path).stem
        discrim_cache_dir = model_dir / "discrim_cache"
        discrim_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create deterministic cache key from training data
        train_data_sample = []
        for batch in DataLoader(train_ds, batch_size=min(100, batch_size), shuffle=False):
            train_data_sample.append(torch.cat([batch[0], batch[1], batch[2]], dim=1))
            if len(train_data_sample) >= 5:  # Use first 5 batches for hash
                break
        
        if train_data_sample:
            train_data_hash = hashlib.md5(torch.cat(train_data_sample)[:500].cpu().numpy().tobytes()).hexdigest()[:8]
        else:
            train_data_hash = "no_data"
        
        # Cache filename template
        def get_discrim_cache_path(discrim_type: str) -> Path:
            if discrim_type == "d12":
                # Use joint_discrim_layers and joint_discrim_hidden_dim for joint discriminator
                cache_filename = f"{model_name}_{discrim_type}_h{joint_discrim_hidden_dim}_l{joint_discrim_layers}_e{discrim_epochs}_s{n_samples}_c{num_clusters}_{cluster_method_discrim}_d{x1_data.size(1)}x{x2_data.size(1)}_comp{int(use_compile_torch)}_{train_data_hash}.pt"
            else:
                # Use regular discrim_layers and discrim_hidden_dim for individual discriminators
                input_dim = x1_data.size(1) if discrim_type == "d1" else x2_data.size(1)
                cache_filename = f"{model_name}_{discrim_type}_h{discrim_hidden_dim}_l{discrim_layers}_e{discrim_epochs}_s{n_samples}_c{num_clusters}_{cluster_method_discrim}_d{input_dim}_comp{int(use_compile_torch)}_{train_data_hash}.pt"
            return discrim_cache_dir / cache_filename
        
        # Create better wandb prefix structure with actual domain names
        domain1_name = domain_names[0] if len(domain_names) > 0 else "domain1"
        domain2_name = domain_names[1] if len(domain_names) > 1 else "domain2"
        
        # ========================================
        # 🔥 DISCRIMINATOR 1 TRAINING 
        # ========================================
        print(f"\n" + "="*80)
        print(f"🧠 TRAINING DISCRIMINATOR 1 ({domain1_name}) - {discrim_epochs} epochs")
        print(f"   Input: {domain1_name} features → Labels")
        print(f"   Architecture: {x1_data.size(1)} → {discrim_hidden_dim} → {num_clusters}")
        print(f"   Wandb prefix: discriminator_1/{model_name}")
        print("="*80)
        
        # Train or load discriminator 1
        discrim_1_cache_path = get_discrim_cache_path("d1")
        force_retrain_discriminators = kwargs.get('force_retrain_discriminators', False) or kwargs.get('force_retrain', False)
        
        if discrim_1_cache_path.exists() and not force_retrain_discriminators:
            try:
                print(f"🔄 Loading cached discriminator 1 from: {discrim_1_cache_path}")
                d1 = torch.load(discrim_1_cache_path, map_location=str(global_device))
                d1.to(global_device)
                print(f"✅ Successfully loaded cached discriminator 1")
            except Exception as e:
                print(f"⚠️  Failed to load cached discriminator 1: {e}, training new one...")
                d1 = None
        elif force_retrain_discriminators:
            print(f"🔄 Force retraining discriminator 1 (cache skipped)")
            d1 = None
        else:
            d1 = None
        
        if d1 is None:
            print(f"🔄 Training new discriminator 1...")
            d1 = Discrim(x1_data.size(1), discrim_hidden_dim, num_clusters, layers=discrim_layers, activation="relu").to(global_device)
            opt1 = torch.optim.Adam(d1.parameters(), lr=1e-3)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            d1 = train_discrim(d1, train_dl, opt1, ([0], [2]), 
                num_epoch=discrim_epochs, wandb_prefix=f"discriminator_1/{model_name}" if use_wandb and HAS_WANDB else None,
                use_compile=use_compile_torch, cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim)
            
            # Save to cache
            try:
                torch.save(d1, discrim_1_cache_path)
                print(f"💾 Cached discriminator 1 to: {discrim_1_cache_path}")
                
                # Save comprehensive metadata
                metadata = {
                    "discriminator_type": "d1",
                    "model_path": str(model_path),
                    "model_name": model_name,
                    "input_dim": x1_data.size(1),
                    "hidden_dim": discrim_hidden_dim,
                    "layers": discrim_layers,
                    "num_clusters": num_clusters,
                    "discrim_epochs": discrim_epochs,
                    "n_samples": n_samples,
                    "cluster_method": cluster_method_discrim,
                    "use_compile": use_compile_torch,
                    "train_data_hash": train_data_hash,
                    "generated_timestamp": str(datetime.now()),
                    "architecture_summary": f"input({x1_data.size(1)}) -> hidden({discrim_hidden_dim}) -> output({num_clusters})",
                    "activation": "relu"
                }
                metadata_path = discrim_1_cache_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"📝 Saved discriminator 1 metadata to: {metadata_path.name}")
            except Exception as e:
                print(f"⚠️  Failed to cache discriminator 1: {e}")
        
        # ========================================
        # 🔥 DISCRIMINATOR 2 TRAINING
        # ========================================
        print(f"\n" + "="*80)
        print(f"🧠 TRAINING DISCRIMINATOR 2 ({domain2_name}) - {discrim_epochs} epochs")
        print(f"   Input: {domain2_name} features → Labels")
        print(f"   Architecture: {x2_data.size(1)} → {discrim_hidden_dim} → {num_clusters}")
        print(f"   Wandb prefix: discriminator_2/{model_name}")
        print("="*80)
        
        # Train or load discriminator 2
        discrim_2_cache_path = get_discrim_cache_path("d2")
        if discrim_2_cache_path.exists() and not force_retrain_discriminators:
            try:
                print(f"🔄 Loading cached discriminator 2 from: {discrim_2_cache_path}")
                d2 = torch.load(discrim_2_cache_path, map_location=str(global_device))
                d2.to(global_device)
                print(f"✅ Successfully loaded cached discriminator 2")
            except Exception as e:
                print(f"⚠️  Failed to load cached discriminator 2: {e}, training new one...")
                d2 = None
        elif force_retrain_discriminators:
            print(f"🔄 Force retraining discriminator 2 (cache skipped)")
            d2 = None
        else:
            d2 = None
        
        if d2 is None:
            print(f"🔄 Training new discriminator 2...")
            d2 = Discrim(x2_data.size(1), discrim_hidden_dim, num_clusters, layers=discrim_layers, activation="relu").to(global_device)
            opt2 = torch.optim.Adam(d2.parameters(), lr=1e-3)
            d2 = train_discrim(d2, train_dl, opt2, ([1], [2]), 
                num_epoch=discrim_epochs, wandb_prefix=f"discriminator_2/{model_name}" if use_wandb and HAS_WANDB else None,
                use_compile=use_compile_torch, cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim)
            
            # Save to cache
            try:
                torch.save(d2, discrim_2_cache_path)
                print(f"💾 Cached discriminator 2 to: {discrim_2_cache_path}")
            except Exception as e:
                print(f"⚠️  Failed to cache discriminator 2: {e}")

        # ========================================
        # 🔥 JOINT DISCRIMINATOR TRAINING
        # ========================================
        print(f"\n" + "="*80)
        print(f"🧠 TRAINING JOINT DISCRIMINATOR ({domain1_name}+{domain2_name}) - {discrim_epochs} epochs")
        print(f"   Input: Combined features → Labels")
        print(f"   Architecture: {x1_data.size(1) + x2_data.size(1)} → {joint_discrim_hidden_dim} → {num_clusters}")
        print(f"   Wandb prefix: discriminator_joint/{model_name}")
        print("="*80)

        # Train or load discriminator 12
        discrim_12_cache_path = get_discrim_cache_path("d12")
        if discrim_12_cache_path.exists() and not force_retrain_discriminators:
            try:
                print(f"🔄 Loading cached discriminator 12 from: {discrim_12_cache_path}")
                d12 = torch.load(discrim_12_cache_path, map_location=str(global_device))
                d12.to(global_device)
                print(f"✅ Successfully loaded cached discriminator 12")
            except Exception as e:
                print(f"⚠️  Failed to load cached discriminator 12: {e}, training new one...")
                d12 = None
        elif force_retrain_discriminators:
            print(f"🔄 Force retraining discriminator 12 (cache skipped)")
            d12 = None
        else:
            d12 = None
        
        if d12 is None:
            print(f"🔄 Training new discriminator 12...")
            d12 = Discrim(x1_data.size(1) + x2_data.size(1), joint_discrim_hidden_dim, num_clusters, layers=joint_discrim_layers, activation="relu").to(global_device)
            opt12 = torch.optim.Adam(d12.parameters(), lr=1e-3)
            d12 = train_discrim(d12, train_dl, opt12, ([0,1], [2]), 
                num_epoch=discrim_epochs, wandb_prefix=f"discriminator_joint/{model_name}" if use_wandb and HAS_WANDB else None,
                use_compile=use_compile_torch, cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim)
            
            # Save to cache
            try:
                torch.save(d12, discrim_12_cache_path)
                print(f"💾 Cached discriminator 12 to: {discrim_12_cache_path}")
                
                # Also save metadata about the discriminator training
                metadata = {
                    "model_path": str(model_path),
                    "model_name": model_name,
                    "discrim_hidden_dim": discrim_hidden_dim,
                    "discrim_layers": discrim_layers,
                    "joint_discrim_layers": joint_discrim_layers,
                    "joint_discrim_hidden_dim": joint_discrim_hidden_dim,
                    "discrim_epochs": discrim_epochs,
                    "n_samples": n_samples,
                    "num_clusters": num_clusters,
                    "train_data_hash": train_data_hash,
                    "x1_dim": x1_data.size(1),
                    "x2_dim": x2_data.size(1),
                    "generated_timestamp": str(datetime.now()),
                    "cluster_method": cluster_method_discrim
                }
                metadata_path = discrim_12_cache_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"📝 Saved discriminator metadata to: {metadata_path}")
            except Exception as e:
                print(f"⚠️  Failed to cache discriminator 12: {e}")

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
            d1, d2, d12, p_y_calc
        ).to(global_device)
        
        # ========================================
        # 🔥 CE ALIGNMENT TRAINING
        # ========================================
        print(f"\n" + "="*80)
        print(f"🔮 TRAINING CE ALIGNMENT NETWORK - {ce_epochs} epochs")
        print(f"   Purpose: Align conditional distributions between domains")
        print(f"   Architecture: Embedding alignment with PID calculation")
        print(f"   Wandb prefix: ce_alignment/{model_name}")
        print(f"   Wandb run active: {wandb.run is not None if HAS_WANDB else 'N/A'}")
        print("="*80)
        
        # Train CE alignment model with consistent field naming
        # CE alignment should start its own step counting from 0
        # The wandb_prefix already separates it from discriminator training
        
        ce_model = train_ce_alignment(
            ce_model, DataLoader(train_ds, batch_size=batch_size),
            torch.optim.Adam, num_epoch=ce_epochs,
            wandb_prefix=f"ce_alignment/{model_name}" if use_wandb and HAS_WANDB else None,
            step_offset=0,  # FIXED: CE alignment should start from step 0, not continue from discriminators
            use_compile=use_compile_torch, test_mode=ce_test_mode_run, max_test_examples=max_test_examples_run,
            auto_find_lr=auto_find_lr_run, lr_finder_steps=lr_finder_steps_run, lr_start=lr_start_run, lr_end=lr_end_run
        )

        pid_results = eval_ce_alignment( 
            ce_model, DataLoader(test_ds, batch_size=batch_size),
            wandb_prefix=f"ce_alignment/{model_name}" if use_wandb and HAS_WANDB else None
        )
        final_models = (ce_model, d1, d2, d12, p_y_calc)

    # ========================================
    # 📊 CLUSTER VISUALIZATION (OPTIONAL)
    # ========================================
    visualization_results = None
    if 'visualize_clusters' in kwargs and kwargs['visualize_clusters']:
        print(f"\n" + "="*80)
        print(f"🎨 GENERATING CLUSTER VISUALIZATIONS FOR MULTIPLE SPLITS")
        print(f"   Grid size: {kwargs.get('viz_grid_size', 10)}×{kwargs.get('viz_grid_size', 10)}")
        print(f"   Samples per cluster: {kwargs.get('viz_samples_per_cluster', 100)}")
        print(f"   Max clusters: {kwargs.get('viz_max_clusters', 20)}")
        print(f"="*80)
        
        # Define splits to visualize
        visualization_splits = ['val']
        all_visualization_results = {}
        
        for split in visualization_splits:
            print(f"\n🔍 PROCESSING SPLIT: {split.upper()}")
            print("─" * 60)
            
            try:
                # Generate samples for this specific split
                print(f"📊 Generating samples from {split} split...")
                split_generated_data = generate_samples_from_model(
                    model=model_gw, 
                    domain_names=domain_names, 
                    n_samples=n_samples, 
                    batch_size=batch_size,
                    device=str(global_device), 
                    use_gw_encoded=use_gw_encoded, 
                    data_module=data_module, 
                    dataset_split=split
                )
                
                # Use the same clustering model/labels for consistency across splits
                # but apply to the new split's data
                split_target_data = split_generated_data[target_config]
                print(f"   └── Target data shape for {split}: {split_target_data.shape}")
                
                # Apply the same clustering model to assign clusters to the new split
                print(f"🔄 Assigning {split} samples to existing clusters...")
                
                # Try to load the pre-trained clustering model first
                clusterer_cache_path = None
                clustering_model = None
                
                # Look for the cached clustering model
                model_dir = Path(model_path).parent
                model_name = Path(model_path).stem
                
                # Create cache path pattern (matching the one used in main.py)
                target_data = generated_data[target_config]
                source_str = "none"
                if source_config:
                    source_items = sorted(source_config.items())
                    source_str = "_".join([f"{k}-{v}" for k, v in source_items])
                    source_str = source_str.replace("/", "-").replace(":", "-").replace(" ", "")
                
                data_shape_str = f"{target_data.shape[0]}x{target_data.shape[1]}"
                cache_filename = (f"{model_name}_synthetic_labels_"
                                 f"{target_config}_{num_clusters}_{cluster_method_discrim}_"
                                 f"samples{n_samples or data_shape_str}_"
                                 f"src{hash(source_str) % 10000:04d}_"
                                 f"split{dataset_split}_"
                                 f"gw{int(use_gw_encoded)}.clusterer.pkl")
                
                clusterer_cache_path = model_dir / cache_filename
                
                # Try to load the pre-trained clustering model
                if clusterer_cache_path.exists():
                    try:
                        from .synthetic_data import load_clustering_model, apply_clustering_model
                        clustering_model = load_clustering_model(str(clusterer_cache_path))
                        print(f"   ✅ Loaded pre-trained clustering model from cache")
                        
                        # Apply the loaded model to the split data
                        split_cluster_labels_tensor = apply_clustering_model(
                            clustering_model, split_target_data, cluster_method_discrim
                        )
                        
                        if cluster_method_discrim == 'kmeans':
                            split_cluster_labels = split_cluster_labels_tensor.cpu().numpy()
                        else:  # GMM - convert probabilities to hard labels for visualization
                            split_cluster_labels = split_cluster_labels_tensor.argmax(dim=1).cpu().numpy()
                            
                    except Exception as e:
                        print(f"   ⚠️  Failed to load cached clustering model: {e}")
                        clustering_model = None
                
                # Fallback to recreating the clustering model (old method)
                if clustering_model is None:
                    print(f"   🔄 Fallback: Recreating clustering model with same parameters...")
                    # For consistency, we use the same synthetic_labels clustering parameters
                    # but apply them to the split data
                    if cluster_method_discrim == 'kmeans':
                        from sklearn.cluster import KMeans
                        # Recreate the clustering model with same parameters
                        clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                        # Fit on original target data to maintain consistency
                        original_target_data = generated_data[target_config]
                        clusterer.fit(original_target_data.detach().cpu().numpy())
                        # Apply to split data
                        split_cluster_labels = clusterer.predict(split_target_data.detach().cpu().numpy())
                    else:  # GMM
                        from sklearn.mixture import GaussianMixture
                        # Recreate the clustering model with same parameters
                        clusterer = GaussianMixture(n_components=num_clusters, random_state=42)
                        # Fit on original target data to maintain consistency
                        original_target_data = generated_data[target_config]
                        clusterer.fit(original_target_data.detach().cpu().numpy())
                        # Apply to split data
                        split_cluster_labels = clusterer.predict(split_target_data.detach().cpu().numpy())
                
                print(f"   └── Assigned {len(split_cluster_labels)} {split} samples to {len(np.unique(split_cluster_labels))} clusters")
                
                # Run cluster visualization pipeline for this split
                from .cluster_visualization import run_cluster_visualization_pipeline
                
                split_wandb_prefix = f"cluster_visualization_{split}"
                print(f"🎨 Creating visualizations for {split} split (wandb prefix: {split_wandb_prefix})...")
                
                # Create synthetic labels tensor for this split
                split_synthetic_labels = torch.from_numpy(split_cluster_labels).to(global_device)
                
                # Extract data_module from data_interface if available for actual image visualization
                data_module_for_viz = None
                if hasattr(data_module, 'data_provider') and hasattr(data_module.data_provider, 'data_module'):
                    data_module_for_viz = data_module.data_provider.data_module
                
                visualization_results = run_cluster_visualization_pipeline(
                    model_path=model_path,
                    domain_modules=domain_modules,
                    generated_data=split_generated_data,
                    domain_names=domain_names,
                    source_config=source_config,
                    target_config=target_config,
                    synthetic_labels=split_synthetic_labels,
                    cluster_method=cluster_method_discrim,
                    num_clusters=num_clusters,
                    grid_size=kwargs.get('viz_grid_size', 10),
                    samples_per_cluster=kwargs.get('viz_samples_per_cluster', 100),
                    max_clusters=kwargs.get('viz_max_clusters', 20),
                    device=str(global_device),
                    use_wandb=use_wandb,
                    wandb_prefix=split_wandb_prefix,
                    data_module=data_module_for_viz,
                    dataset_split=split
                )
                
                all_visualization_results[split] = visualization_results
                print(f"✅ Completed {split} split visualization!")
                
                # Verify wandb logging
                if HAS_WANDB and wandb.run is not None:
                    print(f"🔍 Wandb verification for {split}:")
                    print(f"   └── Active run: {wandb.run.name}")
                    print(f"   └── Prefix used: {split_wandb_prefix}")
                    print(f"   └── Expected keys: {split_wandb_prefix}/vision_cluster_*, {split_wandb_prefix}/text_cluster_*")
                    
                    # Log a summary for this split
                    total_viz = visualization_results.get('total_visualizations', 0) if 'error' not in visualization_results else 0
                    wandb.log({
                        f"{split_wandb_prefix}/split_summary_total_visualizations": total_viz,
                        f"{split_wandb_prefix}/split_summary_split_name": split,
                        f"{split_wandb_prefix}/split_summary_samples_processed": len(split_cluster_labels)
                    })
                    print(f"   └── Logged summary stats to wandb")
                else:
                    print(f"⚠️  Wandb not available for {split} split logging")
                
            except Exception as e:
                print(f"❌ Error processing {split} split: {e}")
                import traceback
                traceback.print_exc()
                all_visualization_results[split] = {'error': str(e)}
        
        # Final summary
        print(f"\n🎉 CLUSTER VISUALIZATION COMPLETE FOR ALL SPLITS")
        successful_splits = [split for split, result in all_visualization_results.items() if 'error' not in result]
        failed_splits = [split for split, result in all_visualization_results.items() if 'error' in result]
        
        print(f"   ✅ Successful splits: {successful_splits}")
        if failed_splits:
            print(f"   ❌ Failed splits: {failed_splits}")
        
        if HAS_WANDB and wandb.run is not None:
            # Log overall summary
            wandb.log({
                "cluster_visualization_summary/successful_splits": len(successful_splits),
                "cluster_visualization_summary/failed_splits": len(failed_splits),
                "cluster_visualization_summary/total_splits_attempted": len(visualization_splits)
            })
            print(f"📊 Overall summary logged to wandb")
            
            # Print expected wandb structure
            print(f"\n📋 EXPECTED WANDB STRUCTURE:")
            for split in successful_splits:
                print(f"   cluster_visualization_{split}/")
                print(f"   ├── vision_cluster_0, vision_cluster_1, ...")
                print(f"   ├── text_cluster_0, text_cluster_1, ...")
                print(f"   └── split_summary_*")
            
        visualization_results = all_visualization_results

    # ========================================
    # 🧮 PID ANALYSIS 
    # ========================================

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
    
    # Basic metrics that are always calculated
    if all_logits and all_targets:
        logits_all_eval = torch.cat(all_logits)
        targets_all_eval_np = torch.cat(all_targets).numpy()
        
        # Ensure targets are 1D
        if targets_all_eval_np.ndim > 1 and targets_all_eval_np.shape[1] == 1:
            targets_all_eval_np = targets_all_eval_np.squeeze(-1)
        elif targets_all_eval_np.ndim > 1:
            targets_all_eval_np = np.argmax(targets_all_eval_np, axis=1)
            
        probs_all_eval_np = F.softmax(logits_all_eval, dim=1).numpy()
        preds_all_eval_np = probs_all_eval_np.argmax(axis=1)
        num_classes_eval = probs_all_eval_np.shape[1]
        
        # Basic metrics: accuracy and top-k accuracy
        accuracy = np.mean(preds_all_eval_np == targets_all_eval_np) if targets_all_eval_np.size > 0 else np.nan
        
        # Top-k accuracy
        top_k_val = 5
        top_k_accuracy = np.nan
        if logits_all_eval.shape[0] > 0:
            actual_k = min(top_k_val, num_classes_eval)
            if actual_k > 0:
                _, pred_top_k = torch.topk(logits_all_eval, actual_k, dim=1, largest=True, sorted=True)
                targets_torch_for_topk = torch.from_numpy(targets_all_eval_np).view(-1, 1).expand_as(pred_top_k)
                correct_k = torch.any(pred_top_k == targets_torch_for_topk, dim=1)
                top_k_accuracy = correct_k.float().mean().item()
        
        # Initialize results with basic metrics
        results = {
            "eval_loss": avg_loss,
            "accuracy": accuracy,
            f"top{top_k_val}_accuracy": top_k_accuracy
        }
        
        # Print basic metrics
        print_str = f"📊 Evaluation | AvgCritLoss: {avg_loss:.4f} Acc: {accuracy:.4f} Top{top_k_val}: {top_k_accuracy:.4f}"
        
        # Extended metrics (only if enabled)
        if enable_extended_metrics:
            # Import required libraries for extended metrics
            from sklearn.metrics import jaccard_score, precision_recall_fscore_support
            from scipy.stats import spearmanr, kendalltau
            from sklearn.calibration import calibration_curve
            
            # 1) Cross-entropy (log-loss)
            ce_loss_metric = F.cross_entropy(logits_all_eval, torch.from_numpy(targets_all_eval_np).long(), reduction='mean').item()
            
            # 2) KL divergence vs. one-hot targets
            one_hot_eval = np.eye(num_classes_eval)[targets_all_eval_np]
            kl_div_metric = np.mean(np.sum(one_hot_eval * (np.log(one_hot_eval + 1e-12) - np.log(probs_all_eval_np + 1e-12)), axis=1))
            
            # 3) Jaccard (for multiclass, average='macro')
            jaccard_metric = jaccard_score(targets_all_eval_np, preds_all_eval_np, average='macro', zero_division=0)
            
            # 4) Precision/Recall/F1 (micro-averaged)
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                targets_all_eval_np, preds_all_eval_np, average='micro', zero_division=0
            )
            
            # 5) Predictive-distribution entropy
            entropies_eval = -np.sum(probs_all_eval_np * np.log(probs_all_eval_np + 1e-12), axis=1)
            entropy_mean_metric = entropies_eval.mean() if entropies_eval.size > 0 else np.nan
            
            # 6) Brier score (for multiclass, use sum of squares)
            brier_metric = np.mean(np.sum((probs_all_eval_np - one_hot_eval)**2, axis=1)) if one_hot_eval.size > 0 else np.nan
            
            # 7) ECE: reliability curve
            ece_metric = np.nan
            max_probs_1d_eval = probs_all_eval_np.max(axis=1)
            targets_eval_np_1d = targets_all_eval_np.squeeze() if targets_all_eval_np.ndim > 1 else targets_all_eval_np
            
            if len(max_probs_1d_eval) == len(targets_eval_np_1d) and len(max_probs_1d_eval) > 0:
                try:
                    y_true_for_ece_eval = (preds_all_eval_np == targets_eval_np_1d).astype(int)
                    if len(np.unique(y_true_for_ece_eval)) >= 2:
                        rel_diag_fraction_of_positives_eval, rel_diag_mean_predicted_value_eval = calibration_curve(
                            y_true_for_ece_eval, max_probs_1d_eval, n_bins=10, strategy='uniform', normalize=True
                        )
                        
                        bins_ece_eval = np.linspace(0, 1, 11)
                        digitized_confidences_eval = np.digitize(max_probs_1d_eval, bins=bins_ece_eval[1:-1])
                        num_returned_bins_eval = len(rel_diag_mean_predicted_value_eval)
                        bin_sample_counts_eval = np.bincount(digitized_confidences_eval, minlength=max(1, num_returned_bins_eval))
                        bin_sample_counts_aligned_eval = bin_sample_counts_eval[:num_returned_bins_eval]
                        
                        if np.sum(bin_sample_counts_aligned_eval) > 0:
                            ece_metric = np.sum(np.abs(rel_diag_fraction_of_positives_eval - rel_diag_mean_predicted_value_eval) * bin_sample_counts_aligned_eval) / np.sum(bin_sample_counts_aligned_eval)
                except Exception as e_cal_eval:
                    print(f"Eval Warning: Could not compute ECE: {e_cal_eval}")
            
            # Add extended metrics to results
            results.update({
                "cross_entropy": ce_loss_metric,
                "kl_divergence": kl_div_metric,
                "jaccard": jaccard_metric,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "entropy_mean": entropy_mean_metric,
                "ece": ece_metric,
                "brier_score": brier_metric
            })
            
            # Extended print string
            print_str += f" || CE: {ce_loss_metric:.4f} KL: {kl_div_metric:.4f} Jaccard: {jaccard_metric:.4f} "
            print_str += f"Pₘᵢ𝚌ᵣₒ: {precision_micro:.4f} Rₘᵢ𝚌ᵣₒ: {recall_micro:.4f} F1ₘᵢ𝚌ᵣₒ: {f1_micro:.4f} "
            print_str += f"H(ent): {entropy_mean_metric:.4f} ECE: {ece_metric:.4f} Brier: {brier_metric:.4f}"
        
        print(print_str)
    else:
        results = {"eval_loss": avg_loss}
        print(f"Eval Discrim ({wandb_prefix if wandb_prefix else 'Unknown'}) - Loss: {avg_loss:.4f}")

    # Log to wandb 
    if HAS_WANDB and wandb_prefix and wandb.run:
        # Filter out NaN values for wandb logging
        log_dict_eval = {}
        for k, v in results.items():
            if not (isinstance(v, float) and np.isnan(v)):
                log_dict_eval[f"{wandb_prefix}/{k}"] = v
        wandb.log(log_dict_eval)
    
    return results

def eval_ce_alignment(model: CEAlignmentInformation, loader: DataLoader, wandb_prefix: Optional[str]=None):
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
    
    # Calculate results
    avg_loss = np.mean(all_losses) if all_losses else np.nan
    avg_pid_vals = np.mean(all_pid_vals, axis=0) if all_pid_vals else np.array([np.nan] * 4)
    
    results = {
        "avg_loss": avg_loss,
        "avg_pid_redundancy": avg_pid_vals[0],
        "avg_pid_unique1": avg_pid_vals[1], 
        "avg_pid_unique2": avg_pid_vals[2],
        "avg_pid_synergy": avg_pid_vals[3]
    }
    
    print(f"Eval CE Alignment ({wandb_prefix if wandb_prefix else 'Unknown'}) - Loss: {avg_loss:.4f}, "
          f"PID [R={avg_pid_vals[0]:.4f}, U1={avg_pid_vals[1]:.4f}, U2={avg_pid_vals[2]:.4f}, S={avg_pid_vals[3]:.4f}]")
    
    # Log to wandb without explicit step to avoid conflicts
    if HAS_WANDB and wandb_prefix and wandb.run:
        wandb.log({f"{wandb_prefix}/{k}": v for k,v in results.items()})
    
    return results

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
    joint_discrim_layers: int = None,
    joint_discrim_hidden_dim: int = None,
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
            joint_discrim_layers=joint_discrim_layers,
            joint_discrim_hidden_dim=joint_discrim_hidden_dim,
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
    joint_discrim_layers: int = None,
    joint_discrim_hidden_dim: int = None,
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
        joint_discrim_layers: Number of layers in joint discriminator network
        joint_discrim_hidden_dim: Hidden dimension for joint discriminator network
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
                    "joint_discrim_layers": joint_discrim_layers,
                    "joint_discrim_hidden_dim": joint_discrim_hidden_dim,
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
            
            # Auto-generate synthetic labels if not provided
            if synthetic_labels is None:
                print("🧮 Generating synthetic labels from model data...")
                
                # Use the enhanced caching function from main.py
                from .main import load_or_generate_synthetic_labels
                
                synthetic_labels = load_or_generate_synthetic_labels(
                    model_path=checkpoint_path,
                    generated_data=generated_data,
                    target_config=target_config,
                    num_clusters=num_clusters,
                    cluster_method=cluster_method_discrim,
                    n_samples=n_samples,
                    source_config=source_config,
                    dataset_split=dataset_split,
                    use_gw_encoded=use_gw_encoded,
                    synthetic_labels_path=None,  # No specific path, use caching system
                    force_regenerate=kwargs.get('force_regenerate_labels', False) or kwargs.get('force_retrain', False)
                )
                
                print(f"✅ Generated synthetic labels with shape: {synthetic_labels.shape}")

            # synthetic_labels are now ensured to be present (either passed or correctly generated)
            train_ds, test_ds, x1, x2, labels = prepare_pid_data(
                generated_data=generated_data, domain_names=domain_names,
                source_config=source_config, target_config=target_config, 
                synthetic_labels=synthetic_labels
            )
            # Move x1, x2, labels to global_device as they are used for CE model init / p_y calc
            x1, x2, labels = x1.to(global_device), x2.to(global_device), labels.to(global_device)

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
                joint_discrim_layers=joint_discrim_layers,
                joint_discrim_hidden_dim=joint_discrim_hidden_dim,
                enable_extended_metrics=kwargs.get('enable_extended_metrics', True),
                run_critic_ce_direct=kwargs.get('run_critic_ce_direct', False),
                **{k: v for k, v in kwargs.items() if k in [
                    'use_compile', 'test_mode', 'max_test_examples', 
                    'auto_find_lr', 'lr_finder_steps', 'lr_start', 'lr_end'
                ]}
            )
            
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
    joint_discrim_layers: int = None,
    joint_discrim_hidden_dim: int = None,
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
        joint_discrim_layers: Number of layers for joint discriminator
        joint_discrim_hidden_dim: Hidden dimension for joint discriminator
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        synthetic_labels: Pre-computed synthetic labels
        **kwargs: Additional arguments (including visualization options)
    
    Returns:
        Dictionary with PID analysis results
    """
    import os
    import numpy as np
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
                'joint_discrim_layers': joint_discrim_layers,
                'joint_discrim_hidden_dim': joint_discrim_hidden_dim,
                'source_config': source_config,
                'target_config': target_config,
                'data_metadata': metadata
            }
        )
    
    # ========================================
    # 📊 CLUSTER VISUALIZATION (OPTIONAL)
    # ========================================
    visualization_results = None
    if kwargs.get('visualize_clusters', False):
        print(f"\n" + "="*80)
        print(f"🎨 GENERATING CLUSTER VISUALIZATIONS")
        print(f"   Grid size: {kwargs.get('viz_grid_size', 10)}×{kwargs.get('viz_grid_size', 10)}")
        print(f"   Samples per cluster: {kwargs.get('viz_samples_per_cluster', 100)}")
        print(f"   Max clusters: {kwargs.get('viz_max_clusters', 20)}")
        print(f"="*80)
        
        try:
            # Get the generated data from the data interface
            generated_data = data_interface.data_provider.get_data(n_samples, batch_size=batch_size)
            domain_names = data_interface.data_provider.get_domain_names()
            
            # Extract data_module from data_interface if available for actual image visualization
            data_module_for_viz = None
            if hasattr(data_interface.data_provider, 'data_module'):
                data_module_for_viz = data_interface.data_provider.data_module
            
            # Run cluster visualization pipeline
            from .cluster_visualization import run_cluster_visualization_pipeline
            
            visualization_wandb_prefix = "cluster_visualization"
            print(f"🎨 Creating visualizations (wandb prefix: {visualization_wandb_prefix})...")
            
            visualization_results = run_cluster_visualization_pipeline(
                model_path=kwargs.get('model_path', 'data_interface'),
                domain_modules=kwargs.get('domain_modules', {}),
                generated_data=generated_data,
                domain_names=domain_names,
                source_config=source_config,
                target_config=target_config,
                synthetic_labels=labels,
                cluster_method=kwargs.get('cluster_method', 'gmm'),
                num_clusters=num_clusters,
                grid_size=kwargs.get('viz_grid_size', 10),
                samples_per_cluster=kwargs.get('viz_samples_per_cluster', 100),
                max_clusters=kwargs.get('viz_max_clusters', 20),
                device=str(global_device),
                use_wandb=use_wandb,
                wandb_prefix=visualization_wandb_prefix,
                data_module=data_module_for_viz,
                dataset_split=kwargs.get('dataset_split', 'train')
            )
            
            print(f"✅ Cluster visualization completed!")
            
            # Verify wandb logging
            if HAS_WANDB and wandb.run is not None:
                print(f"🔍 Wandb verification:")
                print(f"   └── Active run: {wandb.run.name}")
                print(f"   └── Prefix used: {visualization_wandb_prefix}")
                print(f"   └── Expected keys: {visualization_wandb_prefix}/vision_cluster_*, {visualization_wandb_prefix}/text_cluster_*")
                
                # Log a summary
                total_viz = visualization_results.get('total_visualizations', 0) if 'error' not in visualization_results else 0
                wandb.log({
                    f"{visualization_wandb_prefix}/summary_total_visualizations": total_viz,
                    f"{visualization_wandb_prefix}/summary_samples_processed": n_samples
                })
                print(f"   └── Logged summary stats to wandb")
            else:
                print(f"⚠️  Wandb not available for visualization logging")
                
        except Exception as e:
            print(f"❌ Error in cluster visualization: {e}")
            import traceback
            traceback.print_exc()
            visualization_results = {'error': str(e)}
    
    # ========================================
    # 🧮 PID ANALYSIS 
    # ========================================
    
    # Run PID analysis using existing critic_ce_alignment function
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
        joint_discrim_layers=joint_discrim_layers,
        joint_discrim_hidden_dim=joint_discrim_hidden_dim,
        enable_extended_metrics=kwargs.get('enable_extended_metrics', True),
        run_critic_ce_direct=kwargs.get('run_critic_ce_direct', False),
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
        },
        'visualization_results': visualization_results
    }
    
    # Save results
    results_file = output_path / f"pid_results_{target_config}.json"
    with open(results_file, 'w') as f:
        json.dump(prepare_for_json(result), f, indent=2)
    
    if wandb_run:
        wandb_run.finish()
    
    return result 

def save_discriminator_with_metadata(discriminator, cache_path: Path, discrim_type: str, input_dim: int, hidden_dim: int, layers: int):
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
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "num_clusters": num_clusters,
            "discrim_epochs": discrim_epochs,
            "n_samples": n_samples,
            "cluster_method": cluster_method_discrim,
            "use_compile": use_compile_torch,
            "train_data_hash": train_data_hash,
            "generated_timestamp": str(datetime.now()),
            "architecture_summary": f"input({input_dim}) -> hidden({hidden_dim}) -> output({num_clusters})",
            "activation": "relu"
        }
        
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
            if metadata.get('num_clusters') != num_clusters:
                validation_errors.append(f"num_clusters mismatch: cached={metadata.get('num_clusters')}, expected={num_clusters}")
            
            if validation_errors:
                print(f"⚠️  {discrim_type} validation failed: {'; '.join(validation_errors)}")
                print(f"   🔄 Will retrain {discrim_type} with new parameters")
                return None
            else:
                print(f"✅ {discrim_type} metadata validation passed")
        
        # Load the discriminator
        discriminator = torch.load(cache_path, map_location=str(global_device))
        discriminator.to(global_device)
        print(f"✅ Successfully loaded cached {discrim_type}")
        return discriminator
        
    except Exception as e:
        print(f"⚠️  Failed to load cached {discrim_type}: {e}, training new one...")
        return None

# Create better wandb prefix structure with actual domain names
domain1_name = domain_names[0] if len(domain_names) > 0 else "domain1"
domain2_name = domain_names[1] if len(domain_names) > 1 else "domain2"

# ========================================
# 🔥 DISCRIMINATOR 1 TRAINING 
# ========================================
print(f"\n" + "="*80)
print(f"🧠 TRAINING DISCRIMINATOR 1 ({domain1_name}) - {discrim_epochs} epochs")
print(f"   Input: {domain1_name} features → Labels")
print(f"   Architecture: {x1_data.size(1)} → {discrim_hidden_dim} → {num_clusters}")
print(f"   Wandb prefix: discriminator_1/{model_name}")
print("="*80)

# Train or load discriminator 1
discrim_1_cache_path = get_discrim_cache_path("d1")
force_retrain_discriminators = kwargs.get('force_retrain_discriminators', False) or kwargs.get('force_retrain', False)

if discrim_1_cache_path.exists() and not force_retrain_discriminators:
    try:
        print(f"🔄 Loading cached discriminator 1 from: {discrim_1_cache_path}")
        d1 = torch.load(discrim_1_cache_path, map_location=str(global_device))
        d1.to(global_device)
        print(f"✅ Successfully loaded cached discriminator 1")
    except Exception as e:
        print(f"⚠️  Failed to load cached discriminator 1: {e}, training new one...")
        d1 = None
elif force_retrain_discriminators:
    print(f"🔄 Force retraining discriminator 1 (cache skipped)")
    d1 = None
else:
    d1 = None

if d1 is None:
    print(f"🔄 Training new discriminator 1...")
    d1 = Discrim(x1_data.size(1), discrim_hidden_dim, num_clusters, layers=discrim_layers, activation="relu").to(global_device)
    opt1 = torch.optim.Adam(d1.parameters(), lr=1e-3)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    d1 = train_discrim(d1, train_dl, opt1, ([0], [2]), 
        num_epoch=discrim_epochs, wandb_prefix=f"discriminator_1/{model_name}" if use_wandb and HAS_WANDB else None,
        use_compile=use_compile_torch, cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim)
    
    # Save to cache
    try:
        torch.save(d1, discrim_1_cache_path)
        print(f"💾 Cached discriminator 1 to: {discrim_1_cache_path}")
        
        # Save comprehensive metadata
        metadata = {
            "discriminator_type": "d1",
            "model_path": str(model_path),
            "model_name": model_name,
            "input_dim": x1_data.size(1),
            "hidden_dim": discrim_hidden_dim,
            "layers": discrim_layers,
            "num_clusters": num_clusters,
            "discrim_epochs": discrim_epochs,
            "n_samples": n_samples,
            "cluster_method": cluster_method_discrim,
            "use_compile": use_compile_torch,
            "train_data_hash": train_data_hash,
            "generated_timestamp": str(datetime.now()),
            "architecture_summary": f"input({x1_data.size(1)}) -> hidden({discrim_hidden_dim}) -> output({num_clusters})",
            "activation": "relu"
        }
        metadata_path = discrim_1_cache_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Saved discriminator 1 metadata to: {metadata_path.name}")
    except Exception as e:
        print(f"⚠️  Failed to cache discriminator 1: {e}")

# ========================================
# 🔥 DISCRIMINATOR 2 TRAINING
# ========================================
print(f"\n" + "="*80)
print(f"🧠 TRAINING DISCRIMINATOR 2 ({domain2_name}) - {discrim_epochs} epochs")
print(f"   Input: {domain2_name} features → Labels")
print(f"   Architecture: {x2_data.size(1)} → {discrim_hidden_dim} → {num_clusters}")
print(f"   Wandb prefix: discriminator_2/{model_name}")
print("="*80)

# Train or load discriminator 2
discrim_2_cache_path = get_discrim_cache_path("d2")
if discrim_2_cache_path.exists() and not force_retrain_discriminators:
    try:
        print(f"🔄 Loading cached discriminator 2 from: {discrim_2_cache_path}")
        d2 = torch.load(discrim_2_cache_path, map_location=str(global_device))
        d2.to(global_device)
        print(f"✅ Successfully loaded cached discriminator 2")
    except Exception as e:
        print(f"⚠️  Failed to load cached discriminator 2: {e}, training new one...")
        d2 = None
elif force_retrain_discriminators:
    print(f"🔄 Force retraining discriminator 2 (cache skipped)")
    d2 = None
else:
    d2 = None

if d2 is None:
    print(f"🔄 Training new discriminator 2...")
    d2 = Discrim(x2_data.size(1), discrim_hidden_dim, num_clusters, layers=discrim_layers, activation="relu").to(global_device)
    opt2 = torch.optim.Adam(d2.parameters(), lr=1e-3)
    d2 = train_discrim(d2, train_dl, opt2, ([1], [2]), 
        num_epoch=discrim_epochs, wandb_prefix=f"discriminator_2/{model_name}" if use_wandb and HAS_WANDB else None,
        use_compile=use_compile_torch, cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim)
    
    # Save to cache
    try:
        torch.save(d2, discrim_2_cache_path)
        print(f"💾 Cached discriminator 2 to: {discrim_2_cache_path}")
    except Exception as e:
        print(f"⚠️  Failed to cache discriminator 2: {e}")

# ========================================
# 🔥 JOINT DISCRIMINATOR TRAINING
# ========================================
print(f"\n" + "="*80)
print(f"🧠 TRAINING JOINT DISCRIMINATOR ({domain1_name}+{domain2_name}) - {discrim_epochs} epochs")
print(f"   Input: Combined features → Labels")
print(f"   Architecture: {x1_data.size(1) + x2_data.size(1)} → {joint_discrim_hidden_dim} → {num_clusters}")
print(f"   Wandb prefix: discriminator_joint/{model_name}")
print("="*80)

# Train or load discriminator 12
discrim_12_cache_path = get_discrim_cache_path("d12")
if discrim_12_cache_path.exists() and not force_retrain_discriminators:
    try:
        print(f"🔄 Loading cached discriminator 12 from: {discrim_12_cache_path}")
        d12 = torch.load(discrim_12_cache_path, map_location=str(global_device))
        d12.to(global_device)
        print(f"✅ Successfully loaded cached discriminator 12")
    except Exception as e:
        print(f"⚠️  Failed to load cached discriminator 12: {e}, training new one...")
        d12 = None
elif force_retrain_discriminators:
    print(f"🔄 Force retraining discriminator 12 (cache skipped)")
    d12 = None
else:
    d12 = None

if d12 is None:
    print(f"🔄 Training new discriminator 12...")
    d12 = Discrim(x1_data.size(1) + x2_data.size(1), joint_discrim_hidden_dim, num_clusters, layers=joint_discrim_layers, activation="relu").to(global_device)
    opt12 = torch.optim.Adam(d12.parameters(), lr=1e-3)
    d12 = train_discrim(d12, train_dl, opt12, ([0,1], [2]), 
        num_epoch=discrim_epochs, wandb_prefix=f"discriminator_joint/{model_name}" if use_wandb and HAS_WANDB else None,
        use_compile=use_compile_torch, cluster_method=cluster_method_discrim, enable_extended_metrics=enable_extended_metrics_discrim)
    
    # Save to cache
    try:
        torch.save(d12, discrim_12_cache_path)
        print(f"💾 Cached discriminator 12 to: {discrim_12_cache_path}")
        
        # Also save metadata about the discriminator training
        metadata = {
            "model_path": str(model_path),
            "model_name": model_name,
            "discrim_hidden_dim": discrim_hidden_dim,
            "discrim_layers": discrim_layers,
            "joint_discrim_layers": joint_discrim_layers,
            "joint_discrim_hidden_dim": joint_discrim_hidden_dim,
            "discrim_epochs": discrim_epochs,
            "n_samples": n_samples,
            "num_clusters": num_clusters,
            "train_data_hash": train_data_hash,
            "x1_dim": x1_data.size(1),
            "x2_dim": x2_data.size(1),
            "generated_timestamp": str(datetime.now()),
            "cluster_method": cluster_method_discrim
        }
        metadata_path = discrim_12_cache_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Saved discriminator metadata to: {metadata_path}")
    except Exception as e:
        print(f"⚠️  Failed to cache discriminator 12: {e}")

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
    d1, d2, d12, p_y_calc
).to(global_device)

# ========================================
# 🔥 CE ALIGNMENT TRAINING
# ========================================
print(f"\n" + "="*80)
print(f"🔮 TRAINING CE ALIGNMENT NETWORK - {ce_epochs} epochs")
print(f"   Purpose: Align conditional distributions between domains")
print(f"   Architecture: Embedding alignment with PID calculation")
print(f"   Wandb prefix: ce_alignment/{model_name}")
print(f"   Wandb run active: {wandb.run is not None if HAS_WANDB else 'N/A'}")
print("="*80)

# Train CE alignment model with consistent field naming
# CE alignment should start its own step counting from 0
# The wandb_prefix already separates it from discriminator training

ce_model = train_ce_alignment(
    ce_model, DataLoader(train_ds, batch_size=batch_size),
    torch.optim.Adam, num_epoch=ce_epochs,
    wandb_prefix=f"ce_alignment/{model_name}" if use_wandb and HAS_WANDB else None,
    step_offset=0,  # FIXED: CE alignment should start from step 0, not continue from discriminators
    use_compile=use_compile_torch, test_mode=ce_test_mode_run, max_test_examples=max_test_examples_run,
    auto_find_lr=auto_find_lr_run, lr_finder_steps=lr_finder_steps_run, lr_start=lr_start_run, lr_end=lr_end_run
)

pid_results = eval_ce_alignment( 
    ce_model, DataLoader(test_ds, batch_size=batch_size),
    wandb_prefix=f"ce_alignment/{model_name}" if use_wandb and HAS_WANDB else None
)
final_models = (ce_model, d1, d2, d12, p_y_calc)

# ========================================
# 📊 CLUSTER VISUALIZATION (OPTIONAL)
# ========================================
visualization_results = None
if 'visualize_clusters' in kwargs and kwargs['visualize_clusters']:
    print(f"\n" + "="*80)
    print(f"🎨 GENERATING CLUSTER VISUALIZATIONS FOR MULTIPLE SPLITS")
    print(f"   Grid size: {kwargs.get('viz_grid_size', 10)}×{kwargs.get('viz_grid_size', 10)}")
    print(f"   Samples per cluster: {kwargs.get('viz_samples_per_cluster', 100)}")
    print(f"   Max clusters: {kwargs.get('viz_max_clusters', 20)}")
    print(f"="*80)
    
    # Define splits to visualize
    visualization_splits = ['val']
    all_visualization_results = {}
    
    for split in visualization_splits:
        print(f"\n🔍 PROCESSING SPLIT: {split.upper()}")
        print("─" * 60)
        
        try:
            # Generate samples for this specific split
            print(f"📊 Generating samples from {split} split...")
            split_generated_data = generate_samples_from_model(
                model=model_gw, 
                domain_names=domain_names, 
                n_samples=n_samples, 
                batch_size=batch_size,
                device=str(global_device), 
                use_gw_encoded=use_gw_encoded, 
                data_module=data_module, 
                dataset_split=split
            )
            
            # Use the same clustering model/labels for consistency across splits
            # but apply to the new split's data
            split_target_data = split_generated_data[target_config]
            print(f"   └── Target data shape for {split}: {split_target_data.shape}")
            
            # Apply the same clustering model to assign clusters to the new split
            print(f"🔄 Assigning {split} samples to existing clusters...")
            
            # Try to load the pre-trained clustering model first
            clusterer_cache_path = None
            clustering_model = None
            
            # Look for the cached clustering model
            model_dir = Path(model_path).parent
            model_name = Path(model_path).stem
            
            # Create cache path pattern (matching the one used in main.py)
            target_data = generated_data[target_config]
            source_str = "none"
            if source_config:
                source_items = sorted(source_config.items())
                source_str = "_".join([f"{k}-{v}" for k, v in source_items])
                source_str = source_str.replace("/", "-").replace(":", "-").replace(" ", "")
            
            data_shape_str = f"{target_data.shape[0]}x{target_data.shape[1]}"
            cache_filename = (f"{model_name}_synthetic_labels_"
                             f"{target_config}_{num_clusters}_{cluster_method_discrim}_"
                             f"samples{n_samples or data_shape_str}_"
                             f"src{hash(source_str) % 10000:04d}_"
                             f"split{dataset_split}_"
                             f"gw{int(use_gw_encoded)}.clusterer.pkl")
            
            clusterer_cache_path = model_dir / cache_filename
            
            # Try to load the pre-trained clustering model
            if clusterer_cache_path.exists():
                try:
                    from .synthetic_data import load_clustering_model, apply_clustering_model
                    clustering_model = load_clustering_model(str(clusterer_cache_path))
                    print(f"   ✅ Loaded pre-trained clustering model from cache")
                    
                    # Apply the loaded model to the split data
                    split_cluster_labels_tensor = apply_clustering_model(
                        clustering_model, split_target_data, cluster_method_discrim
                    )
                    
                    if cluster_method_discrim == 'kmeans':
                        split_cluster_labels = split_cluster_labels_tensor.cpu().numpy()
                    else:  # GMM - convert probabilities to hard labels for visualization
                        split_cluster_labels = split_cluster_labels_tensor.argmax(dim=1).cpu().numpy()
                        
                except Exception as e:
                    print(f"   ⚠️  Failed to load cached clustering model: {e}")
                    clustering_model = None
                
                # Fallback to recreating the clustering model (old method)
                if clustering_model is None:
                    print(f"   🔄 Fallback: Recreating clustering model with same parameters...")
                    # For consistency, we use the same synthetic_labels clustering parameters
                    # but apply them to the split data
                    if cluster_method_discrim == 'kmeans':
                        from sklearn.cluster import KMeans
                        # Recreate the clustering model with same parameters
                        clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                        # Fit on original target data to maintain consistency
                        original_target_data = generated_data[target_config]
                        clusterer.fit(original_target_data.detach().cpu().numpy())
                        # Apply to split data
                        split_cluster_labels = clusterer.predict(split_target_data.detach().cpu().numpy())
                    else:  # GMM
                        from sklearn.mixture import GaussianMixture
                        # Recreate the clustering model with same parameters
                        clusterer = GaussianMixture(n_components=num_clusters, random_state=42)
                        # Fit on original target data to maintain consistency
                        original_target_data = generated_data[target_config]
                        clusterer.fit(original_target_data.detach().cpu().numpy())
                        # Apply to split data
                        split_cluster_labels = clusterer.predict(split_target_data.detach().cpu().numpy())
                
                print(f"   └── Assigned {len(split_cluster_labels)} {split} samples to {len(np.unique(split_cluster_labels))} clusters")
                
                # Run cluster visualization pipeline for this split
                from .cluster_visualization import run_cluster_visualization_pipeline
                
                split_wandb_prefix = f"cluster_visualization_{split}"
                print(f"🎨 Creating visualizations for {split} split (wandb prefix: {split_wandb_prefix})...")
                
                # Create synthetic labels tensor for this split
                split_synthetic_labels = torch.from_numpy(split_cluster_labels).to(global_device)
                
                # Extract data_module from data_interface if available for actual image visualization
                data_module_for_viz = None
                if hasattr(data_module, 'data_provider') and hasattr(data_module.data_provider, 'data_module'):
                    data_module_for_viz = data_module.data_provider.data_module
                
                visualization_results = run_cluster_visualization_pipeline(
                    model_path=model_path,
                    domain_modules=domain_modules,
                    generated_data=split_generated_data,
                    domain_names=domain_names,
                    source_config=source_config,
                    target_config=target_config,
                    synthetic_labels=split_synthetic_labels,
                    cluster_method=cluster_method_discrim,
                    num_clusters=num_clusters,
                    grid_size=kwargs.get('viz_grid_size', 10),
                    samples_per_cluster=kwargs.get('viz_samples_per_cluster', 100),
                    max_clusters=kwargs.get('viz_max_clusters', 20),
                    device=str(global_device),
                    use_wandb=use_wandb,
                    wandb_prefix=split_wandb_prefix,
                    data_module=data_module_for_viz,
                    dataset_split=split
                )
                
                all_visualization_results[split] = visualization_results
                print(f"✅ Completed {split} split visualization!")
                
                # Verify wandb logging
                if HAS_WANDB and wandb.run is not None:
                    print(f"🔍 Wandb verification for {split}:")
                    print(f"   └── Active run: {wandb.run.name}")
                    print(f"   └── Prefix used: {split_wandb_prefix}")
                    print(f"   └── Expected keys: {split_wandb_prefix}/vision_cluster_*, {split_wandb_prefix}/text_cluster_*")
                    
                    # Log a summary for this split
                    total_viz = visualization_results.get('total_visualizations', 0) if 'error' not in visualization_results else 0
                    wandb.log({
                        f"{split_wandb_prefix}/split_summary_total_visualizations": total_viz,
                        f"{split_wandb_prefix}/split_summary_split_name": split,
                        f"{split_wandb_prefix}/split_summary_samples_processed": len(split_cluster_labels)
                    })
                    print(f"   └── Logged summary stats to wandb")
                else:
                    print(f"⚠️  Wandb not available for {split} split logging")
                
            except Exception as e:
                print(f"❌ Error processing {split} split: {e}")
                import traceback
                traceback.print_exc()
                all_visualization_results[split] = {'error': str(e)}
        
        # Final summary
        print(f"\n🎉 CLUSTER VISUALIZATION COMPLETE FOR ALL SPLITS")
        successful_splits = [split for split, result in all_visualization_results.items() if 'error' not in result]
        failed_splits = [split for split, result in all_visualization_results.items() if 'error' in result]
        
        print(f"   ✅ Successful splits: {successful_splits}")
        if failed_splits:
            print(f"   ❌ Failed splits: {failed_splits}")
        
        if HAS_WANDB and wandb.run is not None:
            # Log overall summary
            wandb.log({
                "cluster_visualization_summary/successful_splits": len(successful_splits),
                "cluster_visualization_summary/failed_splits": len(failed_splits),
                "cluster_visualization_summary/total_splits_attempted": len(visualization_splits)
            })
            print(f"📊 Overall summary logged to wandb")
            
            # Print expected wandb structure
            print(f"\n📋 EXPECTED WANDB STRUCTURE:")
            for split in successful_splits:
                print(f"   cluster_visualization_{split}/")
                print(f"   ├── vision_cluster_0, vision_cluster_1, ...")
                print(f"   ├── text_cluster_0, text_cluster_1, ...")
                print(f"   └── split_summary_*")
            
        visualization_results = all_visualization_results

# ========================================
# 🧮 PID ANALYSIS 
# ========================================

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