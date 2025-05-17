import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Tuple, Any, Union
from collections import defaultdict

from shimmer.modules.domain import DomainModule
from shimmer.modules.gw_module import GWModule, GWEncoder, GWDecoder, LatentsDomainGroupT
from shimmer.modules.selection import RandomSelection

# Try importing wandb for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Experiment tracking disabled.")

# Try importing pretrained domain loading functionality
try:
    from shimmer_ssd.config import LoadedDomainConfig, DomainModuleVariant
    from shimmer_ssd.modules.domains.pretrained import load_pretrained_module
    HAS_SHIMMER_SSD = True
except ImportError:
    HAS_SHIMMER_SSD = False
    print("Warning: shimmer_ssd not found. Loading pretrained domains will not work.")

class GWModuleConfigurableFusion(GWModule):
    """GW Module with configurable fusion weights for each domain."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
        fusion_weights: Dict[str, float],
        fusion_activation_fn = torch.tanh,
    ) -> None:
        super().__init__(
            domain_modules, 
            workspace_dim, 
            gw_encoders, 
            gw_decoders, 
            fusion_activation_fn
        )
        self.fusion_weights = fusion_weights
        # Store architecture parameters for easier checkpoint saving
        self.hidden_dim = None
        self.n_layers = None
        
    def set_fusion_weights(self, fusion_weights: Dict[str, float]) -> None:
        """Set new fusion weights for the domains."""
        self.fusion_weights = fusion_weights

    def fuse(
        self,
        x: LatentsDomainGroupT,
        selection_scores: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Merge domain representations using configured fusion weights."""
        weighted_sum = torch.zeros_like(list(x.values())[0])
        
        for domain, representation in x.items():
            if domain in self.fusion_weights:
                weight = self.fusion_weights[domain]
                weighted_sum += weight * representation
                
        return self.fusion_activation_fn(weighted_sum)
    
def create_gw_model(
    domain_modules: Mapping[str, DomainModule],
    workspace_dim: int,
    hidden_dim: int = 32,
    n_layers: int = 3,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> GWModuleConfigurableFusion:
    """Create a GW model with configurable fusion weights."""
    # Create encoders and decoders for each domain
    gw_encoders = {}
    gw_decoders = {}
    
    for domain_name, domain_module in domain_modules.items():
        # Get latent dimension from domain module
        latent_dim = domain_module.latent_dim
        
        # For text domain with BERT embeddings (768 dimensions)
        if domain_name == 't' and latent_dim == 768:
            # Check if the domain module already has a projector
            if hasattr(domain_module, 'projector'):
                print(f"Found existing projector for text domain. Using pretrained projector.")
                # The projector will be frozen later
            else:
                # In case there's no projector (which shouldn't happen for domain_t.ckpt),
                # we won't create one as per requirements
                print(f"Warning: No projector found for text domain.")
        
        # Create encoder and decoder
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
    
    # Freeze domain modules (including projectors)
    for domain_name, domain_module in domain_modules.items():
        # Use recursive function to freeze all parameters in the domain module
        def freeze_params(module):
            for param in module.parameters():
                param.requires_grad = False
                
        # Apply freezing recursively to the entire domain module
        domain_module.apply(freeze_params)
        
        # Double-check that all parameters are frozen
        for name, param in domain_module.named_parameters():
            assert not param.requires_grad, f"Parameter {name} in {domain_name} was not properly frozen"
            if 'projector' in name:
                print(f"Freezing pretrained projector parameter: {name}")
    
    return gw_module

def load_domain_modules(domain_configs: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]) -> Dict[str, DomainModule]:
    """Load domain modules from configuration.
    
    Args:
        domain_configs: Can be either a list of domain config dicts, or a dict mapping 
                       domain names to their configs.
    """
    if not HAS_SHIMMER_SSD:
        raise ImportError("shimmer_ssd required to load pretrained domain modules")
    
    domain_modules = {}
    
    # Convert dict format to list format if needed
    configs_list = []
    if isinstance(domain_configs, dict):
        for domain_name, config in domain_configs.items():
            if "name" not in config:
                config = config.copy()  # Create a copy to avoid modifying the original
                config["name"] = domain_name
            configs_list.append(config)
    else:
        configs_list = domain_configs
    
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


def save_checkpoint(
    model: GWModuleConfigurableFusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Save model checkpoint."""
    # Create checkpoint directory if needed
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Create checkpoint content
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "fusion_weights": model.fusion_weights,
        "metadata": metadata or {},
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save metadata separately as JSON for easier inspection
    metadata_path = os.path.join(checkpoint_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata or {}, f, indent=2, default=str)
        
    return checkpoint_path

def load_checkpoint(
    checkpoint_path: str,
    domain_modules: Optional[Mapping[str, DomainModule]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> GWModuleConfigurableFusion:
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get fusion weights
    fusion_weights = checkpoint.get("fusion_weights")
    
    # Check if domain modules were provided
    if domain_modules is None:
        raise ValueError("Domain modules must be provided to load checkpoint")
    
    # Extract workspace dimension from metadata if available
    metadata = checkpoint.get("metadata", {})
    workspace_dim = metadata.get("workspace_dim")
    
    # If workspace_dim not found in metadata, try to extract it from model state
    if workspace_dim is None:
        for domain_name, domain_encoders in checkpoint["model_state_dict"].items():
            if "gw_encoders" in domain_name and ".weight" in domain_name:
                # Extract workspace dimension from the last layer of any encoder
                layer_parts = domain_name.split(".")
                if layer_parts[-2] == "2" and layer_parts[-1] == "weight":
                    workspace_dim = checkpoint["model_state_dict"][domain_name].size(0)
                    break
    
    if workspace_dim is None:
        raise ValueError("Could not determine workspace dimension from checkpoint")
    
    # Extract optional architecture params from metadata
    hidden_dim = metadata.get("hidden_dim", 32)
    n_layers = metadata.get("n_layers", 3)
    
    # Create model
    model = create_gw_model(
        domain_modules=domain_modules,
        workspace_dim=workspace_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        fusion_weights=fusion_weights,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    print(f"Loaded model from {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Model was trained for {checkpoint['epoch'] + 1} epochs")
    
    return model


def format_fusion_weights(fusion_weights: Dict[str, float]) -> str:
    """Format fusion weights for filenames."""
    return "_".join([f"{k}_{v:.1f}" for k, v in fusion_weights.items()])


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

def train_model(
    model: GWModuleConfigurableFusion,
    train_data_loader,
    val_data_loader=None,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 1,
    run_name: Optional[str] = None,
    log_to_wandb: bool = False,
    wandb_project: str = "gw-fusion",
    wandb_entity: Optional[str] = None,
    short_circuit: bool = False,
    use_weighted_loss: bool = False,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Tuple[GWModuleConfigurableFusion, str]:
    """Train the GW model with validation.
    
    Args:
        model: The GW model to train
        train_data_loader: DataLoader for training data
        val_data_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        checkpoint_dir: Directory for checkpoints
        checkpoint_interval: Save checkpoint every N epochs
        run_name: Name for the training run
        log_to_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
        short_circuit: Whether to short-circuit for quick testing
        use_weighted_loss: Whether to weight the loss by fusion weights
        loss_weights: Dictionary with keys 'fusion', 'demi_cycle', 'cycle' and weights
    """
    # Move model to device
    model = model.to(device)
    
    # Create selection module (not used but required by API)
    selection_module = RandomSelection(temperature=1.0)
    
    # Create optimizer for trainable parameters
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Set up loss weights with defaults
    if loss_weights is None:
        # Default: use only fusion loss
        loss_weights = {
            'fusion': 1.0,
            'demi_cycle': 0.0,
            'cycle': 0.0
        }
    
    # Store whether to use weights for fusion loss
    model.fusion_weights['use_weights_for_loss'] = use_weighted_loss
    
    # Get model configuration for wandb
    model_config = {
        "fusion_weights": model.fusion_weights,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "device": device,
        "model_type": "GWModuleConfigurableFusion",
        "hidden_dim": model.hidden_dim,
        "n_layers": model.n_layers,
        "workspace_dim": model.workspace_dim,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "batch_size": getattr(train_data_loader, "batch_size", None),
        "short_circuit": short_circuit,
        "use_weighted_loss": use_weighted_loss,
        "loss_weights": loss_weights,
    }
    
    # Setup wandb logging
    if log_to_wandb and HAS_WANDB:
        weights_str = format_fusion_weights(model.fusion_weights)
        run_name = run_name or f"fusion_{weights_str}"
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=model_config,
            tags=["configurable-fusion", f"ws{model.workspace_dim}"]
        )
        
        # Log model architecture as a summary
        if hasattr(wandb, "Table"):
            model_summary = []
            for name, param in model.named_parameters():
                model_summary.append([name, param.shape, param.numel(), param.requires_grad])
            
            wandb.log({
                "model_architecture": wandb.Table(
                    data=model_summary,
                    columns=["Layer", "Shape", "Parameters", "Trainable"]
                )
            })
    
    # Training loop
    best_loss = float('inf')
    best_val_loss = float('inf')
    best_checkpoint_path = None
    training_start_time = datetime.now()
    
    try:
        from tqdm import tqdm
        epoch_iterator = tqdm(range(num_epochs), desc="Training")
    except ImportError:
        epoch_iterator = range(num_epochs)
    
    for epoch in epoch_iterator:
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        domain_train_losses = defaultdict(float)
        epoch_start_time = datetime.now()
        
        # Handle CombinedLoader from PyTorch Lightning
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        try:
            # For CombinedLoader compatibility, initialize the iterator explicitly
            batch_iterator = iter(train_data_loader)
            # Try to get an estimate of the dataset size if possible
            try:
                estimated_size = len(train_data_loader.datasets[list(train_data_loader.datasets.keys())[0]])
                pbar = tqdm(total=estimated_size // train_data_loader.batch_size, desc="Batches", leave=False)
            except (AttributeError, TypeError):
                # If we can't get the size, don't use a progress bar
                pbar = None
                print("  Can't determine dataset size, progress bar disabled")
                
            # Process batches
            while True:
                try:
                    batch = next(batch_iterator)
                    
                    # Process batch
                    processed_batch = process_batch(batch, device)
                    
                    if not processed_batch:
                        continue
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Calculate losses using the new loss calculation function
                    batch_loss, loss_details = calculate_losses_with_weights(
                        model=model,
                        batch=processed_batch,
                        criterion=criterion,
                        loss_weights=loss_weights,
                        device=device
                    )
                    
                    # Skip if loss is None or zero (no domains were processed)
                    if batch_loss is None or batch_loss == 0:
                        continue
                    
                    # Backpropagation
                    batch_loss.backward()
                    
                    # Apply gradients
                    optimizer.step()
                    
                    # Update statistics
                    train_loss += batch_loss.item()
                    num_batches += 1
                    
                    # Update domain-specific losses
                    for key, value in loss_details.items():
                        domain_train_losses[f"train_{key}"] += value
                    
                    # Update progress bar if available
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(loss=f"{batch_loss.item():.6f}")
                        
                    # Short-circuit training for quick testing if enabled
                    if short_circuit and num_batches >= 10:
                        print(f"  Short-circuiting training loop after {num_batches} batches (short-circuit mode)")
                        break
                        
                except StopIteration:
                    break
                    
            # Close progress bar if it exists
            if pbar is not None:
                pbar.close()
                
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        
        # Compute average train loss
        avg_train_loss = train_loss / max(num_batches, 1)
        
        # Compute average per-domain train losses
        for key in domain_train_losses:
            domain_train_losses[key] /= max(num_batches, 1)
        
        # Run validation if validation data loader is provided
        val_loss = None
        val_domain_losses = {}
        
        if val_data_loader is not None:
            try:
                # Evaluate on validation set
                val_loss, val_domain_losses = evaluate_model(
                    model=model,
                    data_loader=val_data_loader,
                    device=device,
                    short_circuit=short_circuit,
                    use_weighted_loss=use_weighted_loss,
                    loss_weights=loss_weights
                )
            except Exception as e:
                print(f"Error during validation: {e}")
                import traceback
                traceback.print_exc()
                
            # Track best validation loss
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Calculate epoch time
        epoch_time = datetime.now() - epoch_start_time
        
        # Log metrics
        status = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}"
        if val_loss is not None:
            status += f", Val Loss: {val_loss:.6f}"
        status += f", Time: {epoch_time}"
        print(status)
        
        # Log to wandb
        if log_to_wandb and HAS_WANDB:
            # Basic metrics
            log_dict = {
                "epoch": epoch + 1,
                "epoch_time": epoch_time.total_seconds(),
                "train_loss": avg_train_loss,
                "learning_rate": learning_rate,
                **domain_train_losses
            }
            
            # Add validation metrics if available
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
                log_dict.update(val_domain_losses)
                
                # Add validation to train loss ratio (useful for monitoring overfitting)
                log_dict["val_train_ratio"] = val_loss / avg_train_loss if avg_train_loss > 0 else 0
                
            wandb.log(log_dict)
        
        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            weights_str = format_fusion_weights(model.fusion_weights)
            filename = f"model_epoch_{epoch+1:03d}_{weights_str}.pt"
            
            # Add metadata
            metadata = {
                "workspace_dim": model.workspace_dim,
                "hidden_dim": model.hidden_dim,
                "n_layers": model.n_layers,
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "fusion_weights": model.fusion_weights,
                "domain_names": list(model.fusion_weights.keys()),
                "run_name": run_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "model_type": "GWModuleConfigurableFusion",
                "epoch_time": epoch_time.total_seconds(),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in model.parameters()),
                "use_weighted_loss": use_weighted_loss,
                "loss_weights": loss_weights,
            }
            
            if val_loss is not None:
                metadata["val_loss"] = val_loss
                for key, value in val_domain_losses.items():
                    metadata[key] = value
            
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_train_loss,
                checkpoint_dir=checkpoint_dir,
                filename=filename,
                metadata=metadata
            )
            
            # Update best checkpoint based on validation loss if available, otherwise training loss
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = checkpoint_path
            else:
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    best_checkpoint_path = checkpoint_path
    
    # Calculate total training time
    total_training_time = datetime.now() - training_start_time
    
    # Save final model
    if checkpoint_dir is not None:
        # Create a timestamp for unique identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a descriptive filename
        domain_names = "_".join(sorted(model.fusion_weights.keys()))
        weights_desc = format_fusion_weights(model.fusion_weights)
        final_filename = f"gw_model_{domain_names}_{weights_desc}_ws{model.workspace_dim}_{timestamp}.pt"
        
        # Create comprehensive metadata
        metadata = {
            "workspace_dim": model.workspace_dim,
            "hidden_dim": model.hidden_dim,
            "n_layers": model.n_layers,
            "domain_names": list(model.fusion_weights.keys()),
            "fusion_weights": model.fusion_weights,
            "timestamp": timestamp,
            "is_final": True,
            "model_type": "GWModuleConfigurableFusion",
            "train_loss": avg_train_loss,
            "run_name": run_name,
            "description": "Global Workspace model with configurable fusion weights",
            "total_training_time": total_training_time.total_seconds(),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
            "short_circuit": short_circuit,
            "use_weighted_loss": use_weighted_loss,
            "loss_weights": loss_weights,
        }
        
        if val_loss is not None:
            metadata["val_loss"] = val_loss
            metadata["best_val_loss"] = best_val_loss
            for key, value in val_domain_losses.items():
                metadata[key] = value
        
        # Save final checkpoint
        final_checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=num_epochs,
            loss=avg_train_loss,
            checkpoint_dir=checkpoint_dir,
            filename=final_filename,
            metadata=metadata
        )
        
        print(f"Final model saved to {final_checkpoint_path}")
        print(f"Total training time: {total_training_time}")
        
        # If no best checkpoint was found, use the final one
        if best_checkpoint_path is None:
            best_checkpoint_path = final_checkpoint_path
    
    # Log final summary to wandb
    if log_to_wandb and HAS_WANDB:
        wandb.run.summary.update({
            "final_train_loss": avg_train_loss,
            "best_val_loss": best_val_loss if val_loss is not None else None,
            "final_val_loss": val_loss,
            "total_training_time": total_training_time.total_seconds(),
            "epochs_completed": num_epochs,
            "final_checkpoint": best_checkpoint_path,
            "use_weighted_loss": use_weighted_loss,
            "loss_weights": loss_weights,
        })
        
        wandb.finish()
    
    return model, best_checkpoint_path or ""

def find_checkpoint_path(path):
    """Find a checkpoint file by trying different paths."""
    if os.path.exists(path):
        return path
    
    # Try alternative paths
    alt_paths = [
        path.replace("../", ""),
        os.path.expanduser(f"~/shimmer-ssd/{path.replace('../', '')}"),
        os.path.join("..", path.replace("../", "")),
        os.path.join("shimmer_ssd", path.replace("../", "")),
        os.path.join("checkpoints", os.path.basename(path)),
    ]
    
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            return alt_path
    
    return None


def generate_weight_configs(domain_names: List[str], n_configs: int = 5) -> List[Dict[str, float]]:
    """Generate a series of weight configurations."""
    if len(domain_names) != 2:
        raise ValueError(f"This function only supports 2 domains, got {len(domain_names)}")
    
    # Validate input
    if n_configs < 2:
        raise ValueError(f"n_configs must be at least 2, got {n_configs}")
    
    domain_a, domain_b = domain_names
    configs = []
    
    for i in range(n_configs):
        # Calculate weights (ranging from 1.0 to 0.0)
        weight_a = 1.0 - (i / (n_configs - 1)) if n_configs > 1 else 0.5
        weight_b = 1.0 - weight_a
        
        # Ensure weights sum to 1.0 (handling floating point precision issues)
        if abs((weight_a + weight_b) - 1.0) > 1e-9:
            weight_b = 1.0 - weight_a
        
        configs.append({domain_a: weight_a, domain_b: weight_b})
        
    return configs


def evaluate_model_quick(
    model: GWModuleConfigurableFusion,
    data_loader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_batches: int = 10,
    use_weighted_loss: bool = False,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Quick evaluation of the model on a limited number of batches."""
    # Set model to evaluation mode
    model.eval()
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Set up loss weights with defaults
    if loss_weights is None:
        # Default: use only fusion loss
        loss_weights = {
            'fusion': 1.0,
            'demi_cycle': 0.0,
            'cycle': 0.0
        }
    
    # Store whether to use weights for fusion loss
    model.fusion_weights['use_weights_for_loss'] = use_weighted_loss
    
    # Track statistics
    val_loss = 0.0
    num_batches = 0
    domain_losses = defaultdict(float)
    
    print(f"Quick evaluation on up to {max_batches} batches")
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        try:
            # For CombinedLoader compatibility, initialize the iterator explicitly
            batch_iterator = iter(data_loader)
            
            # Process limited number of batches
            while num_batches < max_batches:
                try:
                    batch = next(batch_iterator)
                    
                    # Process batch
                    processed_batch = process_batch(batch, device)
                    
                    if not processed_batch:
                        continue
                    
                    # Calculate losses using the new loss calculation function
                    batch_loss, loss_details = calculate_losses_with_weights(
                        model=model,
                        batch=processed_batch,
                        criterion=criterion,
                        loss_weights=loss_weights,
                        device=device
                    )
                    
                    # Skip if loss is None or zero (no domains were processed)
                    if batch_loss is None or batch_loss == 0:
                        continue
                        
                    # Update statistics
                    val_loss += batch_loss.item()
                    num_batches += 1
                    
                    # Update domain-specific losses
                    for key, value in loss_details.items():
                        domain_losses[f"val_{key}"] += value
                    
                    print(f"  Validation batch {num_batches}: loss={batch_loss.item():.6f}")
                    
                except StopIteration:
                    break
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    # Set model back to training mode
    model.train()
    
    # Compute averages
    avg_val_loss = val_loss / max(num_batches, 1)
    
    # Compute domain-specific average losses
    for key in domain_losses:
        domain_losses[key] /= max(num_batches, 1)
    
    return avg_val_loss, domain_losses

def evaluate_model(
    model: GWModuleConfigurableFusion,
    data_loader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    short_circuit: bool = False,
    use_weighted_loss: bool = False,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate the model on a dataset."""
    # Set model to evaluation mode
    model.eval()
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Set up loss weights with defaults
    if loss_weights is None:
        # Default: use only fusion loss
        loss_weights = {
            'fusion': 1.0,
            'demi_cycle': 0.0,
            'cycle': 0.0
        }
    
    # Store whether to use weights for fusion loss
    model.fusion_weights['use_weights_for_loss'] = use_weighted_loss
    
    # Track statistics
    val_loss = 0.0
    num_batches = 0
    domain_losses = defaultdict(float)
    
    print("Running full evaluation")
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        try:
            # For CombinedLoader compatibility, initialize the iterator explicitly
            batch_iterator = iter(data_loader)
            
            # Try to get an estimate of the dataset size if possible
            try:
                estimated_size = len(data_loader.datasets[list(data_loader.datasets.keys())[0]])
                pbar = tqdm(total=estimated_size // data_loader.batch_size, desc="Evaluating", leave=False)
            except (AttributeError, TypeError):
                # If we can't get the size, don't use a progress bar
                pbar = None
                print("  Can't determine dataset size, progress bar disabled")
                
            # Process batches
            while True:
                try:
                    batch = next(batch_iterator)
                    
                    # Process batch
                    processed_batch = process_batch(batch, device)
                    if not processed_batch:
                        continue
                    
                    # Calculate losses using the new loss calculation function
                    batch_loss, loss_details = calculate_losses_with_weights(
                        model=model,
                        batch=processed_batch,
                        criterion=criterion,
                        loss_weights=loss_weights,
                        device=device
                    )
                    
                    # Skip if loss is None or zero (no domains were processed)
                    if batch_loss is None or batch_loss == 0:
                        continue
                        
                    # Update statistics
                    val_loss += batch_loss.item()
                    num_batches += 1
                    
                    # Update domain-specific losses
                    for key, value in loss_details.items():
                        domain_losses[f"val_{key}"] += value
                    
                    # Short-circuit for quick testing
                    if short_circuit and num_batches >= 10:
                        print(f"  Short-circuiting evaluation after {num_batches} batches (short-circuit mode)")
                        break
                        
                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(loss=f"{batch_loss.item():.6f}")
                    
                except StopIteration:
                    break
            
            # Close progress bar if it exists
            if pbar is not None:
                pbar.close()
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    # Set model back to training mode
    model.train()
    
    # Compute averages
    avg_val_loss = val_loss / max(num_batches, 1)
    
    # Compute domain-specific average losses
    for key in domain_losses:
        domain_losses[key] /= max(num_batches, 1)
    
    return avg_val_loss, domain_losses

def train_multiple_configs(
    domain_modules: Dict[str, DomainModule],
    train_data_loader,
    val_data_loader=None,
    workspace_dim: int = 12,
    hidden_dim: int = 32,
    n_layers: int = 3,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    n_configs: int = 5,
    base_checkpoint_dir: str = "checkpoints/fusion",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_to_wandb: bool = False,
    wandb_project: str = "gw-fusion",
    wandb_entity: Optional[str] = None,
    short_circuit: bool = False,
    use_weighted_loss: bool = False,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[str, float]]:
    """Train multiple GW models with different fusion weights and validate them.
    
    Args:
        domain_modules: Dictionary of domain modules
        train_data_loader: DataLoader for training data
        val_data_loader: Optional DataLoader for validation data
        workspace_dim: Dimension of the workspace
        hidden_dim: Hidden dimension of GW encoders/decoders
        n_layers: Number of layers in GW encoders/decoders
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        n_configs: Number of weight configurations to generate
        base_checkpoint_dir: Base directory for checkpoints
        device: Device to train on
        log_to_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_entity: Wandb entity name
        short_circuit: Whether to short-circuit for quick testing
        use_weighted_loss: Whether to weight the loss by fusion weights
        loss_weights: Dictionary with keys 'fusion', 'demi_cycle', 'cycle' and weights
        
    Returns:
        Dictionary mapping configuration names to (checkpoint_path, validation_loss) tuples
    """
    # Create base checkpoint directory
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    # Set default loss weights if not provided
    if loss_weights is None:
        loss_weights = {
            'fusion': 1.0,
            'demi_cycle': 0.0,
            'cycle': 0.0
        }
    
    # Generate weight configurations
    domain_names = list(domain_modules.keys())
    weight_configs = generate_weight_configs(domain_names, n_configs)
    
    # Track results for each configuration
    results = {}
    
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(weight_configs), total=len(weight_configs), desc="Training Configurations")
    except ImportError:
        iterator = enumerate(weight_configs)
    
    for i, fusion_weights in iterator:
        weights_str = format_fusion_weights(fusion_weights)
        config_name = f"config_{i}_{weights_str}"
        
        print(f"\n{'-'*40}")
        print(f"Training configuration {i+1}/{len(weight_configs)}: {weights_str}")
        print(f"{'-'*40}")
        
        # Create model with these weights
        model = create_gw_model(
            domain_modules=domain_modules,
            workspace_dim=workspace_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            fusion_weights=fusion_weights,
        )
        
        # Create config checkpoint directory
        config_checkpoint_dir = os.path.join(base_checkpoint_dir, config_name)
        os.makedirs(config_checkpoint_dir, exist_ok=True)
        
        # Train model
        _, checkpoint_path = train_model(
            model=model,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir=config_checkpoint_dir,
            run_name=config_name,
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            short_circuit=short_circuit,
            use_weighted_loss=use_weighted_loss,
            loss_weights=loss_weights,
        )
        
        # Evaluate model on validation set if available
        val_loss = None
        if val_data_loader is not None:
            print(f"Evaluating configuration {i+1} on validation set")
            val_loss, _ = evaluate_model_quick(
                model=model,
                data_loader=val_data_loader,
                device=device,
                max_batches=5 if short_circuit else 50,
                use_weighted_loss=use_weighted_loss,
                loss_weights=loss_weights
            )
            print(f"Validation loss: {val_loss:.6f}")
        
        # Store results
        results[config_name] = (checkpoint_path, val_loss)
    
    # Print summary of results
    print("\n" + "="*60)
    print(" TRAINING RESULTS SUMMARY ".center(60, "="))
    print("="*60)
    
    # Sort by validation loss if available
    if all(r[1] is not None for r in results.values()):
        sorted_results = sorted(results.items(), key=lambda x: x[1][1])
        print("\nConfigurations sorted by validation loss:")
        for i, (config_name, (checkpoint_path, val_loss)) in enumerate(sorted_results, 1):
            print(f"{i}. {config_name}: val_loss={val_loss:.6f}, checkpoint={checkpoint_path}")
            
        # Log the best configuration to wandb if enabled
        if log_to_wandb and HAS_WANDB:
            try:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name="multi_config_summary",
                    config={
                        "workspace_dim": workspace_dim,
                        "hidden_dim": hidden_dim,
                        "n_layers": n_layers,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "n_configs": n_configs,
                        "short_circuit": short_circuit,
                        "use_weighted_loss": use_weighted_loss,
                        "loss_weights": loss_weights,
                    }
                )
                
                # Create a bar chart of validation losses
                data = [[config_name, val_loss] for config_name, (_, val_loss) in sorted_results]
                table = wandb.Table(data=data, columns=["Configuration", "Validation Loss"])
                wandb.log({
                    "validation_losses": wandb.plot.bar(
                        table, "Configuration", "Validation Loss",
                        title="Validation Losses by Configuration"
                    ),
                    "best_config": sorted_results[0][0],
                    "best_val_loss": sorted_results[0][1][1],
                })
                wandb.finish()
            except Exception as e:
                print(f"Error logging to wandb: {e}")
    else:
        print("\nConfigurations:")
        for i, (config_name, (checkpoint_path, _)) in enumerate(results.items(), 1):
            print(f"{i}. {config_name}: checkpoint={checkpoint_path}")
    
    return results

def text_decode_batch(text_module, latent_vectors, max_length=64, device="cuda"):
    """Decode latent vectors to text tokens using the full text domain module.
    
    Args:
        text_module: The GRUTextDomainModule instance
        latent_vectors: Batch of latent vectors [batch_size, latent_dim]
        max_length: Maximum sequence length to generate
        device: Device to run decoding on
    
    Returns:
        Tuple of (token_ids, logits)
    """
    batch_size = latent_vectors.shape[0]
    
    # Initialize with zeros (will be filled with generated tokens)
    if hasattr(text_module, '_padding_token'):
        start_token = text_module._padding_token
    else:
        start_token = 0  # Default padding token
    
    # Initialize first token for each sequence in batch
    token_ids = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    
    # Track logits for all positions
    all_logits = []
    
    # Create initial hidden state from latent vector
    # For GRU, we need [num_layers, batch_size, hidden_size]
    num_layers = text_module.decoder.num_layers
    hidden_size = text_module.decoder.hidden_size
    
    # Project latent vector to initial hidden state for GRU
    # Create a simple transformation to convert latent_dim to hidden_size
    latent_dim = latent_vectors.shape[1]
    
    # Create a linear projection from latent_dim to hidden_size if dimensions don't match
    if latent_dim != hidden_size:
        # Simple linear projection (could be replaced with a proper learned projection)
        hidden_projection = torch.zeros(hidden_size, latent_dim, device=device)
        for i in range(min(latent_dim, hidden_size)):
            hidden_projection[i, i] = 1.0  # Identity for overlapping dimensions
            
        # Project latent to hidden size
        projected_latents = torch.matmul(hidden_projection, latent_vectors.t()).t()
    else:
        projected_latents = latent_vectors
    
    # Repeat for each layer of the GRU
    h0 = projected_latents.unsqueeze(0).repeat(num_layers, 1, 1)
    
    # Autoregressive decoding loop
    hidden = h0
    
    for i in range(max_length - 1):
        # Get the most recent token
        current_token = token_ids[:, -1:]
        
        # Convert token to embedding
        emb = text_module.embeddings(current_token)
        
        # Pass through GRU decoder
        # Input shape: [batch_size, 1, emb_dim]
        # Hidden shape: [num_layers, batch_size, hidden_size]
        output, hidden = text_module.decoder(emb, hidden)
        
        # Get logits for next token prediction
        # Output shape: [batch_size, 1, hidden_size]
        # We need to squeeze the middle dimension to get [batch_size, hidden_size]
        logits = text_module.text_head(output.squeeze(1))
        
        # Store logits
        all_logits.append(logits)
        
        # Get next token (argmax or sampling)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append to generated sequence
        token_ids = torch.cat([token_ids, next_token], dim=1)
    
    # Stack all logits [max_length-1, batch_size, vocab_size]
    stacked_logits = torch.stack(all_logits)
    
    # Return generated token ids and logits
    return token_ids, stacked_logits

def decode_text_from_latent(text_module, latent_vector, max_length=64, device="cuda"):
    """Decode a single latent vector to text using the text domain module.
    
    Args:
        text_module: The GRUTextDomainModule instance
        latent_vector: Latent vector [latent_dim]
        max_length: Maximum sequence length to generate
        device: Device to run decoding on
    
    Returns:
        List of token IDs
    """
    # Add batch dimension if needed
    if len(latent_vector.shape) == 1:
        latent_vector = latent_vector.unsqueeze(0)
    
    # Move to the correct device
    latent_vector = latent_vector.to(device)
    
    # Decode
    with torch.no_grad():
        token_ids, _ = text_decode_batch(text_module, latent_vector, max_length, device)
    
    # Return just the token IDs (removing batch dimension)
    return token_ids[0].cpu().numpy().tolist()

def generate_text_from_workspace(model, workspace_state, vocab=None, max_length=64, device="cuda"):
    """Generate text from a workspace state using the text domain decoder.
    
    Args:
        model: The GWModuleConfigurableFusion model
        workspace_state: Workspace state tensor
        vocab: Optional vocabulary for token->text conversion
        max_length: Maximum length of text to generate
        device: Device to run generation on
    
    Returns:
        Tuple of (token_ids, decoded_text)
    """
    if not hasattr(model, 'domain_mods') or 't' not in model.domain_mods:
        raise ValueError("Model does not have a text domain module")
    
    # Get text domain module
    text_module = model.domain_mods['t']
    
    # First decode workspace to text domain latent
    if len(workspace_state.shape) == 1:
        workspace_state = workspace_state.unsqueeze(0)
    
    workspace_state = workspace_state.to(device)
    
    # Use model's text domain decoder to get latent
    with torch.no_grad():
        text_latent = model.gw_decoders['t'](workspace_state)
    
    # Decode the latent to token IDs
    token_ids = decode_text_from_latent(text_module, text_latent, max_length, device)
    
    # Convert to text if vocab is provided
    decoded_text = None
    if vocab is not None:
        # Remove any padding tokens
        if hasattr(text_module, '_padding_token'):
            padding_token = text_module._padding_token
            token_ids = [t for t in token_ids if t != padding_token]
        
        # Convert tokens to text
        try:
            if hasattr(vocab, 'decode'):
                # If it's a tokenizer with a decode method
                decoded_text = vocab.decode(token_ids)
            elif hasattr(vocab, 'lookup_tokens'):
                # If it's a vocabulary with a lookup_tokens method (e.g., torchtext)
                decoded_text = ' '.join(vocab.lookup_tokens(token_ids))
            elif isinstance(vocab, dict) or hasattr(vocab, '__getitem__'):
                # If it's a dictionary-like object
                decoded_text = ' '.join([str(vocab.get(t, '<unk>')) for t in token_ids])
            else:
                decoded_text = f"<Tokens: {token_ids}>"
        except Exception as e:
            decoded_text = f"<Error decoding: {e}>"
    
    return token_ids, decoded_text

def demo_text_decoding(model, device='cuda'):
    """Demonstrate the text decoding capabilities of the model.
    
    Args:
        model: Trained GWModuleConfigurableFusion model
        device: Device to run inference on
    """
    print("\n" + "="*60)
    print(" TEXT DECODING DEMONSTRATION ".center(60, "="))
    print("="*60 + "\n")
    
    if not hasattr(model, 'domain_mods') or 't' not in model.domain_mods:
        print("Error: Model does not have a text domain module")
        return
    
    text_module = model.domain_mods['t']
    
    if not all(hasattr(text_module, attr) for attr in ['projector', 'embeddings', 'decoder', 'text_head']):
        print("Error: Text module is missing required components for decoding")
        missing = [attr for attr in ['projector', 'embeddings', 'decoder', 'text_head'] 
                  if not hasattr(text_module, attr)]
        print(f"Missing components: {missing}")
        return
    
    print("Text module successfully loaded with all required components:")
    print(f"- Projector: {type(text_module.projector).__name__}")
    print(f"- Embeddings: {text_module.embeddings}")
    print(f"- Decoder: {text_module.decoder}")
    print(f"- Text head: {text_module.text_head}")
    
    # Create a random workspace state
    print("\nGenerating sample text from random workspace states...\n")
    workspace_dim = model.workspace_dim
    
    for i in range(5):
        # Create a random workspace state
        random_state = torch.randn(1, workspace_dim, device=device)
        
        # Generate text
        token_ids, _ = generate_text_from_workspace(model, random_state, device=device)
        
        # Print results
        print(f"Sample {i+1}:")
        print(f"Tokens: {token_ids[:20]}...")
        print("-" * 40)
    
    # Try encoding and decoding a batch
    print("\nDemonstrating encoding and decoding with the text module\n")
    
    # Create some dummy BERT embeddings (768-dimensional)
    dummy_bert_embeddings = torch.randn(3, 768, device=device)
    
    # Encode through projector to get latent
    text_latents = text_module.projector(dummy_bert_embeddings)
    print(f"BERT embeddings shape: {dummy_bert_embeddings.shape}")
    print(f"Text latents shape after projection: {text_latents.shape}")
    
    # Decode latents to tokens
    for i in range(len(text_latents)):
        latent = text_latents[i:i+1]  # Keep batch dimension
        token_ids, _ = text_decode_batch(text_module, latent, max_length=32, device=device)
        
        print(f"\nSample {i+1} token sequence:")
        print(token_ids[0][:20].tolist())
    
    print("\nText decoding demonstration completed")

def calculate_losses_with_weights(
    model,
    batch,
    criterion,
    loss_weights,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Calculate different types of losses with configurable weights.
    
    Args:
        model: The GWModuleConfigurableFusion model
        batch: Batch of data
        criterion: Loss function (e.g., nn.MSELoss())
        loss_weights: Dictionary with keys 'fusion', 'demi_cycle', 'cycle' and weights
        device: Device to run calculations on
        
    Returns:
        Tuple of (total_loss, loss_details)
    """
    # Process batch to get inputs for each domain
    processed_batch = {}
    for domain_name, domain_input in batch.items():
        # Apply projector for text domain if it exists
        if domain_name == 't' and hasattr(model.domain_mods[domain_name], 'projector'):
            projector = model.domain_mods[domain_name].projector
            processed_batch[domain_name] = projector(domain_input)
        else:
            processed_batch[domain_name] = domain_input
    
    # Calculate different losses based on provided weights
    total_loss = None
    loss_details = {}
    
    # 1. Fusion Loss (encode all domains, fuse in GLW, decode all domains)
    if loss_weights.get('fusion', 0.0) > 0:
        fusion_loss, fusion_details = calculate_fusion_loss(
            model, processed_batch, criterion, model.fusion_weights.get('use_weights_for_loss', False)
        )
        loss_details.update(fusion_details)
        
        weighted_fusion_loss = loss_weights['fusion'] * fusion_loss
        loss_details['weighted_fusion_loss'] = weighted_fusion_loss.item()
        
        # Add to total loss
        if total_loss is None:
            total_loss = weighted_fusion_loss
        else:
            total_loss = total_loss + weighted_fusion_loss
    
    # 2. Demi-Cycle Loss (encode one domain, decode back to same domain)
    if loss_weights.get('demi_cycle', 0.0) > 0:
        demi_cycle_loss, demi_details = calculate_demi_cycle_loss(model, processed_batch, criterion)
        loss_details.update(demi_details)
        
        weighted_demi_loss = loss_weights['demi_cycle'] * demi_cycle_loss
        loss_details['weighted_demi_cycle_loss'] = weighted_demi_loss.item()
        
        # Add to total loss
        if total_loss is None:
            total_loss = weighted_demi_loss
        else:
            total_loss = total_loss + weighted_demi_loss
    
    # 3. Cycle Loss (encode domain A, decode to domain B, encode domain B, decode back to A)
    if loss_weights.get('cycle', 0.0) > 0:
        cycle_loss, cycle_details = calculate_cycle_loss(model, processed_batch, criterion)
        loss_details.update(cycle_details)
        
        weighted_cycle_loss = loss_weights['cycle'] * cycle_loss
        loss_details['weighted_cycle_loss'] = weighted_cycle_loss.item()
        
        # Add to total loss
        if total_loss is None:
            total_loss = weighted_cycle_loss
        else:
            total_loss = total_loss + weighted_cycle_loss
    
    return total_loss, loss_details

def calculate_fusion_loss(model, processed_batch, criterion, use_weights_for_loss=False):
    """Calculate fusion loss (encode all domains, fuse, decode all domains).
    
    This is the original loss used in the model.
    """
    # Dictionary to store loss details
    loss_details = {}
    
    # Forward pass for fusion loss
    encoded = {}
    for domain_name, domain_input in processed_batch.items():
        if domain_name in model.gw_encoders:
            encoded[domain_name] = model.gw_encoders[domain_name](domain_input)
    
    # Skip if no domains were encoded
    if not encoded:
        return torch.tensor(0.0, device=next(model.parameters()).device), loss_details
        
    # Fusion step
    gw_state = model.fuse(encoded, selection_scores={})
    
    # Reconstruction
    decoded = {}
    for domain_name in processed_batch.keys():
        if domain_name in model.gw_decoders:
            decoded[domain_name] = model.gw_decoders[domain_name](gw_state)
    
    # Compute loss
    batch_loss = None  # Initialize as None to ensure first loss is a tensor
    
    if use_weights_for_loss:
        # Weighted loss calculation using fusion weights
        for domain_name, domain_input in processed_batch.items():
            if domain_name in model.fusion_weights and domain_name in decoded:
                # Calculate domain-specific loss
                domain_loss = criterion(decoded[domain_name], domain_input)
                loss_details[f"fusion_{domain_name}_loss"] = domain_loss.item()
                
                # Use fusion weights as loss weights
                weight = model.fusion_weights[domain_name]
                weighted_loss = weight * domain_loss
                
                # Add to the total loss
                if batch_loss is None:
                    batch_loss = weighted_loss
                else:
                    batch_loss = batch_loss + weighted_loss
    else:
        # Standard unweighted loss calculation
        num_domains = 0
        
        for domain_name, domain_input in processed_batch.items():
            if domain_name in decoded:
                # Calculate domain-specific loss
                domain_loss = criterion(decoded[domain_name], domain_input)
                loss_details[f"fusion_{domain_name}_loss"] = domain_loss.item()
                
                # For the tensor calculation (needed for backprop)
                if batch_loss is None:
                    batch_loss = domain_loss
                else:
                    batch_loss = batch_loss + domain_loss
                    
                num_domains += 1
        
        # Average the loss across domains
        if batch_loss is not None and num_domains > 1:
            batch_loss = batch_loss / num_domains
    
    # Return zero loss if no domains were processed
    if batch_loss is None:
        batch_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
    loss_details['fusion_loss'] = batch_loss.item()
    return batch_loss, loss_details

def calculate_demi_cycle_loss(model, processed_batch, criterion):
    """Calculate demi-cycle loss.
    
    Ldcy = ||dx(tanh(ex(ux))) - ux||^2
    
    Each domain is encoded to GLW and then decoded back to the same domain.
    For demi-cycle, we set the weight to 1.0 for the domain being processed
    and 0.0 for other domains, regardless of the model's fusion weights.
    """
    loss_details = {}
    total_loss = None
    num_domains = 0
    
    # Process each domain independently
    for domain_name, domain_input in processed_batch.items():
        if domain_name in model.gw_encoders and domain_name in model.gw_decoders:
            # Create domain-specific weights for this demi-cycle
            # Set weight=1.0 for current domain, 0.0 for others
            demi_cycle_weights = {d: 0.0 for d in model.fusion_weights.keys()}
            demi_cycle_weights[domain_name] = 1.0
            
            # Store original weights
            original_weights = model.fusion_weights.copy()
            
            try:
                # Temporarily set model's fusion weights for this domain only
                model.fusion_weights.update(demi_cycle_weights)
                
                # Encode the domain to GLW
                encoded = model.gw_encoders[domain_name](domain_input)
                
                # Apply activation function (tanh)
                gw_state = torch.tanh(encoded)
                
                # Decode back to the same domain
                decoded = model.gw_decoders[domain_name](gw_state)
                
                # Calculate MSE loss
                domain_loss = criterion(decoded, domain_input)
                loss_details[f"demi_cycle_{domain_name}_loss"] = domain_loss.item()
                
                # Add to total loss
                if total_loss is None:
                    total_loss = domain_loss
                else:
                    total_loss = total_loss + domain_loss
                    
                num_domains += 1
            
            except Exception as e:
                print(f"Error in demi-cycle calculation for domain {domain_name}: {e}")
            
            finally:
                # Restore original fusion weights
                for key, value in original_weights.items():
                    model.fusion_weights[key] = value
    
    # Average across domains
    if total_loss is not None and num_domains > 1:
        total_loss = total_loss / num_domains
    
    # Return zero loss if no domains were processed
    if total_loss is None:
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
    loss_details['demi_cycle_loss'] = total_loss.item()
    return total_loss, loss_details

def calculate_cycle_loss(model, processed_batch, criterion):
    """Calculate cycle loss.
    
    Lcy = ||dx(tanh(ey(dy(tanh(ex(ux)))))) - ux||^2
    
    Each domain is encoded to GLW, decoded to another domain, encoded back to GLW,
    and finally decoded back to the original domain.
    
    For cycle loss, we use domain-specific weights:
    - First set weight=1.0 for source domain, 0.0 for others
    - Then set weight=1.0 for target domain, 0.0 for others
    - Finally set weight=1.0 for source domain again, 0.0 for others
    """
    loss_details = {}
    total_loss = None
    num_cycles = 0
    
    # Get list of domains in the batch
    domains = [d for d in processed_batch.keys() 
               if d in model.gw_encoders and d in model.gw_decoders]
    
    # We need at least 2 domains for cycle loss
    if len(domains) < 2:
        return torch.tensor(0.0, device=next(model.parameters()).device), loss_details
    
    # For each domain pair, calculate cycle loss in both directions
    for i, domain_x in enumerate(domains):
        for j, domain_y in enumerate(domains):
            if i == j:  # Skip same domain (that's demi-cycle)
                continue
                
            # Get inputs
            input_x = processed_batch[domain_x]
            
            # Store original weights
            original_weights = model.fusion_weights.copy()
            
            try:
                # 1. Set weights for source domain X (first encoding)
                source_weights = {d: 0.0 for d in model.fusion_weights.keys()}
                source_weights[domain_x] = 1.0
                model.fusion_weights.update(source_weights)
                
                # 1. Encode domain x to GLW
                encoded_x = model.gw_encoders[domain_x](input_x)
                
                # 2. Apply activation (tanh)
                gw_state_1 = torch.tanh(encoded_x)
                
                # Set weights for target domain Y (decoding and encoding)
                target_weights = {d: 0.0 for d in model.fusion_weights.keys()}
                target_weights[domain_y] = 1.0
                model.fusion_weights.update(target_weights)
                
                # 3. Decode to domain y
                decoded_y = model.gw_decoders[domain_y](gw_state_1)
                
                # 4. Encode domain y back to GLW
                encoded_y = model.gw_encoders[domain_y](decoded_y)
                
                # 5. Apply activation (tanh)
                gw_state_2 = torch.tanh(encoded_y)
                
                # Set weights back to source domain X (final decoding)
                model.fusion_weights.update(source_weights)
                
                # 6. Decode back to domain x
                decoded_x = model.gw_decoders[domain_x](gw_state_2)
                
                # 7. Calculate loss
                cycle_loss = criterion(decoded_x, input_x)
                loss_details[f"cycle_{domain_x}_via_{domain_y}_loss"] = cycle_loss.item()
                
                # Add to total loss
                if total_loss is None:
                    total_loss = cycle_loss
                else:
                    total_loss = total_loss + cycle_loss
                    
                num_cycles += 1
            
            finally:
                # Restore original fusion weights
                for key, value in original_weights.items():
                    model.fusion_weights[key] = value
    
    # Average across all cycles
    if total_loss is not None and num_cycles > 1:
        total_loss = total_loss / num_cycles
    
    # Return zero loss if no cycles were calculated
    if total_loss is None:
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
    loss_details['cycle_loss'] = total_loss.item()
    return total_loss, loss_details

def main():
    """Train and validate a GW model with configurable fusion weights."""
    print("\n" + "="*60)
    print(" CONFIGURABLE FUSION TRAINING WITH VALIDATION ".center(60, "="))
    print("="*60 + "\n")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a GW model with configurable fusion weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--workspace-dim", type=int, default=12, help="Dimension of the workspace")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension for GW encoders/decoders")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers in GW encoders/decoders")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--multi", action="store_true", help="Train multiple models with different fusion weights")
    parser.add_argument("--n-configs", type=int, default=3, help="Number of configurations to train (with --multi)")
    parser.add_argument("--short-circuit", action="store_true", help="Short circuit training/validation for quick testing")
    parser.add_argument("--weighted-loss", action="store_true", help="Use fusion weights to weight the loss function (default: False)")
    parser.add_argument("--fusion-weight", type=float, default=1.0, help="Weight for fusion loss (default: 1.0)")
    parser.add_argument("--demi-cycle-weight", type=float, default=0.0, help="Weight for demi-cycle loss (default: 0.0)")
    parser.add_argument("--cycle-weight", type=float, default=0.0, help="Weight for cycle loss (default: 0.0)")
    parser.add_argument("--demo-text-decoding", action="store_true", help="Run a demonstration of text decoding capabilities")
    args = parser.parse_args()
    
    # Create loss weights dictionary
    loss_weights = {
        'fusion': args.fusion_weight,
        'demi_cycle': args.demi_cycle_weight,
        'cycle': args.cycle_weight
    }
    
    # Print loss configuration
    print(f"Loss configuration:")
    for loss_type, weight in loss_weights.items():
        print(f"  {loss_type}: {weight}")
    
    # Check if we can use GPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Domain module configurations
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
    
    # Check and adjust checkpoint paths
    for config in domain_configs:
        path = find_checkpoint_path(config["checkpoint_path"])
        if path:
            config["checkpoint_path"] = path
            print(f"Found checkpoint at {path}")
        else:
            print(f"Warning: Could not find checkpoint for {config['name']}")
    
    # Set up checkpoint directory
    checkpoint_dir = "checkpoints/fusion"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Load domain modules
        domain_modules = load_domain_modules(domain_configs)
        print(f"Loaded domain modules: {list(domain_modules.keys())}")
        
        # Set up data module
        try:
            # Add the simple-shapes-dataset directory to path
            import sys
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple-shapes-dataset"))
            
            from simple_shapes_dataset.data_module import SimpleShapesDataModule
            from simple_shapes_dataset.domain import DomainDesc
            
            # Find dataset path
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
            def custom_collate_fn(batch):
                # Simple collate function that handles the basic case
                from torch.utils.data._utils.collate import default_collate
                import torch
                
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
            
            # Create data loaders
            train_loader = data_module.train_dataloader(drop_last=True)
            if not args.no_validation:
                val_loader = data_module.val_dataloader()
            else:
                val_loader = None
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            import traceback
            traceback.print_exc()
            return
            
        if args.multi:
            # Train multiple models with different fusion weights
            print(f"\nTraining {args.n_configs} models with different fusion weights")
            print(f"Short-circuit mode: {'Enabled' if args.short_circuit else 'Disabled'}")
            print(f"Weighted loss: {'Enabled' if args.weighted_loss else 'Disabled (standard MSE)'}")
            results = train_multiple_configs(
                domain_modules=domain_modules,
                train_data_loader=train_loader,
                val_data_loader=val_loader,
                workspace_dim=args.workspace_dim,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                n_configs=args.n_configs,
                base_checkpoint_dir=checkpoint_dir,
                device=device,
                log_to_wandb=args.wandb,
                wandb_project="gw-fusion-multi" if args.wandb else None,
                wandb_entity=None,
                short_circuit=args.short_circuit,
                use_weighted_loss=args.weighted_loss,
                loss_weights=loss_weights
            )
        else:
            # Set up fusion weights
            print("\nUsing equal fusion weights")
            fusion_weights = {domain: 0.5 for domain in domain_modules}
            
            # Create GW model
            model = create_gw_model(
                domain_modules=domain_modules,
                workspace_dim=args.workspace_dim,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                fusion_weights=fusion_weights,
            )
            
            # Train model
            print(f"\nTraining for {args.epochs} epochs with validation {'enabled' if val_loader is not None else 'disabled'}")
            print(f"Short-circuit mode: {'Enabled' if args.short_circuit else 'Disabled'}")
            print(f"Weighted loss: {'Enabled' if args.weighted_loss else 'Disabled (standard MSE)'}")
            trained_model, checkpoint_path = train_model(
                model=model,
                train_data_loader=train_loader,
                val_data_loader=val_loader,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                device=device,
                checkpoint_dir=checkpoint_dir,
                run_name="validation_test",
                log_to_wandb=args.wandb,
                short_circuit=args.short_circuit,
                use_weighted_loss=args.weighted_loss,
                loss_weights=loss_weights
            )
            
            print(f"\nTraining completed. Checkpoint saved at: {checkpoint_path}")
            
            # Final evaluation on test data if available
            if test_dataset and not args.no_validation:
                print("\nRunning final evaluation on test set")
                test_loader = data_module.test_dataloader()
                test_loss, test_domain_losses = evaluate_model(
                    model=trained_model,
                    data_loader=test_loader,
                    device=device,
                    short_circuit=args.short_circuit,
                    use_weighted_loss=args.weighted_loss,
                    loss_weights=loss_weights
                )
                print(f"Test Loss: {test_loss:.6f}")
                for domain, loss in test_domain_losses.items():
                    print(f"  {domain}: {loss:.6f}")
        
        print("\n" + "="*60)
        print(" TRAINING COMPLETED SUCCESSFULLY ".center(60, "="))
        print("="*60 + "\n")
        
        # Run text decoding demo if requested
        if args.demo_text_decoding and not args.multi:
            if 't' in domain_modules and all(hasattr(domain_modules['t'], attr) 
                                              for attr in ['projector', 'embeddings', 'decoder', 'text_head']):
                print("Running text decoding demonstration...")
                demo_text_decoding(trained_model, device=device)
            else:
                print("Cannot run text decoding demo: Text domain module is missing required components")
        elif args.demo_text_decoding and args.multi:
            print("Text decoding demo is not available in multi-configuration mode")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPossible solutions:")
        print("1. Ensure shimmer_ssd is installed and accessible")
        print("2. Check that the checkpoint paths are correct")
        print("3. Ensure simple_shapes_dataset is installed and accessible")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    