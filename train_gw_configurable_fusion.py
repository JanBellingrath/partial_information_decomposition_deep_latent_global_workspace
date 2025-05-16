import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Mapping, Optional, List, Tuple, Union, Any

from shimmer.modules.domain import DomainModule
from shimmer.modules.selection import UniformSelection, SelectionBase
from shimmer.modules.gw import GWEncoder, GWDecoder

from gw_module_configurable_fusion import GWModuleConfigurableFusion
from generate_pid_data import generate_and_save_pid_data

# Try importing the pretrained domain loading if available
try:
    from shimmer_ssd.config import LoadedDomainConfig, DomainModuleVariant
    from shimmer_ssd.modules.domains.pretrained import load_pretrained_module
    HAS_SHIMMER_SSD = True
except ImportError:
    HAS_SHIMMER_SSD = False #TODO check if it imports the setting

print("pretrained domain loading:", HAS_SHIMMER_SSD)

def create_gw_with_configurable_fusion(
    domain_modules: Mapping[str, DomainModule],
    workspace_dim: int,
    hidden_dim: int,
    n_layers: int = 2,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> GWModuleConfigurableFusion:
    """
    Create a GWModule with configurable fusion weights and inner encoder-decoder pairs.
    
    Args:
        domain_modules: Dictionary mapping domain names to DomainModule instances
        workspace_dim: Dimension of the global workspace
        hidden_dim: Hidden dimension size for encoders and decoders
        n_layers: Number of layers in encoders and decoders
        fusion_weights: Optional dictionary of domain name to fusion weight
        
    Returns:
        GWModuleConfigurableFusion instance
    """
    # Create GW encoders and decoders for each domain
    gw_encoders = {}
    gw_decoders = {}
    
    for domain_name, domain_module in domain_modules.items():
        # Get the latent dimension from the domain module
        latent_dim = domain_module.latent_dim
        
        # Create encoder and decoder for this domain
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
    
    # Create the GW module with configurable fusion weights
    gw_module = GWModuleConfigurableFusion(
        domain_modules=domain_modules,
        workspace_dim=workspace_dim,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        fusion_weights=fusion_weights,
    )
    
    # Freeze the unimodal encoders in the domain modules
    for domain_module in domain_modules.values():
        for param in domain_module.encoder.parameters():
            param.requires_grad = False
    
    return gw_module


def generate_weight_configurations(
    domain_names: List[str],
    n_configs: int = 10,
) -> List[Dict[str, float]]:
    """
    Generate a series of weight configurations from all weight on one domain
    to equal weights to all weight on another domain.
    
    Args:
        domain_names: List of domain names (should be 2 for this function)
        n_configs: Number of configurations to generate
        
    Returns:
        List of weight configurations as dictionaries
    """
    if len(domain_names) != 2:
        raise ValueError(
            f"This function currently only supports 2 domains, got {len(domain_names)}"
        )
        
    domain_a, domain_b = domain_names
    configs = []
    
    for i in range(n_configs):
        # Calculate weight for domain A (ranging from 1.0 to 0.0)
        weight_a = 1.0 - (i / (n_configs - 1)) if n_configs > 1 else 0.5
        weight_b = 1.0 - weight_a
        
        configs.append({
            domain_a: weight_a,
            domain_b: weight_b,
        })
        
    return configs


def format_fusion_weights(fusion_weights: Dict[str, float]) -> str:
    """
    Format fusion weights into a string for use in filenames.
    
    Args:
        fusion_weights: Dictionary mapping domain names to their fusion weights
        
    Returns:
        Formatted string like "domain1_0.7_domain2_0.3"
    """
    return "_".join([f"{k}_{v:.1f}" for k, v in fusion_weights.items()])


def train_gw_model(
    gw_module: GWModuleConfigurableFusion,
    train_data_loader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3, #TODO find out the learning rate from the default run by Benjamin
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 1,
    run_name: Optional[str] = None,
) -> GWModuleConfigurableFusion:
    """
    Train the GW model with configurable fusion weights.
    
    Args:
        gw_module: The GWModuleConfigurableFusion instance to train
        train_data_loader: DataLoader providing training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        device: Device to train on ("cuda" or "cpu")
        checkpoint_dir: Directory to save checkpoints to (None = don't save)
        checkpoint_interval: Save checkpoint every N epochs
        run_name: Optional name for this training run (used in checkpoint filenames)
        
    Returns:
        Trained GWModuleConfigurableFusion
    """
    # Move model to device
    gw_module = gw_module.to(device)
    
    # Create selection module (uniform selection as it won't be used with fixed weights)
    selection_module = UniformSelection() #TODO check if this is correct
    
    # Create optimizer (only train parameters that require gradients)
    optimizer = optim.Adam(
        [p for p in gw_module.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Define reconstruction loss
    mse_loss = nn.MSELoss()
    
    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get fusion weights string for filenames if available
    fusion_weights_str = ""
    if gw_module.fusion_weights:
        fusion_weights_str = "_" + format_fusion_weights(gw_module.fusion_weights)
    
    # Run name for checkpoint
    if run_name:
        run_name = f"_{run_name}"
    else:
        run_name = ""
    
    # Training loop
    gw_module.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = gw_module(batch, selection_module)
            
            # Compute reconstruction loss for all domains
            loss = 0.0
            for domain_name, domain_input in batch.items():
                if domain_name in predictions.broadcasts:
                    domain_pred = predictions.broadcasts[domain_name]
                    loss += mse_loss(domain_pred, domain_input)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Print epoch statistics
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint if requested
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model=gw_module,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                checkpoint_dir=checkpoint_dir,
                filename=f"gw{fusion_weights_str}{run_name}_epoch_{epoch+1}.pt",
                metadata={
                    "workspace_dim": gw_module.workspace_dim,
                    "domain_names": list(gw_module.domain_mods.keys()),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "fusion_weights": gw_module.fusion_weights,
                }
            )
    
    # Save final model if using checkpoints
    if checkpoint_dir is not None:
        final_checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"gw{fusion_weights_str}{run_name}_final.pt"
        )
        
        save_checkpoint(
            model=gw_module,
            optimizer=optimizer,
            epoch=num_epochs - 1,
            loss=avg_loss,
            checkpoint_dir=checkpoint_dir,
            filename=f"gw{fusion_weights_str}{run_name}_final.pt",
            metadata={
                "workspace_dim": gw_module.workspace_dim,
                "domain_names": list(gw_module.domain_mods.keys()),
                "epoch": num_epochs,
                "loss": avg_loss,
                "fusion_weights": gw_module.fusion_weights,
                "is_final": True,
            }
        )
        
        return gw_module, final_checkpoint_path
    
    return gw_module, None


def save_checkpoint(
    model: GWModuleConfigurableFusion,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a model checkpoint to disk.
    
    Args:
        model: The model to save
        optimizer: The optimizer used for training
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_dir: Directory to save the checkpoint
        filename: Name of the checkpoint file
        metadata: Additional metadata to save with the checkpoint
    """
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Prepare the checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "fusion_weights": model.fusion_weights,
    }
    
    # Add metadata if provided
    if metadata:
        checkpoint["metadata"] = metadata
        
        # Also save metadata separately as JSON for easy inspection
        metadata_path = os.path.join(checkpoint_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    domain_modules: Optional[Mapping[str, DomainModule]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> GWModuleConfigurableFusion:
    """
    Load a model checkpoint from disk.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        domain_modules: Domain modules to use (if None, assumed to be in the checkpoint)
        device: Device to load the model onto
        
    Returns:
        Loaded GWModuleConfigurableFusion model
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract basic information and metadata
    metadata = checkpoint.get("metadata", {})
    workspace_dim = metadata.get("workspace_dim", 128)  # Default if not found
    fusion_weights = checkpoint.get("fusion_weights")
    
    # If domain modules are not provided, check if we can reconstruct them
    if domain_modules is None:
        # This would require saving the domain modules as well or reconstructing them
        raise ValueError(
            "Currently, domain_modules must be provided when loading a checkpoint. "
            "Support for storing domain modules in the checkpoint may be added in the future."
        )
    
    # Create a new model with the same architecture
    model = create_gw_with_configurable_fusion(
        domain_modules=domain_modules,
        workspace_dim=workspace_dim,
        hidden_dim=96,  # Default, can be in metadata in a real implementation
        n_layers=2,     # Default, can be in metadata in a real implementation
        fusion_weights=fusion_weights,
    )
    
    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Set to eval mode by default
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Model was trained for {checkpoint['epoch'] + 1} epochs")
    if fusion_weights:
        print(f"Fusion weights: {fusion_weights}")
    
    return model


def load_domain_modules_from_config(
    domain_configs: List[Dict[str, Any]]
) -> Dict[str, DomainModule]:
    """
    Load domain modules from configuration.
    
    Args:
        domain_configs: List of domain configuration dictionaries
        
    Returns:
        Dictionary mapping domain names to loaded domain modules
    """
    if not HAS_SHIMMER_SSD:
        raise ImportError(
            "shimmer_ssd is required to load pretrained domain modules. "
            "Please install it or provide domain modules directly."
        )
    
    domain_modules = {}
    
    for config in domain_configs:
        # Convert config to LoadedDomainConfig
        domain_config = LoadedDomainConfig(
            domain_type=DomainModuleVariant(config["domain_type"]), 
            checkpoint_path=config["checkpoint_path"],
            args=config.get("args", {})
        )
        
        # Load the module
        domain_module = load_pretrained_module(domain_config)
        domain_name = config.get("name", domain_config.domain_type.kind.value.kind)
        
        domain_modules[domain_name] = domain_module
        print(f"Loaded domain module {domain_name} from {config['checkpoint_path']}")
    
    return domain_modules


def train_multiple_fusion_configs(
    domain_modules: Dict[str, DomainModule],
    train_data_loader,
    workspace_dim: int = 128,
    hidden_dim: int = 96,
    n_layers: int = 2,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    n_configs: int = 10,
    base_checkpoint_dir: str = "checkpoints",
    generate_pid_data_after_training: bool = True,
    pid_samples: int = 10000,
    pid_save_frequency: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, str]:
    """
    Train multiple GW models with different fusion weight configurations.
    
    Args:
        domain_modules: Dictionary mapping domain names to DomainModule instances
        train_data_loader: DataLoader for training data
        workspace_dim: Dimension of the global workspace
        hidden_dim: Hidden dimension size for encoders and decoders
        n_layers: Number of layers in encoders and decoders
        num_epochs: Number of epochs to train each model
        learning_rate: Learning rate for training
        n_configs: Number of different fusion weight configurations to train
        base_checkpoint_dir: Base directory for saving checkpoints
        generate_pid_data_after_training: Whether to generate PID data after training
        pid_samples: Number of samples to generate for PID data
        pid_save_frequency: How often to save PID data batches
        device: Device to train on
        
    Returns:
        Dictionary mapping configuration names to their checkpoint paths
    """
    # Create base checkpoint directory
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    # Get domain names
    domain_names = list(domain_modules.keys())
    if len(domain_names) != 2:
        raise ValueError(
            f"This function currently only works with exactly 2 domains, got {len(domain_names)}"
        )
    
    # Generate fusion weight configurations
    weight_configs = generate_weight_configurations(domain_names, n_configs)
    
    # Train models for each configuration
    checkpoint_paths = {}
    
    for i, fusion_weights in enumerate(weight_configs):
        weights_str = format_fusion_weights(fusion_weights)
        config_name = f"config_{i}_{weights_str}"
        print(f"\n===== Training model with fusion weights: {fusion_weights} =====\n")
        
        # Create model with these fusion weights
        model = create_gw_with_configurable_fusion(
            domain_modules=domain_modules,
            workspace_dim=workspace_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            fusion_weights=fusion_weights,
        )
        
        # Train model
        config_checkpoint_dir = os.path.join(base_checkpoint_dir, config_name)
        os.makedirs(config_checkpoint_dir, exist_ok=True)
        
        _, checkpoint_path = train_gw_model(
            gw_module=model,
            train_data_loader=train_data_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir=config_checkpoint_dir,
            checkpoint_interval=1,
            run_name=config_name,
        )
        
        checkpoint_paths[config_name] = checkpoint_path
        
        # Generate PID data if requested
        if generate_pid_data_after_training and checkpoint_path:
            print(f"\n===== Generating PID data for {config_name} =====\n")
            
            # Create PID data directory
            pid_dir = f"pid_data/fusion_{weights_str}"
            os.makedirs(pid_dir, exist_ok=True)
            
            # Generate PID data
            selection_module = UniformSelection()
            generate_and_save_pid_data(
                model=model,
                selection_module=selection_module,
                output_dir=pid_dir,
                n_samples=pid_samples,
                save_frequency=pid_save_frequency,
                metadata={
                    "fusion_weights": fusion_weights,
                    "checkpoint_path": checkpoint_path,
                    "workspace_dim": workspace_dim,
                    "hidden_dim": hidden_dim,
                    "n_layers": n_layers,
                    "config_name": config_name,
                }
            )
    
    return checkpoint_paths


def main():
    """Example usage of training a GW model with configurable fusion weights."""
    
    # Example domain module configurations (replace with your actual configurations)
    domain_configs = [
        {
            "name": "image",
            "domain_type": "v_latents",
            "checkpoint_path": "checkpoints/pretrained/visual_checkpoint.ckpt",
        },
        {
            "name": "text", 
            "domain_type": "t",
            "checkpoint_path": "checkpoints/pretrained/text_checkpoint.ckpt",
        }
    ]
    
    # Load domain modules
    try:
        domain_modules = load_domain_modules_from_config(domain_configs)
    except (ImportError, FileNotFoundError) as e:
        print(f"Error loading domain modules: {e}")
        print("Please provide domain modules directly or ensure shimmer_ssd is installed.")
        return
    
    # Create data loader (replace with your actual data loader)
    # train_data_loader = ...
    
    # Train multiple models with different fusion weight configurations
    # checkpoint_paths = train_multiple_fusion_configs(
    #     domain_modules=domain_modules,
    #     train_data_loader=train_data_loader,
    #     workspace_dim=128,
    #     hidden_dim=96,
    #     n_layers=2,
    #     num_epochs=5,
    #     learning_rate=1e-3,
    #     n_configs=10,
    # )
    
    print("To train this model, replace the placeholder domain modules and data loader with your actual data.")


if __name__ == "__main__":
    main() 