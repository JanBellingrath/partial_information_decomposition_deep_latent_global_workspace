"""
Generalized data interface for PID analysis.
Supports both synthetic data and model-based data generation.
"""

import torch
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import json

from .synthetic_data import (
    generate_boolean_data, 
    save_synthetic_data, 
    load_synthetic_data,
    get_theoretical_pid_values,
    create_synthetic_labels
)
from .data import prepare_pid_data, MultimodalDataset
from .utils import generate_samples_from_model, load_checkpoint


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_data(self, n_samples: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate or load data."""
        pass
    
    @abstractmethod
    def get_domain_names(self) -> List[str]:
        """Get list of domain names."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data provider."""
        pass


class SyntheticDataProvider(DataProvider):
    """Data provider for synthetic Boolean functions."""
    
    def __init__(
        self,
        functions: Optional[List[str]] = None,
        seed: int = 42,
        theoretical_pid: bool = True
    ):
        """
        Initialize synthetic data provider.
        
        Args:
            functions: List of Boolean functions to generate
            seed: Random seed
            theoretical_pid: Whether to include theoretical PID values
        """
        if functions is None:
            functions = ['and', 'xor', 'id_a']
        
        self.functions = functions
        self.seed = seed
        self.theoretical_pid = theoretical_pid
        
        # Domain names are input sources and target functions
        self.domain_names = ['input_a', 'input_b'] + functions
    
    def get_data(self, n_samples: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate synthetic Boolean function data."""
        # Extract noise parameters from kwargs
        add_noise = kwargs.get('add_noise', True)
        noise_std = kwargs.get('noise_std', 0.1)
        
        return generate_boolean_data(
            n_samples=n_samples,
            seed=self.seed,
            functions=self.functions,
            add_noise=add_noise,
            noise_std=noise_std
        )
    
    def get_domain_names(self) -> List[str]:
        """Get domain names."""
        return self.domain_names.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about synthetic data."""
        metadata = {
            'provider_type': 'synthetic',
            'functions': self.functions,
            'seed': self.seed,
            'domain_names': self.domain_names
        }
        
        if self.theoretical_pid:
            metadata['theoretical_pid'] = get_theoretical_pid_values()
        
        return metadata


class ModelDataProvider(DataProvider):
    """Data provider for model-based data generation."""
    
    def __init__(
        self,
        model_path: str,
        domain_modules: Dict[str, Any],
        data_module=None,
        dataset_split: str = "test",
        use_gw_encoded: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize model data provider.
        
        Args:
            model_path: Path to model checkpoint
            domain_modules: Dictionary of domain modules
            data_module: Data module for dataset-based sampling
            dataset_split: Which split to use for dataset sampling
            use_gw_encoded: Whether to use GW-encoded data
            device: Device to run model on
        """
        self.model_path = model_path
        self.domain_modules = domain_modules
        self.data_module = data_module
        self.dataset_split = dataset_split
        self.use_gw_encoded = use_gw_encoded
        self.device = device
        
        # Extract domain names from domain_modules
        self.domain_names = list(domain_modules.keys())
        
        # Load model for inspection
        self.model = None
    
    def get_data(self, n_samples: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate data from model."""
        batch_size = kwargs.get('batch_size', 128)
        
        if self.model is None:
            self.model = load_checkpoint(
                self.model_path,
                self.domain_modules,
                self.device
            )
        
        return generate_samples_from_model(
            model=self.model,
            domain_names=self.domain_names,
            n_samples=n_samples,
            batch_size=batch_size,
            device=self.device,
            use_gw_encoded=self.use_gw_encoded,
            data_module=self.data_module,
            dataset_split=self.dataset_split
        )
    
    def get_domain_names(self) -> List[str]:
        """Get domain names."""
        return self.domain_names.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about model data."""
        return {
            'provider_type': 'model',
            'model_path': self.model_path,
            'domain_names': self.domain_names,
            'dataset_split': self.dataset_split,
            'use_gw_encoded': self.use_gw_encoded,
            'device': self.device
        }


class FileDataProvider(DataProvider):
    """Data provider for pre-saved data files."""
    
    def __init__(
        self,
        data_dir: str,
        domain_names: List[str],
        file_pattern: str = "{domain}.pt"
    ):
        """
        Initialize file data provider.
        
        Args:
            data_dir: Directory containing data files
            domain_names: List of domain names to load
            file_pattern: Pattern for filenames (use {domain} placeholder)
        """
        self.data_dir = Path(data_dir)
        self.domain_names = domain_names
        self.file_pattern = file_pattern
        
        # Validate that files exist
        missing_files = []
        for domain in domain_names:
            filepath = self.data_dir / self.file_pattern.format(domain=domain)
            if not filepath.exists():
                missing_files.append(str(filepath))
        
        if missing_files:
            raise FileNotFoundError(f"Missing data files: {missing_files}")
    
    def get_data(self, n_samples: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Load data from files."""
        data = {}
        
        for domain in self.domain_names:
            filepath = self.data_dir / self.file_pattern.format(domain=domain)
            tensor = torch.load(filepath)
            
            # Truncate to requested number of samples
            if len(tensor) > n_samples:
                tensor = tensor[:n_samples]
            
            data[domain] = tensor
        
        return data
    
    def get_domain_names(self) -> List[str]:
        """Get domain names."""
        return self.domain_names.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about file data."""
        return {
            'provider_type': 'file',
            'data_dir': str(self.data_dir),
            'domain_names': self.domain_names,
            'file_pattern': self.file_pattern
        }


class GeneralizedDataInterface:
    """Generalized interface for data loading and PID analysis."""
    
    def __init__(self, data_provider: DataProvider):
        """
        Initialize with a data provider.
        
        Args:
            data_provider: Instance of DataProvider subclass
        """
        self.data_provider = data_provider
    
    def prepare_pid_data(
        self,
        source_config: Dict[str, str],
        target_config: str,
        n_samples: int = 10000,
        synthetic_labels: Optional[torch.Tensor] = None,
        num_clusters: int = 10,
        cluster_method: str = 'gmm',
        **provider_kwargs
    ) -> Tuple[MultimodalDataset, MultimodalDataset, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for PID analysis using the configured provider.
        
        Args:
            source_config: Configuration for source domains
            target_config: Target domain name  
            n_samples: Number of samples to generate/load
            synthetic_labels: Pre-computed labels (optional)
            num_clusters: Number of clusters for label generation
            cluster_method: Clustering method
            **provider_kwargs: Additional arguments for data provider
        
        Returns:
            Tuple of (train_dataset, test_dataset, x1, x2, labels)
        """
        # Get data from provider
        generated_data = self.data_provider.get_data(n_samples, **provider_kwargs)
        domain_names = self.data_provider.get_domain_names()
        
        # Fix source_config for synthetic data - map domain names to actual keys
        fixed_source_config = {}
        for domain_key, target_key in source_config.items():
            # For each domain in source_config, find the correct data key
            if target_key in generated_data:
                # Use the target_key as-is if it exists
                fixed_source_config[domain_key] = target_key
            elif domain_key in generated_data:
                # Use the domain_key directly if it exists in data  
                fixed_source_config[domain_key] = domain_key
            else:
                # Try mapping domain_key to the corresponding data key
                # For synthetic data, domain_a -> input_a, domain_b -> input_b
                if domain_key == 'domain_a' and 'input_a' in generated_data:
                    fixed_source_config[domain_key] = 'input_a'
                elif domain_key == 'domain_b' and 'input_b' in generated_data:
                    fixed_source_config[domain_key] = 'input_b'
                elif target_key == 'input_a' and 'input_a' in generated_data:
                    fixed_source_config[domain_key] = 'input_a'
                elif target_key == 'input_b' and 'input_b' in generated_data:
                    fixed_source_config[domain_key] = 'input_b'
                else:
                    # Keep original mapping
                    fixed_source_config[domain_key] = target_key
        
        # Also ensure the domain_names match what's expected by prepare_pid_data
        # The prepare_pid_data function expects domain_names to be the actual domain identifiers
        # that will be used as keys in source_config
        relevant_domains = []
        for domain_key in source_config.keys():
            relevant_domains.append(domain_key)
        
        if len(relevant_domains) >= 2:
            domain_names = relevant_domains[:2]
        
        # If no synthetic labels provided, generate them from target data
        if synthetic_labels is None:
            if target_config in generated_data:
                target_data = generated_data[target_config]
                synthetic_labels = create_synthetic_labels(
                    data=target_data,
                    num_clusters=num_clusters,
                    cluster_method=cluster_method
                )
            else:
                raise ValueError(f"Target '{target_config}' not found in data and no synthetic_labels provided")
        
        # Use existing prepare_pid_data function
        return prepare_pid_data(
            generated_data=generated_data,
            domain_names=domain_names,
            source_config=fixed_source_config,
            target_config=target_config,
            synthetic_labels=synthetic_labels  # Now mandatory
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the data provider."""
        return self.data_provider.get_metadata()
    
    def save_data(
        self,
        output_dir: str,
        n_samples: int = 10000,
        prefix: str = "data",
        **provider_kwargs
    ) -> Dict[str, str]:
        """
        Save data from provider to disk.
        
        Args:
            output_dir: Directory to save data
            n_samples: Number of samples to generate/load
            prefix: Prefix for saved files
            **provider_kwargs: Additional arguments for data provider
        
        Returns:
            Dictionary mapping domain names to file paths
        """
        data = self.data_provider.get_data(n_samples, **provider_kwargs)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for domain, tensor in data.items():
            filename = f"{prefix}_{domain}.pt"
            filepath = output_path / filename
            torch.save(tensor, filepath)
            saved_files[domain] = str(filepath)
        
        # Save metadata
        metadata = self.get_metadata()
        metadata_path = output_path / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return saved_files


def create_data_interface(
    data_type: str,
    **kwargs
) -> GeneralizedDataInterface:
    """
    Factory function to create data interface.
    
    Args:
        data_type: Type of data ('synthetic', 'model', 'file')
        **kwargs: Arguments for the specific data provider
    
    Returns:
        GeneralizedDataInterface instance
    """
    if data_type == 'synthetic':
        provider = SyntheticDataProvider(**kwargs)
    elif data_type == 'model':
        provider = ModelDataProvider(**kwargs)
    elif data_type == 'file':
        provider = FileDataProvider(**kwargs)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return GeneralizedDataInterface(provider)


# Convenience functions for backward compatibility
def create_synthetic_interface(
    functions: Optional[List[str]] = None,
    seed: int = 42
) -> GeneralizedDataInterface:
    """Create interface for synthetic Boolean data."""
    return create_data_interface(
        'synthetic',
        functions=functions,
        seed=seed
    )


def create_model_interface(
    model_path: str,
    domain_modules: Dict[str, Any],
    **kwargs
) -> GeneralizedDataInterface:
    """Create interface for model-based data."""
    return create_data_interface(
        'model',
        model_path=model_path,
        domain_modules=domain_modules,
        **kwargs
    ) 