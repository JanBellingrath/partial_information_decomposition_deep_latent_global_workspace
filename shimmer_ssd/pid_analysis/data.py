import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import sys # For sys.path manipulation
from pathlib import Path # For path manipulation

# Global device setting (consider moving to a config or utils module)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalDataset(Dataset):
    """
    Dataset for multimodal data.
    
    This dataset handles multiple tensors (one per modality) and labels.
    All tensors are kept on CPU to avoid CUDA initialization errors in worker processes.
    Labels can be either hard labels (integers) or soft labels (probability distributions).
    """
    
    def __init__(self, data: List[torch.Tensor], labels: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            data: List of tensors, one per modality
            labels: Tensor of labels (can be either hard labels or soft probabilities)
        """
        # Move all tensors to CPU
        self.data = [t.cpu() if isinstance(t, torch.Tensor) else t for t in data]
        self.labels = labels.cpu() if isinstance(labels, torch.Tensor) else labels
        
        # Validate data
        n_samples = self.data[0].size(0)
        for i, tensor in enumerate(self.data):
            if tensor.size(0) != n_samples:
                raise ValueError(
                    f"All data tensors must have the same first dimension. "
                    f"Got {tensor.size(0)} for tensor {i}, expected {n_samples}"
                )
        
        if self.labels.size(0) != n_samples:
            raise ValueError(
                f"Labels must have the same first dimension as data. "
                f"Got {self.labels.size(0)}, expected {n_samples}"
            )
        
        # Log dimensions and type
        print(f"MultimodalDataset init:")
        print(f"├─ Labels shape: {self.labels.shape}")
        print(f"├─ Labels dim: {self.labels.dim()}")
        print(f"└─ Labels type: {'soft' if self.labels.dim() > 1 else 'hard'}")
        print(f"All tensors moved to CPU for DataLoader worker compatibility")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.data[0].size(0)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get sample by index.
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (modality1, modality2, ..., label)
            For GMM, label is a probability distribution over clusters
            For kmeans, label is a single integer
        """
        # Get data for each modality (already on CPU)
        modalities = [tensor[idx] for tensor in self.data]
        
        # Get label (already on CPU)
        label = self.labels[idx]
        
        # Return as tuple
        return tuple(modalities) + (label,)

def prepare_pid_data(
    generated_data: Dict[str, torch.Tensor],
    domain_names: List[str],
    source_config: Dict[str, str],
    target_config: str,  # Kept for consistency, though labels are now mandatory
    synthetic_labels: torch.Tensor
) -> Tuple[MultimodalDataset, MultimodalDataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data for PID analysis.

    Args:
        generated_data: Dictionary of generated data
        domain_names: List of domain names
        source_config: Dictionary mapping domain names to source representations
        target_config: Target representation (used for consistency, not for label generation)
        synthetic_labels: Pre-computed synthetic labels. This is now a mandatory argument.

    Returns:
        Tuple of (train_dataset, test_dataset, domain1_features, domain2_features, labels)
    """
    print(f"Preparing data for PID analysis with provided synthetic labels.")
    available_keys = list(generated_data.keys())
    print(f"Available keys in generated_data: {available_keys}")
    
    if len(domain_names) < 2:
        print("Warning: Insufficient domain names provided. Attempting to extract from generated data.")
        extracted_domains = []
        for key in available_keys:
            if key == 'gw_rep':
                continue
            for suffix in ['_latent', '_decoded', '_gw_encoded']:
                if key.endswith(suffix):
                    domain = key[:-len(suffix)]
                    if domain not in extracted_domains:
                        extracted_domains.append(domain)
        
        print(f"Extracted domain prefixes: {extracted_domains}")
        if len(extracted_domains) >= 2:
            domain_names = extracted_domains[:2]
        print(f"Using domain names: {domain_names}")
        if len(domain_names) < 2:
            raise ValueError(f"Need at least 2 domains, got {len(domain_names)} even after extraction")
    
    domain_a, domain_b = domain_names[:2]
    
    source_key_a = source_config.get(domain_a, None)
    source_key_b = source_config.get(domain_b, None)
    
    if source_key_a is None:
        source_key_a = f"{domain_a}_latent"
        if source_key_a not in generated_data:
            source_key_a = f"{domain_a}_decoded"
    
    if source_key_b is None:
        source_key_b = f"{domain_b}_latent"
        if source_key_b not in generated_data:
            source_key_b = f"{domain_b}_decoded"
    
    if source_key_a not in generated_data:
        raise KeyError(f"Source key '{source_key_a}' for domain '{domain_a}' not found in generated data. Available keys: {available_keys}")
    if source_key_b not in generated_data:
        raise KeyError(f"Source key '{source_key_b}' for domain '{domain_b}' not found in generated data. Available keys: {available_keys}")
    if target_config not in generated_data:
        raise KeyError(f"Target key '{target_config}' not found in generated data. Available keys: {available_keys}")
    
    x1 = generated_data[source_key_a]
    x2 = generated_data[source_key_b]
    
    # Synthetic labels are now a mandatory input
    labels = synthetic_labels
    
    # Move to CPU for DataLoader compatibility (devices will be handled in training)
    x1 = x1.to('cpu')
    x2 = x2.to('cpu')
    labels = labels.to('cpu')
    
    n_samples = x1.size(0)
    n_train = int(0.8 * n_samples)
    
    train_indices = torch.arange(n_train)
    test_indices = torch.arange(n_train, n_samples)
    
    train_ds = MultimodalDataset(
        data=[x1[train_indices], x2[train_indices]],
        labels=labels[train_indices]
    )
    
    test_ds = MultimodalDataset(
        data=[x1[test_indices], x2[test_indices]],
        labels=labels[test_indices]
    )
    
    return train_ds, test_ds, x1, x2, labels

def process_domain_data(domain_data, domain_name):
    """
    Helper function to extract the actual tensor data from the domain data structure.
    Handles different domain data formats from SimpleShapesDataModule.
    
    Args:
        domain_data: The data returned for a domain
        domain_name: The name of the domain
    
    Returns:
        The tensor data for the domain
    """
    if isinstance(domain_data, dict):
        if domain_name in domain_data:
            data = domain_data[domain_name]
        else:
            for k, v in domain_data.items():
                if isinstance(k, frozenset) and domain_name in k:
                    data = v
                    break
            else:
                data = domain_data
    else:
        data = domain_data
    
    if hasattr(data, 'bert'):
        return data.bert
    elif isinstance(data, dict) and domain_name in data:
        value = data[domain_name]
        if hasattr(value, 'bert'):
            return value.bert
        elif isinstance(value, torch.Tensor):
            return value
        else:
            raise ValueError(f"Unsupported nested value type for {domain_name}: {type(value)}")
    elif hasattr(data, 'category'):
        attrs = [
            data.category.float() if not isinstance(data.category, torch.Tensor) else data.category.float(),
            data.x.float() if not isinstance(data.x, torch.Tensor) else data.x.float(),
            data.y.float() if not isinstance(data.y, torch.Tensor) else data.y.float(),
            data.size.float() if not isinstance(data.size, torch.Tensor) else data.size.float(),
            data.rotation.float() if not isinstance(data.rotation, torch.Tensor) else data.rotation.float(),
            data.color_r.float() if not isinstance(data.color_r, torch.Tensor) else data.color_r.float(),
            data.color_g.float() if not isinstance(data.color_g, torch.Tensor) else data.color_g.float(),
            data.color_b.float() if not isinstance(data.color_b, torch.Tensor) else data.color_b.float()
        ]
        return torch.stack(attrs, dim=-1)
    elif isinstance(data, torch.Tensor):
        if domain_name == 'v_latents' and data.dim() > 2:
            return data[:, 0, :]
        return data
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"Empty list for {domain_name}")
        processed_items = []
        for item in data:
            if isinstance(item, dict):
                if domain_name in item:
                    value = item[domain_name]
                    if hasattr(value, 'bert'):
                        processed_items.append(value.bert)
                    elif isinstance(value, torch.Tensor):
                        processed_items.append(value)
                    else:
                        raise ValueError(f"Unsupported nested value type in list for {domain_name}: {type(value)}")
                else:
                    for k, v in item.items():
                        if hasattr(v, 'bert'):
                            processed_items.append(v.bert)
                            break
                        elif isinstance(v, torch.Tensor):
                            processed_items.append(v)
                            break
            elif hasattr(item, 'bert'):
                processed_items.append(item.bert)
            elif isinstance(item, torch.Tensor):
                processed_items.append(item)
            elif hasattr(item, 'category'):
                attrs = torch.tensor([
                    item.category.float() if not isinstance(item.category, torch.Tensor) else item.category.float(),
                    item.x.float() if not isinstance(item.x, torch.Tensor) else item.x.float(),
                    item.y.float() if not isinstance(item.y, torch.Tensor) else item.y.float(),
                    item.size.float() if not isinstance(item.size, torch.Tensor) else item.size.float(),
                    item.rotation.float() if not isinstance(item.rotation, torch.Tensor) else item.rotation.float(),
                    item.color_r.float() if not isinstance(item.color_r, torch.Tensor) else item.color_r.float(),
                    item.color_g.float() if not isinstance(item.color_g, torch.Tensor) else item.color_g.float(),
                    item.color_b.float() if not isinstance(item.color_b, torch.Tensor) else item.color_b.float()
                ])
                processed_items.append(attrs)
        if not processed_items:
            raise ValueError(f"Could not extract any valid tensors for {domain_name}")
        try:
            return torch.stack(processed_items)
        except Exception as e:
            print(f"Error stacking tensors for {domain_name}: {e}")
            print(f"First processed item type: {type(processed_items[0])}")
            print(f"First processed item shape: {processed_items[0].shape if hasattr(processed_items[0], 'shape') else 'unknown'}")
            raise ValueError(f"Could not stack tensors for {domain_name}: {e}")
    raise ValueError(f"Unsupported domain data type for {domain_name}: {type(data)}")

def prepare_dataset_pid_data(
    generated_data: Dict[str, torch.Tensor],
    domain_names: List[str],
    analysis_domain: str,
    synthetic_labels: torch.Tensor # Made mandatory
):
    """
    Prepares PID data for a specific dataset structure.
    
    Args:
        generated_data: A dictionary containing tensors for different data representations.
                        Example: {'latent_A': tensor_A, 'latent_B': tensor_B, 'target_representation': tensor_target}
        domain_names: A list of two strings representing the names of the domains to be analyzed (e.g., ['domainA', 'domainB']).
        analysis_domain: String, name of the domain used to derive the target variable 'y'.
        synthetic_labels: Pre-computed synthetic labels. This is now a mandatory argument.

    Returns:
        A tuple containing:
        - train_ds (MultimodalDataset): Training dataset.
        - test_ds (MultimodalDataset): Testing dataset.
        - x1 (torch.Tensor): Data from the first domain.
        - x2 (torch.Tensor): Data from the second domain.
        - labels (torch.Tensor): The synthetic labels used for analysis.
    """
    
    print(f"Preparing dataset PID data with provided synthetic labels for analysis_domain: {analysis_domain}")
    available_keys = list(generated_data.keys())
    print(f"Available keys in generated_data: {available_keys}")

    if len(domain_names) != 2:
        raise ValueError(f"Expected 2 domain names, got {len(domain_names)}")

    domain_a, domain_b = domain_names[0], domain_names[1]

    # Construct keys for accessing data from generated_data
    # Assuming a naming convention like 'latent_domainA', 'latent_domainB'
    # If your keys are different, you might need to adjust this logic or pass specific keys.
    key_a = f"latent_{domain_a}" 
    key_b = f"latent_{domain_b}"

    if key_a not in generated_data:
        raise KeyError(f"Data for domain '{domain_a}' (key: '{key_a}') not found in generated_data. Available keys: {available_keys}")
    if key_b not in generated_data:
        raise KeyError(f"Data for domain '{domain_b}' (key: '{key_b}') not found in generated_data. Available keys: {available_keys}")

    x1 = generated_data[key_a]
    x2 = generated_data[key_b]
    
    # Synthetic labels are now a mandatory input
    labels = synthetic_labels
    
    # Move to CPU for DataLoader compatibility (devices will be handled in training)
    x1 = x1.to('cpu')
    x2 = x2.to('cpu')
    labels = labels.to('cpu')

    # Create train/test split
    n_samples = x1.size(0)
    n_train = int(0.8 * n_samples)
    
    train_indices = torch.arange(n_train)
    test_indices = torch.arange(n_train, n_samples)
    
    # Create MultimodalDataset instances
    train_ds = MultimodalDataset(
        data=[x1[train_indices], x2[train_indices]],
        labels=labels[train_indices]
    )
    
    test_ds = MultimodalDataset(
        data=[x1[test_indices], x2[test_indices]],
        labels=labels[test_indices]
    )
    
    return train_ds, test_ds, x1, x2, labels

# Ensure device is defined if it's used within these functions, 
# or pass it as an argument. For now, assuming it's globally available.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --- Add new function for SimpleShapesDataModule setup ---
def create_simple_shapes_data_module(
    dataset_path_arg: Optional[str],
    domain_modules_loaded: Dict[str, Any], # Using Any for DomainModule duck-typing
    batch_size: int,
    # dataset_split_arg: str, # This argument seems unused by the original setup logic for the module itself
    #                         # The split is used when calling dataloader() on the instance.
    # For now, the function will return the setup data_module.
    # The caller can then get the appropriate dataloader.
    shapes_dataset_lib_path_override: Optional[str] = None, # For testing or specific structures
    shapes_domain_args_override: Optional[Dict[str,Dict]] = None # Allow overriding domain_args
) -> Optional[Any]: # Returns a SimpleShapesDataModule instance or None
    """
    Sets up and returns a SimpleShapesDataModule instance.
    This function encapsulates the logic previously in main.py for setting up this specific data module.

    Args:
        dataset_path_arg: Path to the dataset (e.g., SimpleShapes dataset) from args.
        domain_modules_loaded: Dictionary of loaded domain modules (e.g., from utils.load_domain_modules).
                               Used to configure SimpleShapes domain_classes.
        batch_size: Batch size for the data module.
        shapes_dataset_lib_path_override: Optional override for the path to simple-shapes-dataset library.
        shapes_domain_args_override: Optional override for the domain_args passed to SimpleShapesDataModule.

    Returns:
        An initialized SimpleShapesDataModule instance, or None if setup fails.
    """
    data_module_instance = None
    print("Attempting to set up SimpleShapesDataModule...")

    try:
        # --- 1. Ensure simple-shapes-dataset library is importable ---
        try:
            # First, try a direct import
            from simple_shapes_dataset.data_module import SimpleShapesDataModule
            from simple_shapes_dataset.domain import DomainDesc
            print("Successfully imported SimpleShapesDataModule.")
        except ImportError:
            print("SimpleShapesDataModule not directly importable. Attempting to find and add to sys.path.")
            # This path logic is from the original script and might be fragile.
            paths_to_try = []
            if shapes_dataset_lib_path_override:
                paths_to_try.append(Path(shapes_dataset_lib_path_override))
            
            # Path relative to this file (pid_analysis/data.py)
            # pid_analysis/ -> project_root / simple-shapes-dataset
            paths_to_try.append(Path(__file__).resolve().parent.parent / "simple-shapes-dataset")

            if dataset_path_arg:
                # Path relative to the provided dataset_path argument
                # Assuming dataset_path_arg might be /path/to/some_dataset_root/simple_shapes_dataset
                # then simple-shapes-dataset is dataset_path_arg.parent
                paths_to_try.append(Path(dataset_path_arg).parent) 
                # Or if dataset_path_arg is /path/to/simple-shapes-dataset
                paths_to_try.append(Path(dataset_path_arg))


            # Fallback to a direct name guess
            paths_to_try.append(Path("simple-shapes-dataset"))

            simple_shapes_lib_found = False
            for lib_path_candidate in paths_to_try:
                if lib_path_candidate.is_dir() and (lib_path_candidate / "simple_shapes_dataset" / "__init__.py").exists():
                    # This means lib_path_candidate is likely the PARENT of the actual simple_shapes_dataset package dir
                    sys.path.append(str(lib_path_candidate))
                    print(f"Added '{lib_path_candidate}' to sys.path for SimpleShapesDataModule.")
                    try:
                        from simple_shapes_dataset.data_module import SimpleShapesDataModule
                        from simple_shapes_dataset.domain import DomainDesc
                        print("Successfully imported SimpleShapesDataModule after adding path.")
                        simple_shapes_lib_found = True
                        break
                    except ImportError:
                        print(f"Failed to import from '{lib_path_candidate}' even after adding to path.")
                        sys.path.remove(str(lib_path_candidate)) # Clean up if import failed
                elif lib_path_candidate.is_dir() and (lib_path_candidate / "__init__.py").exists() and lib_path_candidate.name == "simple_shapes_dataset":
                    # This means lib_path_candidate IS the simple_shapes_dataset package dir, add its parent
                    sys.path.append(str(lib_path_candidate.parent))
                    print(f"Added '{lib_path_candidate.parent}' to sys.path for SimpleShapesDataModule.")
                    try:
                        from simple_shapes_dataset.data_module import SimpleShapesDataModule
                        from simple_shapes_dataset.domain import DomainDesc
                        print("Successfully imported SimpleShapesDataModule after adding parent path.")
                        simple_shapes_lib_found = True
                        break
                    except ImportError:
                        print(f"Failed to import from '{lib_path_candidate.parent}' even after adding to path.")
                        sys.path.remove(str(lib_path_candidate.parent))


            if not simple_shapes_lib_found:
                raise ImportError("SimpleShapesDataModule library not found or could not be imported. "
                                  "Please ensure 'simple-shapes-dataset' is installed or correctly pathed.")

        # --- 2. Determine the actual dataset path ---
        dataset_actual_path = dataset_path_arg
        if dataset_actual_path is None or not Path(dataset_actual_path).exists():
            print(f"Dataset path '{dataset_path_arg}' not provided or not found. Trying default locations...")
            # Default path logic from original script, relative to this file's project root
            project_root = Path(__file__).resolve().parent.parent
            full_dataset_path_check = project_root / "full_shapes_dataset" / "simple_shapes_dataset"
            # Assuming simple_shapes_path was found correctly above or is a common location
            # This part is tricky because simple_shapes_path discovery is dynamic.
            # For simplicity, let's assume a common structure if defaults are used.
            sample_dataset_path_check = project_root / "simple-shapes-dataset" / "sample_dataset"

            if full_dataset_path_check.exists():
                dataset_actual_path = str(full_dataset_path_check)
                print(f"Using default full dataset: {dataset_actual_path}")
            elif sample_dataset_path_check.exists():
                dataset_actual_path = str(sample_dataset_path_check)
                print(f"Full dataset not found, falling back to sample dataset at: {dataset_actual_path}")
            else:
                raise FileNotFoundError("Dataset path not specified and default dataset locations not found. Please specify --dataset-path.")
        
        print(f"Using dataset for SimpleShapesDataModule at: {dataset_actual_path}")

        # --- 3. Configure domain_classes and domain_args ---
        domain_classes = {}
        # domain_args_dict is used by SimpleShapesDataModule constructor
        domain_args_for_module = {} 
        if shapes_domain_args_override:
            domain_args_for_module = shapes_domain_args_override
        
        # This logic needs access to domain_modules_loaded to configure SimpleShapes
        # It maps the 'name' from domain_modules_loaded (e.g., "v_latents", "t")
        # to the specific class and arguments required by SimpleShapesDataModule.
        for dm_name, dm_instance in domain_modules_loaded.items():
            # This mapping is highly specific to SimpleShapes and the expected domain names/types.
            if dm_name == "v_latents": # Name used in load_domain_modules
                # Add both raw images and latent representations for visualization
                from simple_shapes_dataset.domain import SimpleShapesPretrainedVisual, SimpleShapesImages
                from torchvision.transforms import ToTensor
                
                # Add raw images domain for actual visualization
                domain_classes[DomainDesc(base="v", kind="v")] = SimpleShapesImages
                if "v" not in domain_args_for_module:
                    domain_args_for_module["v"] = {}
                
                # Try to add latent representations for clustering if saved_latents exist
                try:
                    # Check if saved_latents directory exists before configuring v_latents
                    saved_latents_path = Path(dataset_actual_path) / "saved_latents"
                    if saved_latents_path.exists() and any(saved_latents_path.glob("*/*.npy")):
                        print(f"   ✅ Found saved_latents directory, enabling v_latents domain")
                        domain_classes[DomainDesc(base="v", kind="v_latents")] = SimpleShapesPretrainedVisual
                        if "v_latents" not in domain_args_for_module:
                            domain_args_for_module["v_latents"] = { 
                                "presaved_path": "calmip-822888_epoch=282-step=1105680_future.npy",
                                "use_unpaired": False
                            }
                    else:
                        print(f"   ⚠️  No saved_latents directory found, skipping v_latents domain")
                        print(f"   🔄 Will use raw images (v) for both visualization and clustering")
                except Exception as e:
                    print(f"   ⚠️  Error checking saved_latents: {e}, skipping v_latents domain")
                
            elif dm_name == "t": # Name used in load_domain_modules
                from simple_shapes_dataset.domain import SimpleShapesText
                domain_classes[DomainDesc(base="t", kind="t")] = SimpleShapesText
                # Add any default domain_args for "t" if necessary and not overridden
            # Add other domain mappings if necessary, based on how domain_modules are named/typed.

        if not domain_classes:
            print("Warning: No domain_classes configured for SimpleShapesDataModule based on loaded domain_modules. Module might not work correctly.")
        
        # Domain proportions: include both raw images and latents for v_latents domains
        # This also depends on the names used in domain_modules_loaded matching expected SimpleShapes bases.
        domain_proportions = {}
        for dm_name, dm_instance in domain_modules_loaded.items():
            if dm_name == "v_latents":
                # Include both raw images and latents
                domain_proportions[frozenset(["v"])] = 1.0  # Raw images for visualization
                domain_proportions[frozenset(["v_latents"])] = 1.0  # Latents for clustering
            elif dm_name == "t":
                domain_proportions[frozenset(["t"])] = 1.0
        
        if not domain_proportions:
            # Fallback if names don't match bases directly (e.g. "v_latents" vs "v")
            domain_proportions = {frozenset([dd.base]): 1.0 for dd in domain_classes.keys()}


        # --- 4. Define the local_custom_collate_fn ---
        # This collate function was part of the original main.py's data_module setup.
        # It needs to handle the batch structure produced by SimpleShapesDataModule.
        def local_custom_collate_fn(batch):
            from torch.utils.data._utils.collate import default_collate
            from torchvision.transforms import ToTensor
            
            # The structure of 'batch' depends on SimpleShapesDataModule's output.
            # Original collate was complex. Starting with a simpler one, may need refinement.
            if isinstance(batch, list) and len(batch) > 0:
                # Common case: list of dictionaries, where each dict is a sample
                # and keys in dict are DomainDesc or frozensets of DomainDesc.
                if isinstance(batch[0], dict):
                    # We need to collate items for each domain across the batch.
                    # Example: batch = [ {DomainDesc('v'): tens_v1, DomainDesc('t'): tens_t1}, 
                    #                    {DomainDesc('v'): tens_v2, DomainDesc('t'): tens_t2} ]
                    # Result should be: { DomainDesc('v'): collated_v, DomainDesc('t'): collated_t }
                    
                    # Get all unique keys (domains) present in the batch
                    all_keys = set()
                    for sample_dict in batch:
                        all_keys.update(sample_dict.keys())
                    
                    collated_batch = {}
                    to_tensor = ToTensor()
                    
                    for key_domain_desc in all_keys:
                        # Gather all items for this key_domain_desc from the batch
                        items_for_domain = [sample_dict[key_domain_desc] for sample_dict in batch if key_domain_desc in sample_dict]
                        if items_for_domain:
                            # Special handling for different domain types
                            is_text_domain = (isinstance(key_domain_desc, DomainDesc) and key_domain_desc.base == 't') or \
                                             (isinstance(key_domain_desc, frozenset) and any(d.base == 't' for d in key_domain_desc))
                            
                            is_raw_image_domain = (isinstance(key_domain_desc, DomainDesc) and key_domain_desc.base == 'v' and key_domain_desc.kind == 'v') or \
                                                  (isinstance(key_domain_desc, frozenset) and any(d.base == 'v' and d.kind == 'v' for d in key_domain_desc))

                            if is_text_domain and hasattr(items_for_domain[0], 'bert'): # Example: list of Text objects
                                bert_tensors = [item.bert for item in items_for_domain]
                                collated_batch[key_domain_desc] = default_collate(bert_tensors)
                            elif is_text_domain and isinstance(items_for_domain[0], dict) and 'bert' in items_for_domain[0]:
                                bert_tensors = [item['bert'] for item in items_for_domain]
                                collated_batch[key_domain_desc] = default_collate(bert_tensors)
                            elif is_raw_image_domain:
                                # Handle raw PIL images - convert to tensors
                                try:
                                    # Apply ToTensor transform to PIL images
                                    tensor_items = []
                                    for item in items_for_domain:
                                        if hasattr(item, 'mode'):  # PIL Image
                                            tensor_items.append(to_tensor(item))
                                        else:  # Already a tensor
                                            tensor_items.append(item)
                                    collated_batch[key_domain_desc] = default_collate(tensor_items)
                                except Exception as e_image:
                                    print(f"Image collate warning for key {key_domain_desc}: {e_image}. Storing as list.")
                                    collated_batch[key_domain_desc] = items_for_domain
                            else:
                                try:
                                    collated_batch[key_domain_desc] = default_collate(items_for_domain)
                                except Exception as e_collate_inner:
                                    print(f"Collate warning for key {key_domain_desc}: {e_collate_inner}. Storing as list.")
                                    collated_batch[key_domain_desc] = items_for_domain # Fallback
                        else: # Should not happen if all_keys is derived correctly
                            collated_batch[key_domain_desc] = []
                    return collated_batch
            # Fallback to default_collate if structure is not list of dicts
            try:
                return default_collate(batch)
            except Exception as e_collate_default:
                print(f"Default collate failed: {e_collate_default}. Returning batch as is.")
                return batch

        # --- 5. Initialize and setup SimpleShapesDataModule ---
        if domain_classes: # Only proceed if we have something to load
            data_module_instance = SimpleShapesDataModule(
                dataset_path=str(dataset_actual_path),
                domain_classes=domain_classes,
                domain_proportions=domain_proportions,
                batch_size=batch_size,
                num_workers=0,  # Keep low (0 or 1) for debugging to avoid CUDA errors in workers
                seed=42,        # Consistent seed for reproducibility
                domain_args=domain_args_for_module, # Pass the potentially overridden domain_args
                collate_fn=local_custom_collate_fn 
            )
            data_module_instance.setup() # Prepare datasets (train, val, test)
            print(f"SimpleShapesDataModule setup complete. Datasets ready.")
            # Example: train_loader = data_module_instance.train_dataloader()
        else:
            print("Skipping SimpleShapesDataModule setup as no domain_classes were derived from loaded domain modules.")
            data_module_instance = None

    except ImportError as e_import:
        print(f"Failed to import SimpleShapesDataModule or its dependencies: {e_import}. "
              "Dataset features will be unavailable.")
        data_module_instance = None
    except FileNotFoundError as e_file:
        print(f"Dataset file error for SimpleShapesDataModule: {e_file}. "
              "Dataset features will be unavailable.")
        data_module_instance = None
    except Exception as e_general:
        print(f"An unexpected error occurred during SimpleShapesDataModule setup: {e_general}")
        import traceback
        traceback.print_exc()
        data_module_instance = None
        print("Falling back, SimpleShapesDataModule will not be used.")

    return data_module_instance 

# NOTE: create_synthetic_labels function removed as per user request
# Users must provide pre-computed synthetic_labels to all functions that need them 