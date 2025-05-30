"""
Main entry point for PID analysis.

This module provides a command-line interface for running PID analysis
on trained GW-module models.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from eval.py since analysis.py doesn't exist
from pid_analysis.eval import (
    analyze_model,
    analyze_multiple_models,
    analyze_multiple_models_from_list,
    analyze_with_data_interface
)
from pid_analysis.utils import (
    load_domain_modules,
    find_latest_model_checkpoints,
    validate_source_target_config,
    USE_AMP, CHUNK_SIZE, MEMORY_CLEANUP_INTERVAL, AGGRESSIVE_CLEANUP
)
from pid_analysis.data_interface import create_synthetic_interface, create_data_interface

# Try to import wandb, but make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable experiment tracking.")


def load_synthetic_labels(labels_path: str) -> torch.Tensor:
    """
    Load pre-computed synthetic labels from a file.
    
    Args:
        labels_path: Path to the .pt file containing synthetic labels
        
    Returns:
        Tensor containing the synthetic labels
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Synthetic labels file not found: {labels_path}")
    
    labels = torch.load(labels_path, map_location='cpu')
    print(f"Loaded synthetic labels from {labels_path}, shape: {labels.shape}")
    return labels


def parse_json_or_file(value: str) -> Dict[str, Any]:
    """
    Parse a JSON string or load from a JSON file.
    
    Args:
        value: JSON string or path to JSON file
        
    Returns:
        Parsed dictionary
    """
    # Try to parse as JSON string first
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # If that fails, try to load as a file
        if os.path.exists(value):
            with open(value, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Could not parse as JSON string or find file: {value}")


def main():
    """Parse arguments and run PID analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze information flow in GW-module models using PID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Original model-based analysis (keep existing functionality)
    model_parser = subparsers.add_parser("model", help="Analyze PID using trained models")
    
    # Analysis mode
    mode_group = model_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single-model", action="store_true", 
                           help="Analyze a single model checkpoint")
    mode_group.add_argument("--multiple-models", action="store_true", 
                           help="Analyze multiple models from a directory")
    mode_group.add_argument("--model-list", action="store_true", 
                           help="Analyze models from a list of checkpoints")
    mode_group.add_argument("--find-latest", action="store_true", 
                           help="Find and analyze the latest checkpoint")
    
    # Model paths and directories
    model_parser.add_argument("--model-path", type=str, 
                       help="Path to model checkpoint (for --single-model)")
    model_parser.add_argument("--checkpoint-dir", type=str, 
                       help="Directory containing checkpoints (for --multiple-models)")
    model_parser.add_argument("--checkpoint-list", type=str, nargs="+", 
                       help="List of checkpoint paths (for --model-list)")
    model_parser.add_argument("--base-dir", type=str, default="checkpoints/fusion",
                       help="Base directory for finding latest checkpoint (for --find-latest)")
    
    # Required synthetic labels
    model_parser.add_argument("--synthetic-labels", type=str, required=True,
                       help="Path to pre-computed synthetic labels (.pt file)")
    
    # Add missing num_clusters parameter
    model_parser.add_argument("--num-clusters", type=int, default=10,
                       help="Number of clusters for synthetic labels")
    
    # Output configuration
    model_parser.add_argument("--output-dir", type=str, default="pid_results",
                       help="Directory to save results")
    
    # Domain module configuration
    model_parser.add_argument("--domain-configs", type=str, nargs="+", required=True,
                       help="Domain configuration JSON strings or paths to JSON files")
    
    # PID analysis configuration
    model_parser.add_argument("--source-config", type=str, required=True,
                       help="Source configuration JSON string or path to JSON file")
    model_parser.add_argument("--target-config", type=str, default="gw_latent",
                       help="Target representation for PID analysis")
    
    # Data generation parameters
    model_parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of samples to generate for analysis")
    model_parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for data generation")
    
    # Training parameters
    model_parser.add_argument("--discrim-epochs", type=int, default=40,
                       help="Number of epochs to train discriminators")
    model_parser.add_argument("--ce-epochs", type=int, default=10,
                       help="Number of epochs to train CE alignment")
    model_parser.add_argument("--discrim-hidden-dim", type=int, default=64,
                       help="Hidden dimension for discriminator networks")
    model_parser.add_argument("--discrim-layers", type=int, default=5,
                       help="Number of layers in discriminator networks")
    
    # Performance and optimization
    model_parser.add_argument("--device", type=str, default=None,
                       help="Device to use (auto-detected if not specified)")
    model_parser.add_argument("--use-compile", action="store_true",
                       help="Use torch.compile for model optimization")
    model_parser.add_argument("--use-amp", action="store_true",
                       help="Enable Automatic Mixed Precision")
    model_parser.add_argument("--chunk-size", type=int, default=None,
                       help="Chunk size for operations")
    model_parser.add_argument("--memory-cleanup-interval", type=int, default=None,
                       help="Interval for memory cleanup")
    model_parser.add_argument("--aggressive-cleanup", action="store_true",
                       help="Enable aggressive memory cleanup")
    model_parser.add_argument("--gpu-memory-fraction", type=float, default=None,
                       help="Limit GPU memory usage (0.0 to 1.0)")
    
    # Learning rate finder parameters
    model_parser.add_argument("--auto-find-lr", action="store_true",
                       help="Enable LR finder for CE alignment training")
    model_parser.add_argument("--lr-finder-steps", type=int, default=200,
                       help="Number of iterations for LR finder")
    model_parser.add_argument("--lr-start", type=float, default=1e-7,
                       help="Start LR for finder")
    model_parser.add_argument("--lr-end", type=float, default=1.0,
                       help="End LR for finder")
    
    # CE test mode parameters
    model_parser.add_argument("--ce-test-mode", action="store_true",
                       help="Run CE training in test mode with limited examples")
    model_parser.add_argument("--max-test-examples", type=int, default=3000,
                       help="Max examples for CE test mode")
    
    # Dataset parameters
    model_parser.add_argument("--use-dataset", action="store_true",
                       help="Use real data from a dataset module")
    model_parser.add_argument("--dataset-path", type=str,
                       help="Path to the dataset")
    model_parser.add_argument("--dataset-split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to use")
    model_parser.add_argument("--use-gw-encoded", action="store_true",
                       help="Use GW-encoded vectors instead of raw latents")
    
    # W&B configuration
    model_parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    model_parser.add_argument("--wandb-project", type=str, default="pid-analysis",
                       help="W&B project name")
    model_parser.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity name")
    
    # Additional parameters
    model_parser.add_argument("--disable-extended-metrics", action="store_true", default=False,
                       help="Disable extended classifier metrics (enabled by default)")
    model_parser.add_argument("--use-domain-for-labels", type=str, default="both",
                       choices=["both", "v_latents", "t"],
                       help="Domain to use for label generation (for --model-list)")
    
    # Synthetic data analysis
    synthetic_parser = subparsers.add_parser("synthetic", help="Analyze PID using synthetic Boolean data")
    
    # Synthetic data arguments
    synthetic_parser.add_argument("--functions", nargs="+", 
                                choices=['and', 'xor', 'id_a', 'id_b', 'or', 'nand', 'nor'],
                                default=['and', 'xor', 'id_a'],
                                help="Boolean functions to analyze")
    synthetic_parser.add_argument("--source-a", type=str, default="input_a",
                                help="Source domain A name")
    synthetic_parser.add_argument("--source-b", type=str, default="input_b", 
                                help="Source domain B name")
    synthetic_parser.add_argument("--target", type=str, required=True,
                                help="Target function name")
    synthetic_parser.add_argument("--output-dir", type=str, required=True,
                                help="Output directory for results")
    synthetic_parser.add_argument("--n-samples", type=int, default=10000,
                                help="Number of samples to generate")
    synthetic_parser.add_argument("--seed", type=int, default=42,
                                help="Random seed for reproducibility")
    synthetic_parser.add_argument("--num-clusters", type=int, default=2,
                                help="Number of clusters (should be 2 for Boolean)")
    synthetic_parser.add_argument("--batch-size", type=int, default=128,
                                help="Batch size for training")
    synthetic_parser.add_argument("--discrim-epochs", type=int, default=40,
                                help="Number of epochs for discriminator training")
    synthetic_parser.add_argument("--ce-epochs", type=int, default=10,
                                help="Number of epochs for CE alignment training")
    synthetic_parser.add_argument("--discrim-hidden-dim", type=int, default=64,
                                help="Hidden dimension for discriminator")
    synthetic_parser.add_argument("--discrim-layers", type=int, default=5,
                                help="Number of layers for discriminator")
    synthetic_parser.add_argument("--wandb", action="store_true",
                                help="Enable Weights & Biases logging")
    synthetic_parser.add_argument("--wandb-project", type=str, default="pid-synthetic",
                                help="W&B project name")
    synthetic_parser.add_argument("--wandb-entity", type=str,
                                help="W&B entity name")
    synthetic_parser.add_argument("--compare-theoretical", action="store_true",
                                help="Compare results with theoretical PID values")
    
    # File-based data analysis
    file_parser = subparsers.add_parser("file", help="Analyze PID using pre-saved data files")
    
    file_parser.add_argument("--data-dir", type=str, required=True,
                           help="Directory containing data files")
    file_parser.add_argument("--domain-names", nargs="+", required=True,
                           help="List of domain names to load")
    file_parser.add_argument("--file-pattern", type=str, default="{domain}.pt",
                           help="File pattern (use {domain} placeholder)")
    file_parser.add_argument("--source-config", type=parse_json_or_file, required=True,
                           help="Source configuration as JSON string or file path")
    file_parser.add_argument("--target-config", type=str, required=True,
                           help="Target domain name")
    file_parser.add_argument("--output-dir", type=str, required=True,
                           help="Output directory for results")
    file_parser.add_argument("--n-samples", type=int, default=10000,
                           help="Number of samples to use")
    file_parser.add_argument("--num-clusters", type=int, default=10,
                           help="Number of clusters for synthetic labels")
    # ... add other common arguments for file parser ...
    
    args = parser.parse_args()
    
    if args.command == "synthetic":
        print(f"🧪 Analyzing synthetic Boolean functions: {args.functions}")
        print(f"📊 Target function: {args.target}")
        
        # Create synthetic data interface with noise
        from pid_analysis.data_interface import SyntheticDataProvider, GeneralizedDataInterface
        provider = SyntheticDataProvider(
            functions=args.functions,
            seed=args.seed,
            theoretical_pid=True
        )
        data_interface = GeneralizedDataInterface(provider)
        
        # Set up source and target configurations for Boolean analysis
        source_config = {"domain_a": "input_a", "domain_b": "input_b"}
        target_config = args.target
        
        print(f"🔧 Source config: {source_config}")
        print(f"🎯 Target config: {target_config}")
        
        # Run analysis with data interface
        result = analyze_with_data_interface(
            data_interface=data_interface,
            source_config=source_config,
            target_config=target_config,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            num_clusters=args.num_clusters,
            discrim_epochs=args.discrim_epochs,
            ce_epochs=args.ce_epochs,
            discrim_hidden_dim=args.discrim_hidden_dim,
            discrim_layers=args.discrim_layers,
            # Pass provider-specific kwargs
            add_noise=True,
            noise_std=0.1
        )
        
        # Compare with theoretical values if requested
        if args.compare_theoretical:
            from pid_analysis.synthetic_data import get_theoretical_pid_values
            theoretical = get_theoretical_pid_values()
            
            if args.target in theoretical:
                print(f"\n📊 Comparison for {args.target}:")
                print("=" * 50)
                
                # The PID results are returned as a list [redundant, unique_1, unique_2, synergistic]
                pid_results = result['pid_results']
                
                # Handle different return formats from critic_ce_alignment
                pid_values = None
                if isinstance(pid_results, tuple) and len(pid_results) > 0:
                    # If it's a tuple, first element should be the tensor with PID values
                    if hasattr(pid_results[0], 'tolist'):  # It's a tensor
                        pid_values = pid_results[0].tolist()
                    elif isinstance(pid_results[0], list):
                        pid_values = pid_results[0]
                elif isinstance(pid_results, list) and len(pid_results) > 0:
                    # Handle nested list structure - extract the actual PID values
                    if isinstance(pid_results[0], list) and len(pid_results[0]) >= 4:
                        pid_values = pid_results[0]
                    # If it's directly a list of numbers, use that
                    elif len(pid_results) >= 4 and all(isinstance(x, (int, float)) for x in pid_results[:4]):
                        pid_values = pid_results
                elif hasattr(pid_results, 'tolist'):  # Direct tensor
                    pid_values = pid_results.tolist()
                
                if pid_values and len(pid_values) >= 4:
                    measured_redundant = pid_values[0]
                    measured_unique_a = pid_values[1] 
                    measured_unique_b = pid_values[2]
                    measured_synergistic = pid_values[3]
                    
                    theo_values = theoretical[args.target]
                    
                    comparisons = [
                        ('redundant', measured_redundant, theo_values['redundant']),
                        ('unique_a', measured_unique_a, theo_values['unique_a']),
                        ('unique_b', measured_unique_b, theo_values['unique_b']),
                        ('synergistic', measured_synergistic, theo_values['synergistic'])
                    ]
                    
                    for comp_name, measured, expected in comparisons:
                        diff = abs(measured - expected)
                        print(f"{comp_name:12}: measured={measured:.3f}, expected={expected:.3f}, diff={diff:.3f}")
                else:
                    print("Could not parse PID results for comparison")
                    print(f"Debug: pid_results type={type(pid_results)}")
                    if hasattr(pid_results, '__len__') and len(pid_results) > 0:
                        print(f"First element type: {type(pid_results[0])}")
                        if hasattr(pid_results[0], 'shape'):
                            print(f"First element shape: {pid_results[0].shape}")
                        elif hasattr(pid_results[0], '__len__'):
                            print(f"First element length: {len(pid_results[0])}")
            else:
                print(f"No theoretical values available for {args.target}")
        
        print(f"✅ Synthetic analysis complete! Results saved to: {args.output_dir}")
        
    elif args.command == "file":
        # Handle file-based data analysis
        print(f"📁 Analyzing data from files in: {args.data_dir}")
        
        # Create file data interface
        data_interface = create_data_interface(
            'file',
            data_dir=args.data_dir,
            domain_names=args.domain_names,
            file_pattern=args.file_pattern
        )
        
        # Run analysis
        result = analyze_with_data_interface(
            data_interface=data_interface,
            source_config=args.source_config,
            target_config=args.target_config,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            num_clusters=args.num_clusters,
            # ... other arguments ...
        )
        
        print(f"✅ File-based analysis complete! Results saved to: {args.output_dir}")
        
    elif args.command == "model":
        # Set global configuration variables
        if args.use_amp:
            import pid_analysis.utils as pid_utils
            pid_utils.USE_AMP = True
        if args.chunk_size is not None:
            import pid_analysis.utils as pid_utils
            pid_utils.CHUNK_SIZE = args.chunk_size
        if args.memory_cleanup_interval is not None:
            import pid_analysis.utils as pid_utils
            pid_utils.MEMORY_CLEANUP_INTERVAL = args.memory_cleanup_interval
        if args.aggressive_cleanup:
            import pid_analysis.utils as pid_utils
            pid_utils.AGGRESSIVE_CLEANUP = True
        
        # Set device
        if args.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        
        print(f"Using device: {device}")
        
        # Set GPU memory fraction if specified
        if torch.cuda.is_available() and args.gpu_memory_fraction is not None:
            if 0.0 < args.gpu_memory_fraction <= 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
                    print(f"Set GPU memory fraction to {args.gpu_memory_fraction}")
                except Exception as e:
                    print(f"Warning: Could not set GPU memory fraction: {e}")
            else:
                print(f"Warning: Invalid GPU memory fraction {args.gpu_memory_fraction}")
        
        # Load synthetic labels
        synthetic_labels = load_synthetic_labels(args.synthetic_labels)
        
        # Parse domain configurations
        domain_configs = []
        for config_str in args.domain_configs:
            domain_configs.append(parse_json_or_file(config_str))
        
        # Parse source configuration
        source_config = parse_json_or_file(args.source_config)
        
        # Load domain modules
        print("Loading domain modules...")
        domain_modules = load_domain_modules(domain_configs, eval_mode=True, device=device)
        print(f"Loaded domain modules: {list(domain_modules.keys())}")
        
        # Set up data module if requested
        data_module = None
        if args.use_dataset and args.dataset_path:
            print(f"Setting up dataset from {args.dataset_path}")
            # This would need to be implemented based on the specific dataset format
            # For now, we'll leave it as None
            print("Warning: Dataset loading not implemented yet. Using model-generated data.")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run analysis based on mode
        if args.single_model:
            if not args.model_path:
                parser.error("--model-path is required when using --single-model")
            
            print(f"Analyzing single model: {args.model_path}")
            result = analyze_model(
                model_path=args.model_path,
                domain_modules=domain_modules,
                output_dir=args.output_dir,
                source_config=source_config,
                target_config=args.target_config,
                synthetic_labels=synthetic_labels,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                num_clusters=args.num_clusters,
                discrim_epochs=args.discrim_epochs,
                ce_epochs=args.ce_epochs,
                discrim_hidden_dim=args.discrim_hidden_dim,
                discrim_layers=args.discrim_layers,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                data_module=data_module,
                dataset_split=args.dataset_split,
                use_gw_encoded=args.use_gw_encoded,
                use_compile_torch=args.use_compile,
                ce_test_mode_run=args.ce_test_mode,
                max_test_examples_run=args.max_test_examples,
                auto_find_lr_run=args.auto_find_lr,
                lr_finder_steps_run=args.lr_finder_steps,
                lr_start_run=args.lr_start,
                lr_end_run=args.lr_end,
                enable_extended_metrics_discrim=not args.disable_extended_metrics
            )
            
            print(f"Analysis complete. Results saved to {args.output_dir}")
            print(f"PID values: {result['pid_results']}")
        
        elif args.multiple_models:
            if not args.checkpoint_dir:
                parser.error("--checkpoint-dir is required when using --multiple-models")
            
            print(f"Analyzing multiple models from: {args.checkpoint_dir}")
            results = analyze_multiple_models(
                checkpoint_dir=args.checkpoint_dir,
                domain_modules=domain_modules,
                output_dir=args.output_dir,
                source_config=source_config,
                target_config=args.target_config,
                synthetic_labels=synthetic_labels,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                num_clusters=args.num_clusters,
                discrim_epochs=args.discrim_epochs,
                ce_epochs=args.ce_epochs,
                discrim_hidden_dim=args.discrim_hidden_dim,
                discrim_layers=args.discrim_layers,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                data_module=data_module,
                dataset_split=args.dataset_split,
                use_gw_encoded=args.use_gw_encoded,
                use_compile_torch=args.use_compile,
                ce_test_mode_run=args.ce_test_mode,
                max_test_examples_run=args.max_test_examples,
                auto_find_lr_run=args.auto_find_lr,
                lr_finder_steps_run=args.lr_finder_steps,
                lr_start_run=args.lr_start,
                lr_end_run=args.lr_end,
                enable_extended_metrics_discrim=not args.disable_extended_metrics
            )
            
            print(f"Analysis complete. Analyzed {len(results)} models.")
            print(f"Results saved to {args.output_dir}")
        
        elif args.model_list:
            if not args.checkpoint_list:
                parser.error("--checkpoint-list is required when using --model-list")
            
            # Create matching domain configs for each checkpoint
            domain_configs_list = domain_configs * len(args.checkpoint_list)
            
            print(f"Analyzing {len(args.checkpoint_list)} models from list")
            results = analyze_multiple_models_from_list(
                checkpoint_list=args.checkpoint_list,
                domain_configs=domain_configs_list,
                output_dir=args.output_dir,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                num_clusters=args.num_clusters,
                discrim_epochs=args.discrim_epochs,
                ce_epochs=args.ce_epochs,
                device=device,
                use_domain_for_labels=args.use_domain_for_labels,
                discrim_hidden_dim=args.discrim_hidden_dim,
                discrim_layers=args.discrim_layers,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                data_module=data_module,
                dataset_split=args.dataset_split,
                use_gw_encoded=args.use_gw_encoded,
                use_compile=args.use_compile
            )
            
            print(f"Analysis complete. Analyzed {len(results)} models.")
            print(f"Results saved to {args.output_dir}")
        
        elif args.find_latest:
            print(f"Finding latest checkpoint in: {args.base_dir}")
            checkpoint_list = find_latest_model_checkpoints(args.base_dir)
            
            if not checkpoint_list:
                print("No latest checkpoint found.")
                return
            
            checkpoint_path = checkpoint_list[0]
            print(f"Found latest checkpoint: {checkpoint_path}")
            
            result = analyze_model(
                model_path=checkpoint_path,
                domain_modules=domain_modules,
                output_dir=args.output_dir,
                source_config=source_config,
                target_config=args.target_config,
                synthetic_labels=synthetic_labels,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                num_clusters=args.num_clusters,
                discrim_epochs=args.discrim_epochs,
                ce_epochs=args.ce_epochs,
                discrim_hidden_dim=args.discrim_hidden_dim,
                discrim_layers=args.discrim_layers,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                data_module=data_module,
                dataset_split=args.dataset_split,
                use_gw_encoded=args.use_gw_encoded,
                use_compile_torch=args.use_compile,
                ce_test_mode_run=args.ce_test_mode,
                max_test_examples_run=args.max_test_examples,
                auto_find_lr_run=args.auto_find_lr,
                lr_finder_steps_run=args.lr_finder_steps,
                lr_start_run=args.lr_start,
                lr_end_run=args.lr_end,
                enable_extended_metrics_discrim=not args.disable_extended_metrics
            )
            
            print(f"Analysis complete. Results saved to {args.output_dir}")
            print(f"PID values: {result['pid_results']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 