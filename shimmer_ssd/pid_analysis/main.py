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
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np

# Add the root directory to the path for imports (like analyze_pid_new.py does)
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import from eval.py since analysis.py doesn't exist
from shimmer_ssd.pid_analysis.eval import (
    analyze_model,
    analyze_multiple_models,
    analyze_multiple_models_from_list,
    analyze_with_data_interface
)

from shimmer_ssd.pid_analysis.data_interface import create_synthetic_interface, create_data_interface

# Import utility functions
from shimmer_ssd.pid_analysis.utils import load_domain_modules, find_latest_model_checkpoints

# Try to import wandb, but make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  Warning: wandb not installed. Run 'pip install wandb' to enable experiment tracking.")

# Import cluster validation module for professional cluster validation
try:
    from shimmer_ssd.pid_analysis.cluster_visualization_validation import (
        run_cluster_validation_from_results
    )
    HAS_CLUSTER_VALIDATION = True
except ImportError as e:
    HAS_CLUSTER_VALIDATION = False
    print(f"⚠️  Warning: Cluster validation module not available: {e}")


# Set up device
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = global_device

#TODO give this function a new name, it's not a good name, and move to utils
def load_or_generate_synthetic_labels(
    model_path: str,
    generated_data: Dict[str, torch.Tensor],
    target_config: str = "gw_rep",
    num_clusters: int = 10,
    cluster_method: str = 'gmm'
) -> Tuple[torch.Tensor, Any]:
    """
    Load existing synthetic labels or generate them from model data.
    
    Args:
        model_path: Path to the model checkpoint
        generated_data: Dictionary containing model-generated data
        target_config: Key for the target representation to cluster
        num_clusters: Number of clusters for label generation
        cluster_method: Clustering method ('gmm' or 'kmeans')
        
    Returns:
        A tuple containing:
        - Tensor of synthetic labels
        - The fitted clustering model object
    """
    # Create a cache filename based on model path and clustering parameters
    model_dir = Path(model_path).parent
    model_name = Path(model_path).stem
    cache_filename = f"{model_name}_synthetic_labels_{target_config}_{num_clusters}_{cluster_method}.pt"
    cache_path = model_dir / cache_filename
    
    # Try to load existing labels
    if cache_path.exists():
        print(f"📁 Loading cached synthetic labels from: {cache_path}")
        try:
            # For now, we only load labels. If they exist, we must regenerate the model.
            synthetic_labels = torch.load(cache_path, map_location='cpu')
            print(f"✅ Loaded synthetic labels with shape: {synthetic_labels.shape}")
            # We still need to return a model, so we'll have to generate it.
        except Exception as e:
            print(f"⚠️  Failed to load cached labels: {e}")
            print("🔄 Generating new labels...")
    
    # Generate new labels
    if target_config not in generated_data:
        raise KeyError(f"Target '{target_config}' not found in generated data. Available keys: {list(generated_data.keys())}")
    
    target_data = generated_data[target_config]
    print(f"🧮 Generating synthetic labels by clustering {target_config} with shape {target_data.shape}")
    print(f"   Method: {cluster_method}, Clusters: {num_clusters}")
    
    # Import here to avoid dependency if not used
    from shimmer_ssd.pid_analysis.synthetic_data import create_synthetic_labels_with_model
    
    synthetic_labels, clustering_model = create_synthetic_labels_with_model(
        data=target_data,
        num_clusters=num_clusters,
        cluster_method=cluster_method
    )
    
    # Save to cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(synthetic_labels, cache_path)
        print(f"💾 Cached synthetic labels to: {cache_path}")
    except Exception as e:
        print(f"⚠️  Failed to cache labels: {e}")
    
    print(f"✅ Generated synthetic labels with shape: {synthetic_labels.shape}")
    return synthetic_labels, clustering_model

#TODO move somewhere else
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
    
    # Optional synthetic labels (auto-generated if not provided)
    model_parser.add_argument("--synthetic-labels", type=str, required=False, default=None,
                       help="Path to pre-computed synthetic labels (.pt file) - auto-generated if not provided")
    
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
    model_parser.add_argument("--model-type", type=str, default="complete_MLP",
                       choices=["complete_MLP", "pretrained_encoders"],
                       help="Type of model architecture to use: 'complete_MLP' (original) or 'pretrained_encoders' (with frozen encoders)")
    model_parser.add_argument("--discrim-epochs", type=int, default=40,
                       help="Number of epochs to train discriminators")
    model_parser.add_argument("--ce-epochs", type=int, default=10,
                       help="Number of epochs to train CE alignment")
    model_parser.add_argument("--discrim-hidden-dim", type=int, default=64,
                       help="Hidden dimension for discriminator networks")
    model_parser.add_argument("--joint-discrim-hidden-dim", type=int, default=None,
                       help="Hidden dimension for joint discriminator network (defaults to same as --discrim-hidden-dim if not specified)")
    model_parser.add_argument("--discrim-layers", type=int, default=5,
                       help="Number of layers in discriminator networks")
    model_parser.add_argument("--joint-discrim-layers", type=int, default=None,
                       help="Number of layers in joint discriminator network (defaults to same as --discrim-layers if not specified)")
    
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
    model_parser.add_argument("--cluster-method", type=str, default="gmm",
                       choices=["gmm", "kmeans"],
                       help="Clustering method for synthetic label generation")
    model_parser.add_argument("--run-critic-ce-direct", action="store_true",
                       help="Run critic_ce_alignment directly instead of separate training")
    
    # Visualization arguments (match shimmer_ssd/pid_analysis/main.py)
    model_parser.add_argument('--visualize-clusters', action='store_true',
                       help='Create cluster visualizations')
    model_parser.add_argument('--viz-samples-per-cluster', type=int, default=100,
                       help='Number of samples to visualize per cluster')
    model_parser.add_argument('--viz-grid-size', type=int, default=10,
                       help='Grid size for cluster visualization')
    model_parser.add_argument('--viz-max-clusters', type=int, default=20,
                       help='Maximum number of clusters to visualize')
    
    # Cluster validation arguments - professional validation of cluster meaningfulness
    model_parser.add_argument('--validate-clusters', action='store_true',
                       help='Validate cluster meaningfulness using validation images')
    model_parser.add_argument('--val-images-path', type=str,
                       default="/home/janerik/shimmer-ssd/simple_shapes_dataset/val",
                       help='Path to validation images directory')
    model_parser.add_argument('--val-dataset-path', type=str,
                       default="/home/janerik/shimmer-ssd/simple_shapes_dataset",
                       help='Path to validation dataset root directory')
    model_parser.add_argument('--val-n-samples', type=int, default=10000,
                       help='Number of validation samples to process for cluster validation')
    model_parser.add_argument('--val-max-clusters', type=int, default=20,
                       help='Maximum number of clusters to visualize in validation')
    model_parser.add_argument('--val-samples-per-cluster', type=int, default=100,
                       help='Number of validation samples to show per cluster')
    
    # New flag for cluster inspection only
    model_parser.add_argument("--only-inspect-clusters", action="store_true",
                          help="Bypass PID analysis to only load data, generate and visualize clusters.")
    
    # Synthetic data analysis
    synthetic_parser = subparsers.add_parser("synthetic", help="Analyze PID using synthetic Boolean data")
    
    # Synthetic data arguments
    synthetic_parser.add_argument("--functions", nargs="+", 
                                choices=['and', 'or', 'xor', 'nand', 'nor', 'xnor', 
                                        'id_a', 'id_b', 'not_a', 'not_b',
                                        'imp_a_b', 'imp_b_a', 'nimp_a_b', 'nimp_b_a',
                                        'const_0', 'const_1'],
                                default=['and', 'xor', 'id_a'],
                                help="Boolean functions to analyze (all 16 Boolean functions available)")
    synthetic_parser.add_argument("--source-a", type=str, default="input_a",
                                help="Source domain A name")
    synthetic_parser.add_argument("--source-b", type=str, default="input_b", 
                                help="Source domain B name")
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
    synthetic_parser.add_argument("--joint-discrim-hidden-dim", type=int, default=None,
                                help="Hidden dimension for joint discriminator network (defaults to same as --discrim-hidden-dim if not specified)")
    synthetic_parser.add_argument("--discrim-layers", type=int, default=3,
                                help="Number of layers for discriminator")
    synthetic_parser.add_argument("--joint-discrim-layers", type=int, default=None,
                                help="Number of layers for joint discriminator network (defaults to same as --discrim-layers if not specified)")
    synthetic_parser.add_argument("--model-type", type=str, default="complete_MLP", help=argparse.SUPPRESS)
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
        # Set default for joint_discrim_hidden_dim if not specified
        if args.joint_discrim_hidden_dim is None:
            args.joint_discrim_hidden_dim = args.discrim_hidden_dim
        if args.joint_discrim_layers is None:
            args.joint_discrim_layers = args.discrim_layers
            
        print("\n" + "="*60)
        print("🧪 PID ANALYSIS - SYNTHETIC BOOLEAN FUNCTIONS")
        print("="*60)
        print(f"🔧 Analyzing Functions: {', '.join(args.functions)}")
        print(f"📊 Samples: {args.n_samples:,}")
        print(f"🎯 Output Directory: {args.output_dir}")
        print("="*60 + "\n")
        
        # Create synthetic data interface with noise
        from shimmer_ssd.pid_analysis.data_interface import SyntheticDataProvider, GeneralizedDataInterface
        from shimmer_ssd.pid_analysis.synthetic_data import get_theoretical_pid_values
        
        # Set up source configuration for Boolean analysis
        source_config = {"domain_a": "input_a", "domain_b": "input_b"}
        
        print(f"🔧 Configuration:")
        print(f"   ├── Source: {source_config}")
        print(f"   └── Analyzing {len(args.functions)} functions sequentially\n")
        
        # Store results for all functions
        all_results = {}
        theoretical_values = get_theoretical_pid_values() if args.compare_theoretical else {}
        
        # Analyze each function sequentially
        for i, function_name in enumerate(args.functions, 1):
            print(f"🔍 [{i}/{len(args.functions)}] ANALYZING: {function_name.upper()}")
            print("─" * 40)
            
            try:
                # Create provider for this specific function
                provider = SyntheticDataProvider(
                    functions=[function_name],  # Only include the current function
                    seed=args.seed,
                    theoretical_pid=True
                )
                data_interface = GeneralizedDataInterface(provider)
                
                print(f"🚀 Starting analysis for {function_name}...")
                
                # Run analysis with data interface
                result = analyze_with_data_interface(
                    data_interface=data_interface,
                    source_config=source_config,
                    target_config=function_name,  # Use current function as target
                    output_dir=f"{args.output_dir}/{function_name}",
                    n_samples=args.n_samples,
                    batch_size=args.batch_size,
                    num_clusters=args.num_clusters,
                    discrim_epochs=args.discrim_epochs,
                    ce_epochs=args.ce_epochs,
                    discrim_hidden_dim=args.discrim_hidden_dim,
                    joint_discrim_hidden_dim=args.joint_discrim_hidden_dim,
                    discrim_layers=args.discrim_layers,
                    joint_discrim_layers=args.joint_discrim_layers,
                    # Pass provider-specific kwargs
                    model_type=args.model_type
                )
                
                # Parse PID results
                pid_results = result['pid_results']
                pid_values = None
                
                if isinstance(pid_results, tuple) and len(pid_results) > 0:
                    if hasattr(pid_results[0], 'tolist'):
                        pid_values = pid_results[0].tolist()
                    elif isinstance(pid_results[0], list):
                        pid_values = pid_results[0]
                elif isinstance(pid_results, list) and len(pid_results) > 0:
                    if isinstance(pid_results[0], list) and len(pid_results[0]) >= 4:
                        pid_values = pid_results[0]
                    elif len(pid_results) >= 4 and all(isinstance(x, (int, float)) for x in pid_results[:4]):
                        pid_values = pid_results
                elif hasattr(pid_results, 'tolist'):
                    pid_values = pid_results.tolist()
                
                if pid_values and len(pid_values) >= 4:
                    all_results[function_name] = {
                        'redundant': pid_values[0],
                        'unique_a': pid_values[1],
                        'unique_b': pid_values[2],
                        'synergistic': pid_values[3]
                    }
                    
                    print(f"✅ {function_name}: R={pid_values[0]:.3f}, UA={pid_values[1]:.3f}, UB={pid_values[2]:.3f}, S={pid_values[3]:.3f}")
                else:
                    print(f"❌ Failed to parse results for {function_name}")
                    all_results[function_name] = {
                        'redundant': float('nan'),
                        'unique_a': float('nan'),
                        'unique_b': float('nan'),
                        'synergistic': float('nan')
                    }
                
            except Exception as e:
                print(f"❌ Error analyzing {function_name}: {str(e)}")
                all_results[function_name] = {
                    'redundant': float('nan'),
                    'unique_a': float('nan'),
                    'unique_b': float('nan'),
                    'synergistic': float('nan')
                }
            
            print("─" * 40 + "\n")
        
        # Create comprehensive comparison table
        if args.compare_theoretical and theoretical_values:
            print("\n" + "="*80)
            print("📊 COMPREHENSIVE THEORETICAL COMPARISON")
            print("="*80)
            print("Function       │ Component   │ Measured │ Expected │ Difference │ Status")
            print("───────────────┼─────────────┼──────────┼──────────┼────────────┼────────")
            
            # Sort functions for consistent output
            sorted_functions = sorted([f for f in args.functions if f in all_results])
            
            for function_name in sorted_functions:
                if function_name in theoretical_values and function_name in all_results:
                    measured = all_results[function_name]
                    expected = theoretical_values[function_name]
                    
                    components = [
                        ('🔄 Redundant', 'redundant'),
                        ('🅰️  Unique A', 'unique_a'),
                        ('🅱️  Unique B', 'unique_b'),
                        ('⚡ Synergistic', 'synergistic')
                    ]
                    
                    for i, (comp_display, comp_key) in enumerate(components):
                        m_val = measured[comp_key]
                        e_val = expected[comp_key]
                        
                        if not (isinstance(m_val, float) and np.isnan(m_val)):
                            diff = abs(m_val - e_val)
                            status = "✅" if diff < 0.1 else "⚠️" if diff < 0.2 else "❌"
                            
                            if i == 0:  # First row for this function
                                func_display = f"{function_name:<14}"
                            else:
                                func_display = " " * 14
                            
                            print(f"{func_display} │ {comp_display:<11} │  {m_val:6.3f}  │  {e_val:6.3f}  │   {diff:7.3f}  │  {status}")
                        else:
                            if i == 0:  # First row for this function
                                func_display = f"{function_name:<14}"
                            else:
                                func_display = " " * 14
                            print(f"{func_display} │ {comp_display:<11} │    NaN   │  {e_val:6.3f}  │     ---    │  ❌")
                    
                    # Add separator between functions
                    if function_name != sorted_functions[-1]:
                        print("───────────────┼─────────────┼──────────┼──────────┼────────────┼────────")
        
        # Summary statistics
        if all_results:
            print(f"\n📈 SUMMARY STATISTICS")
            print("="*40)
            
            valid_results = {k: v for k, v in all_results.items() 
                           if not any(isinstance(val, float) and np.isnan(val) for val in v.values())}
            
            if valid_results:
                print(f"✅ Successfully analyzed: {len(valid_results)}/{len(args.functions)} functions")
                
                # Calculate average absolute errors if theoretical values available
                if args.compare_theoretical and theoretical_values:
                    total_errors = []
                    for func_name, measured in valid_results.items():
                        if func_name in theoretical_values:
                            expected = theoretical_values[func_name]
                            for comp in ['redundant', 'unique_a', 'unique_b', 'synergistic']:
                                total_errors.append(abs(measured[comp] - expected[comp]))
                    
                    if total_errors:
                        avg_error = np.mean(total_errors)
                        max_error = np.max(total_errors)
                        print(f"📊 Average absolute error: {avg_error:.3f}")
                        print(f"📊 Maximum absolute error: {max_error:.3f}")
                        print(f"🎯 Functions with error < 0.1: {sum(1 for e in total_errors if e < 0.1)}/{len(total_errors)} components")
            else:
                print(f"❌ No valid results obtained")
            
            print(f"📁 Individual results saved to: {args.output_dir}/[function_name]/")
        
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!") #TODO should move this out of main presumably
        print("="*60 + "\n")
        
    elif args.command == "model":
        print("\n" + "="*60)
        print("🤖 PID ANALYSIS - MODEL-BASED")
        print("="*60)
        
        if args.chunk_size is not None:
            import shimmer_ssd.pid_analysis.utils as pid_utils
            pid_utils.CHUNK_SIZE = args.chunk_size
        if args.memory_cleanup_interval is not None:
            import shimmer_ssd.pid_analysis.utils as pid_utils
            pid_utils.MEMORY_CLEANUP_INTERVAL = args.memory_cleanup_interval
        if args.aggressive_cleanup:
            import shimmer_ssd.pid_analysis.utils as pid_utils
            pid_utils.AGGRESSIVE_CLEANUP = True
        
        # Set device
        if args.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        
        print(f"🖥 Device: {device}")
        
        # Set GPU memory fraction if specified
        if torch.cuda.is_available() and args.gpu_memory_fraction is not None:
            if 0.0 < args.gpu_memory_fraction <= 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
                    print(f"🎛️  GPU Memory Fraction: {args.gpu_memory_fraction}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not set GPU memory fraction: {e}")
            else:
                print(f"⚠️  Warning: Invalid GPU memory fraction {args.gpu_memory_fraction}")
        
        # Parse domain configurations
        domain_configs = []
        for config_str in args.domain_configs:
            domain_configs.append(parse_json_or_file(config_str))
        
        # Parse source configuration
        source_config = parse_json_or_file(args.source_config)
        
        # Load domain modules
        print("🔧 Loading domain modules...")
        domain_modules = load_domain_modules(domain_configs)
        print(f"✅ Loaded domain modules: {', '.join(list(domain_modules.keys()))}")
        
        # Set up data module if requested
        data_module = None
        if args.use_dataset:
            print("🔄 Setting up data module for real dataset usage...")
            try:
                # Find root directory first (go up from pid_analysis to shimmer-ssd root)
                root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
                
                # Add the simple-shapes-dataset directory to path
                import sys
                sys.path.append(os.path.join(root_dir, "simple-shapes-dataset"))
                
                from simple_shapes_dataset.data_module import SimpleShapesDataModule
                from simple_shapes_dataset.domain import DomainDesc
                
                # Find dataset path
                dataset_path = os.path.join(root_dir, "full_shapes_dataset/simple_shapes_dataset")
                if not os.path.exists(dataset_path):
                    dataset_path = os.path.join(root_dir, "simple-shapes-dataset/sample_dataset")
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
                
                print(f"✅ Successfully set up data module!")
                
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                import traceback
                traceback.print_exc()
                print("🔄 Falling back to cached data files only...")
                data_module = None
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 Output directory: {args.output_dir}")
        print("="*60 + "\n")
        
        # Run analysis based on mode
        if args.single_model:
            
            print("🔍 SINGLE MODEL ANALYSIS")
            print("─"*30)
            print(f"📍 Model Path: {args.model_path}")
            print(f"🎯 Target Config: {args.target_config}")
            print(f"📊 Samples: {args.n_samples:,}")
            print("─"*30 + "\n")
            
            # Use the data_module that was already created at the top level if --use-dataset was specified
            # (No need for a second data module setup here)
            
            print("🚀 Starting single model analysis using data interface...")
            
            # Create ModelDataProvider using the data_interface.py
            from shimmer_ssd.pid_analysis.data_interface import ModelDataProvider, GeneralizedDataInterface
            
            provider = ModelDataProvider(
                model_path=args.model_path,
                domain_modules=domain_modules,
                data_module=data_module,
                dataset_split=args.dataset_split,
                use_gw_encoded=args.use_gw_encoded,
                device=device
            )
            
            data_interface = GeneralizedDataInterface(provider)
            
            # Use the data interface for analysis
            result = analyze_with_data_interface(
                data_interface=data_interface,
                source_config=source_config,
                target_config=args.target_config,
                output_dir=args.output_dir,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                num_clusters=args.num_clusters,
                discrim_epochs=args.discrim_epochs,
                ce_epochs=args.ce_epochs,
                discrim_hidden_dim=args.discrim_hidden_dim,
                joint_discrim_hidden_dim=args.joint_discrim_hidden_dim,
                discrim_layers=args.discrim_layers,
                joint_discrim_layers=args.joint_discrim_layers,
                use_wandb=args.wandb or args.validate_clusters,  # Auto-enable wandb for cluster validation
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                synthetic_labels=None,  # Will be auto-generated from clustering
                force_retrain_discriminators=getattr(args, 'force_retrain_discriminators', False),
                model_type=args.model_type,
                finish_wandb_run=not args.validate_clusters,  # Don't finish wandb run if cluster validation is requested
                # Additional provider-specific kwargs
                use_compile=args.use_compile,
                ce_test_mode=args.ce_test_mode,
                max_test_examples=args.max_test_examples,
                auto_find_lr=args.auto_find_lr,
                lr_finder_steps=args.lr_finder_steps,
                lr_start=args.lr_start,
                lr_end=args.lr_end,
                cluster_method=args.cluster_method,
                enable_extended_metrics=not args.disable_extended_metrics,
                run_critic_ce_direct=args.run_critic_ce_direct,
                visualize_clusters=args.visualize_clusters,
                viz_samples_per_cluster=args.viz_samples_per_cluster,
                viz_grid_size=args.viz_grid_size,
                viz_max_clusters=args.viz_max_clusters,
                # Additional kwargs needed for visualization
                model_path=args.model_path,
                domain_modules=domain_modules,
                # Cluster validation arguments - PASS TO EVAL.PY
                validate_clusters=args.validate_clusters,
                val_images_path=args.val_images_path,
                val_dataset_path=args.val_dataset_path,
                val_n_samples=args.val_n_samples,
                val_max_clusters=args.val_max_clusters,
                val_samples_per_cluster=args.val_samples_per_cluster
            )
            
            # Run cluster validation if requested
            if args.validate_clusters:
                if HAS_CLUSTER_VALIDATION:
                    print("\n🔬 STARTING CLUSTER VALIDATION")
                    print("="*60)
                    
                    # Set up validation configuration
                    validation_config = {
                        'val_images_path': args.val_images_path,
                        'dataset_path': args.val_dataset_path,
                        'n_samples': args.val_n_samples,
                        'max_clusters': args.val_max_clusters,
                        'samples_per_cluster': args.val_samples_per_cluster
                    }
                    
                    # Get the current wandb run (should exist from main analysis)
                    current_wandb_run = wandb.run if HAS_WANDB else None
                    
                    if current_wandb_run is None:
                        print("⚠️  Warning: No active wandb run found for cluster validation logging")
                        print("   Cluster validation will proceed but won't log to wandb")
                    else:
                        print(f"✅ Found active wandb run for cluster validation: {current_wandb_run.name}")
                        print(f"   Project: {current_wandb_run.project}, ID: {current_wandb_run.id}")
                    
                    # Run cluster validation using the clean module interface
                    validation_results = run_cluster_validation_from_results(
                        model_path=args.model_path,
                        domain_modules=domain_modules,
                        analysis_results=result,
                        wandb_run=current_wandb_run,
                        validation_config=validation_config
                    )
                    
                    # Add validation results to main results
                    result['cluster_validation'] = validation_results
                    
                    if validation_results.get('status') == 'completed':
                        print(f"\n🎉 CLUSTER VALIDATION COMPLETE!")
                        print(f"📊 Validated {validation_results.get('validation_samples', 0)} samples")
                        print(f"🎨 Created visualizations for {validation_results.get('visualized_clusters', 0)} clusters")
                    else:
                        print(f"\n⚠️  Cluster validation {validation_results.get('status', 'unknown')}")
                        if 'reason' in validation_results:
                            print(f"    Reason: {validation_results['reason']}")
                    
                    # Now finish the wandb run after cluster validation is complete
                    if current_wandb_run is not None and HAS_WANDB:
                        wandb.finish()
                        print("🏁 Finished wandb run after cluster validation")
                    
                    print("="*60)
                else:
                    print("\n⚠️  Cluster validation requested but module not available")
                    print("   Make sure cluster_visualization_validation.py is properly installed")
                    
                    # Still finish wandb run if it exists
                    if wandb.run is not None and HAS_WANDB:
                        wandb.finish()
                        print("🏁 Finished wandb run (cluster validation unavailable)")
                    
                    print("="*60)
            
            print(f"\n✅ SINGLE MODEL ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {args.output_dir}")
            print(f"📊 PID Values: {result['pid_results']}")
            if args.validate_clusters and 'cluster_validation' in result:
                validation_status = result['cluster_validation'].get('status', 'unknown')
                if validation_status == 'completed':
                    print(f"🔬 Cluster validation: {result['cluster_validation'].get('visualized_clusters', 0)} clusters validated")
                else:
                    print(f"🔬 Cluster validation: {validation_status}")
            print("="*60 + "\n")
        
        elif args.multiple_models:
            if not args.checkpoint_dir:
                parser.error("❌ --checkpoint-dir is required when using --multiple-models")
            
            print("🔍 MULTIPLE MODELS ANALYSIS")
            print("─"*35)
            print(f"📂 Checkpoint Directory: {args.checkpoint_dir}")
            print(f"🎯 Target Config: {args.target_config}")
            print(f"📊 Samples per Model: {args.n_samples:,}")
            print("─"*35 + "\n")
            
            print("🚀 Starting multiple models analysis...")
            
            results = analyze_multiple_models(
                checkpoint_dir=args.checkpoint_dir,
                domain_modules=domain_modules,
                output_dir=args.output_dir,
                source_config=source_config,
                target_config=args.target_config,
                synthetic_labels=None,  # Will be auto-generated for each model
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
                enable_extended_metrics_discrim=not args.disable_extended_metrics,
                cluster_method_discrim=args.cluster_method  # Pass cluster method
            )
            
            print(f"\n✅ MULTIPLE MODELS ANALYSIS COMPLETE!")
            print(f"📊 Analyzed {len(results)} models")
            print(f"📁 Results saved to: {args.output_dir}")
            print("="*60 + "\n")
        
        elif args.model_list:
            if not args.checkpoint_list:
                parser.error("❌ --checkpoint-list is required when using --model-list")
            
            # Create matching domain configs for each checkpoint
            domain_configs_list = domain_configs * len(args.checkpoint_list)
            
            print("🔍 MODEL LIST ANALYSIS")
            print("─"*30)
            print(f"📝 Number of Models: {len(args.checkpoint_list)}")
            print(f"🎯 Target Config: {args.target_config}")
            print(f"📊 Samples per Model: {args.n_samples:,}")
            print("─"*30 + "\n")
            
            print("🚀 Starting model list analysis...")
            
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
            
            print(f"\n✅ MODEL LIST ANALYSIS COMPLETE!")
            print(f"📊 Analyzed {len(results)} models")
            print(f"📁 Results saved to: {args.output_dir}")
            print("="*60 + "\n")
        
        elif args.find_latest:
            print("🔍 LATEST CHECKPOINT ANALYSIS")
            print("─"*35)
            print(f"📂 Base Directory: {args.base_dir}")
            print("─"*35 + "\n")
            
            print("🔎 Finding latest checkpoint...")
            checkpoint_list = find_latest_model_checkpoints(args.base_dir)
            
            if not checkpoint_list:
                print("❌ No latest checkpoint found.")
                print("="*60 + "\n")
                return
            
            checkpoint_path = checkpoint_list[0]
            print(f"✅ Found latest checkpoint: {checkpoint_path}")
            print(f"🎯 Target Config: {args.target_config}")
            print(f"📊 Samples: {args.n_samples:,}\n")
            
            print("🚀 Starting analysis...")
            
            result = analyze_model(
                model_path=checkpoint_path,
                domain_modules=domain_modules,
                output_dir=args.output_dir,
                source_config=source_config,
                target_config=args.target_config,
                synthetic_labels=None,  # Will be auto-generated
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
                enable_extended_metrics_discrim=not args.disable_extended_metrics,
                cluster_method_discrim=args.cluster_method  # Pass cluster method
            )
            
            print(f"\n✅ LATEST CHECKPOINT ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {args.output_dir}")
            print(f"📊 PID Values: {result['pid_results']}")
            print("="*60 + "\n")
        else:
            parser.print_help()


if __name__ == "__main__":
    main() 
