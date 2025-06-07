# PID_GW_new

This repository contains the implementation of Partial Information Decomposition (PID) analysis for Global Workspace models.

## Overview

This codebase provides tools and functions for analyzing partial information decomposition in neural networks, particularly focusing on Global Workspace architectures. The implementation includes data processing, model training, evaluation, and visualization components.

## Main Components

### Core Files

- **`main.py`** - Main entry point for running PID analysis
- **`models.py`** - Neural network model definitions
- **`train.py`** - Training routines and procedures
- **`eval.py`** - Evaluation and analysis functions
- **`utils.py`** - Utility functions and helper methods

### Data Processing

- **`data.py`** - Data loading and preprocessing
- **`synthetic_data.py`** - Synthetic data generation
- **`data_interface.py`** - Data interface and handling

### Analysis & Visualization

- **`cluster_visualization.py`** - Clustering and visualization tools
- **`sinkhorn.py`** - Sinkhorn algorithm implementation

### Configuration

- **`domain_t_config.json`** - Temporal domain configuration
- **`domain_v_config.json`** - Visual domain configuration

## Installation

```bash
# Clone the repository
git clone https://github.com/JanBellingrath/PID_GW_new.git
cd PID_GW_new

# Install dependencies (requirements.txt to be added)
pip install -r requirements.txt
```

## Usage

```python
# Basic usage example
from main import main
from models import your_model
from data import load_data

# Run PID analysis
results = main()
```

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Other dependencies (see requirements.txt)

## Author

Jan Bellingrath  
Computational Neuroscience x Artificial Intelligence  
Cerco (CNRS), Toulouse

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
``` 