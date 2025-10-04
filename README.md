# Spectral Assessment for Graph Neural Networks

This repository contains a comprehensive framework for analyzing the spectral properties of Graph Neural Network (GNN) model weights using Random Matrix Theory (RMT) and WeightWatcher analysis. The project is built on top of GraphGPS and supports multiple GNN architectures including GPS, GatedGCN, GINE, Graphormer, and SAN models.

## Overview

The project focuses on understanding the learned representations and weight distributions in GNNs through spectral analysis. It provides tools to:

- Analyze weight matrices using Random Matrix Theory principles (weight_analysis.py)
- Compare attention mechanisms to theoretical distributions (Wigner, Marcenko-Pastur, Wishart) (MP_analysis.py)
- Compare spectral properties across different training stages
- Visualize weight distributions and their evolution during training

## Project Structure

```
spectral_assess/
├── main.py                    # Main training script
├── weight_analysis.py         # matrix weight analysis script
├── setup.py                   # Package setup configuration
├── README.md                  # This file
├── analyze_weights/
│   └── spectral_analysis.py   # Core spectral analysis functions
├── configs/                   # Model configuration files
│   ├── GPS/                   # Graph Transformer (GPS) configs
│   ├── GatedGCN/             # GatedGCN model configs
│   ├── GINE/                 # Graph Isomorphism Network configs
│   ├── Graphormer/           # Graphormer configs
│   └── SAN/                  # Spectral Attention Network configs
├── graphgps/                 # GraphGPS framework
├── run/                      # Experiment execution scripts
├── table_fig_creation/       # Result visualization tools
├── tests/                    # Test configurations
└── unittests/               # Unit tests
```



## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- WeightWatcher
- NumPy, SciPy, Matplotlib
- NetworkX (for graph analysis)

## Usage

### 1. Basic Training with Spectral Monitoring
```bash
# Train a GPS model on CIFAR-10 with default settings
python main.py --cfg configs/GPS/cifar10-GPS.yaml --repeat 1

# Train with custom output directory
python main.py --cfg configs/GPS/cifar10-GPS.yaml --opts out_dir results/my_experiment
```

### 2. WeightWatcher Analysis
```bash
# Analyze trained model weights
python weight_analysis.py
```

This script will:
- Load a trained model checkpoint
- Perform RMT analysis
- Generate spectral quality metrics
- Save results to CSV format

### 3. Custom Spectral Analysis
```python
from analyze_weights.spectral_analysis import *

# Load checkpoint
ckpt_path = "path/to/your/checkpoint.ckpt"
state_dict = load_checkpoint(ckpt_path)['model_state']

# Analyze attention weights
analyze_attention_weights(state_dict)

# Compare two training stages
compare_checkpoints("checkpoint_early.ckpt", "checkpoint_final.ckpt")

# Per-head attention analysis
per_head_weights(state_dict, layer=0)
```

## Results and Visualization

The MP_analysis function generates several types of visualizations:

1. **Spectral Density Plots**: Histograms of eigenvalues/singular values with theoretical overlays
2. **Attention Heatmaps**: Visualization of attention weight matrices
3. **Training Evolution**: Spectral property changes across checkpoints
4. **Per-Head Comparisons**: Multi-head attention analysis
5. **Quality Metrics**: WeightWatcher-derived model assessments


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [GraphGPS](https://github.com/rampasek/GraphGPS)
- Uses [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) for quality analysis
- Inspired by Random Matrix Theory applications in deep learning

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: sydney.dolan@[institution].edu

---

*Last updated: October 2025*
