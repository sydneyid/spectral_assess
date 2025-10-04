# Spectral Assessment for Graph Neural Networks

This repository contains a comprehensive framework for analyzing the spectral properties of Graph Neural Network (GNN) model weights using Random Matrix Theory (RMT) and WeightWatcher analysis. The project is built on top of GraphGPS and supports multiple GNN architectures including GPS, GatedGCN, GINE, Graphormer, and SAN models.

## Overview

The project focuses on understanding the learned representations and weight distributions in GNNs through spectral analysis. It provides tools to:

- Analyze weight matrices using Random Matrix Theory principles
- Compare attention mechanisms to theoretical distributions (Wigner, Marcenko-Pastur, Wishart)
- Perform WeightWatcher analysis for model quality assessment
- Compare spectral properties across different training stages
- Visualize weight distributions and their evolution during training

## Project Structure

```
spectral_assess/
├── main.py                    # Main training script
├── weight_analysis.py         # WeightWatcher analysis script
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

## Key Features

### 1. Random Matrix Theory Analysis
- **Eigenvalue Spectrum Analysis**: Compare model weights to theoretical predictions from RMT
- **Wigner Semicircle Law**: Analyze square weight matrices against Wigner's predictions
- **Marcenko-Pastur Law**: Compare singular value distributions of rectangular matrices
- **Wishart Distribution**: Analyze covariance-like structures in attention weights

### 2. Attention Mechanism Analysis
- **Per-Head Analysis**: Individual spectral analysis of multi-head attention components
- **Adjacency Matrix Interpretation**: Treat attention weights as graph adjacency matrices
- **Temporal Evolution**: Track how attention patterns change during training

### 3. WeightWatcher Integration
- **Model Quality Metrics**: Automated assessment using WeightWatcher framework
- **Heavy-Tailed Analysis**: Detection of heavy-tailed eigenvalue distributions
- **Spectral Norm Analysis**: Track spectral properties across layers

### 4. Multi-Architecture Support
Supports analysis of various GNN architectures:
- **GPS (Graph Transformer)**: Hybrid MPNN + Transformer architecture
- **GatedGCN**: Gated Graph Convolutional Networks
- **GINE**: Graph Isomorphism Network with Edge features
- **Graphormer**: Pure Transformer for graphs
- **SAN**: Spectral Attention Networks

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- WeightWatcher
- NumPy, SciPy, Matplotlib
- NetworkX (for graph analysis)

### Setup
```bash
# Clone the repository
git clone https://github.com/sydneyid/spectral_assess.git
cd spectral_assess

# Install dependencies
pip install torch torch-geometric
pip install weightwatcher
pip install scipy matplotlib networkx
pip install -r requirements.txt  # if available

# Install the package
pip install -e .
```

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
- Perform WeightWatcher analysis
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

### 4. Configuration Examples

The `configs/` directory contains pre-configured experiments for different datasets and architectures:

```yaml
# Example GPS configuration (configs/GPS/cifar10-GPS.yaml)
model:
  type: GPSModel
gt:
  layer_type: CustomGatedGCN+Transformer
  layers: 3
  n_heads: 4
  dim_hidden: 52
gnn:
  layers_post_mp: 4
  dim_inner: 52
```

## Key Analysis Functions

### Spectral Analysis (`analyze_weights/spectral_analysis.py`)

```python
# Compare weight distributions to theoretical predictions
compare_to_gaussian_ensembles(matrix, name)

# Analyze MPNN weights with RMT
analyze_mpnn_weights_rmt(state_dict)

# Compare attention weights across training
compare_attention_maps(state_dict_early, state_dict_final)

# Per-head attention analysis
analyze_per_head_attention(attn_weights, attn_outputs, targets)
```

### Core Analysis Types

1. **Eigenvalue Distribution Analysis**
   - Comparison with Wigner semicircle law
   - Gaussian ensemble comparisons (GOE, GUE, GSE)

2. **Singular Value Analysis** 
   - Marcenko-Pastur law fitting
   - Wishart distribution comparisons

3. **Attention Pattern Analysis**
   - Graph connectivity metrics
   - Spectral clustering properties
   - Evolution during training

## Theoretical Background

This project implements several key concepts from Random Matrix Theory:

### Wigner Semicircle Law
For large random symmetric matrices, eigenvalues follow a semicircle distribution:
$$\rho(x) = \frac{2}{\pi R^2}\sqrt{R^2 - x^2}$$

### Marcenko-Pastur Law
For rectangular random matrices, singular values follow:
$$\rho(x) = \frac{1}{2\pi\sigma^2 q x}\sqrt{(b-x)(x-a)}$$
where $a = \sigma^2(1-\sqrt{q})^2$ and $b = \sigma^2(1+\sqrt{q})^2$

### Applications to Neural Networks
- **Weight Initialization**: Understanding spectral properties of initial weights
- **Training Dynamics**: How spectra evolve during optimization
- **Generalization**: Connection between spectral properties and model performance

## Results and Visualization

The analysis generates several types of visualizations:

1. **Spectral Density Plots**: Histograms of eigenvalues/singular values with theoretical overlays
2. **Attention Heatmaps**: Visualization of attention weight matrices
3. **Training Evolution**: Spectral property changes across checkpoints
4. **Per-Head Comparisons**: Multi-head attention analysis
5. **Quality Metrics**: WeightWatcher-derived model assessments

## Datasets Supported

The framework supports multiple graph datasets:
- **CIFAR-10** (superpixel graphs)
- **MNIST** (superpixel graphs)  
- **COCO Superpixels**
- **VOC Superpixels**
- **Peptides** (functional and structural)
- **PCQM4M** (quantum chemistry)
- **OGB datasets** (molecular property prediction)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new spectral analysis'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## Known Issues and Solutions

### Common Import Error
If you encounter:
```
AttributeError: 'NoneType' object has no attribute 'mem'
```

This has been fixed in the current version. Ensure you're using the latest code where `graphgps/act/example.py` handles `None` configuration gracefully.

### CUDA Memory Issues
For large models, consider:
- Reducing batch size in configs
- Using gradient checkpointing
- Analyzing smaller model subsets

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{spectral_assess_2025,
  title={Spectral Assessment for Graph Neural Networks},
  author={Sydney Dolan},
  year={2025},
  url={https://github.com/sydneyid/spectral_assess}
}
```

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