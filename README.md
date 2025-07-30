# DSSA
DSSA: Dual-Stream Synthetic Accessibility Framework for Chemical Compounds


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2022.03%2B-green)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art deep learning framework for predicting synthetic accessibility of chemical compounds using dual-stream graph attention networks.

🌟 Features

- Dual-Stream Architecture: Combines molecular graph representation with chemical descriptors
- Graph Attention Networks: Leverages GAT and GCN layers for molecular feature extraction
- High Performance: Achieves superior accuracy in synthetic accessibility prediction
- Easy Integration: Simple API for batch predictions and single molecule scoring
- Pre-trained Models: Ready-to-use models trained on large chemical databases

Quick Start

Online Predictor
For quick predictions without installation, use our web-based predictor:
[DSSA Predictor](http://dssa.denglab.org) - Get instant synthetic accessibility scores online!

Local Installation

```bash
# Clone the repository
git clone https://github.com/Q-Aljanabi/DSSA.git
cd DSSA

# Install dependencies
pip install -r requirements.txt
```

📁 Project Structure

```
DSSA/
├── DSSA_model.py             # Main model architecture
├── train.py                  # Training script
├── test.py                   # Testing and evaluation
├── test_coconut_moses.py     # Additional testing
├── Score.py                  # Scoring utilities
├── data_preprocess.py        # Data preprocessing
├── best_model.pth            # Pre-trained model weights (4.43MB)
├── datasets_emolecules_uspto.pkl.gz  # Training dataset (5.91MB)
├── datasets/                 # Additional datasets
├── saved_plots/             # Visualization outputs
├── requirements.txt         # Python dependencies
├── training.log             # Training logs
├── .gitattributes           # Git LFS configuration
└── README.md               # This file
```

🔧 Usage

### Quick Prediction

```python
from DSSA_model import DSSA
from Score import score

# Load pre-trained model
model = DSSAModel.load_pretrained('best_model.pth')

# Predict synthetic accessibility for a SMILES string
smiles = "CCO"  # Ethanol
score = calculate_dssa_score(model, smiles)
print(f"DSSA Score: {score:.3f}")
```

### Batch Prediction

```python
import pandas as pd
from Score import batch_predict

# Load your dataset
df = pd.read_csv('your_molecules.csv')
smiles_list = df['SMILES'].tolist()

# Get predictions
scores = batch_predict(model, smiles_list)
df['DSSA_Score'] = scores
```

### Training Your Own Model

```python
# Train a new model
python train.py --data_path datasets_emolecules_uspto.pkl.gz \
                --epochs 200 \
                --batch_size 256 \
                --learning_rate 0.001
```

## 📊 Model Performance

Our DSSA model achieves exceptional performance across multiple test datasets:

| Dataset   | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity |
|-----------|----------|-----------|--------|----------|---------|--------|-------------|
| Main Test | 95.8%    | 94.7%     | 97.0%  | 95.9%    | 99.4%   | 99.4%  |  94.6%      | 
| TS1       | 99.4%    | 99.1%     | 99.7%  | 99.4%    | 100.0%  | 100.0% |  99.1%      | 
| TS2       | 87.4%    | 91.0%     | 78.2%  | 84.1%    | 95.3%   | 94.0%  |  94.2%      | 
| TS3       | 80.3%    | 91.6%     | 66.8%  | 77.2%    | 92.8%   | 91.6%  |  93.9%      | 

Key Highlights:
- 🎯 Outstanding performance on main test set with 95.8% accuracy
- 🏆 Near-perfect results on TS1 dataset (99.4% accuracy)
- 📈 Robust ROC-AUC scores above 92% across all datasets
- ⚡ Low loss values indicating excellent model convergence

🔬 Methodology

DSSA employs a dual-stream architecture that processes molecular information through two parallel pathways:

1. **Graph Stream**: Uses Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN) to capture structural patterns
2. **Descriptor Stream**: Processes molecular descriptors and fingerprints
3. **Fusion Layer**: Combines both streams for final prediction

Key Components

- **Molecular Graph Representation**: Atoms as nodes, bonds as edges
- **Attention Mechanism**: Focuses on synthetically relevant substructures
- **Multi-task Learning**: Joint optimization of accessibility and complexity prediction

📈 Interpreting DSSA Scores

- Score Range: 0.0 - 1.0
- 0.0 - 0.3: Hard to synthesize
- 0.3 - 0.7: Moderate difficulty
- 0.7 - 1.0: Easy to synthesize

🔗 Web Interface

Visit our online predictor for instant results:
[DSSA Predictor](http://dssa.denglab.org)

Features:
- ✅ Single molecule prediction
- ✅ Batch upload support
- ✅ Visualization of molecular complexity
- ✅ Export results to CSV
- ✅ No installation required

📋 Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.1+
- RDKit 2022.03+
- NumPy, Pandas, Matplotlib
- See `requirements.txt` for full list

🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.



📖 Citation

If you use DSSA in your research, please cite:

```bibtex



📄 License

This project is licensed under the MIT License .

👥 Authors



🙏 Acknowledgments

- RDKit community for cheminformatics tools
- PyTorch Geometric team for graph neural network implementations
- Chemical databases providers (eMolecules, USPTO)
📞 Support

- 🌐 Web Predictor: [http://dssa.denglab.org](http://dssa.denglab.org)
- 📧 Email: [qahtan.shuheep@qq.com]
- 🐛 Issues: [GitHub Issues](https://github.com/Q-Aljanabi/DSSA/issues)

---

