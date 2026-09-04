# Blood Cell Classification Using Deep Learning

A comprehensive deep learning project for classifying blood cell types using PyTorch and Convolutional Neural Networks. It classifies blood-cell images into 8 types using SimpleCNN, ImprovedCNN, and ResNet architectures, with data augmentation, ablation studies, and detailed evaluation.

## Project Overview

This project implements multiple neural network architectures to classify 8 types of blood cells:
- Basophil
- Eosinophil
- Erythroblast
- IG (Immature Granulocyte)
- Lymphocyte
- Monocyte
- Neutrophil
- Platelet

## Models Implemented

1. **SimpleCNN**: Baseline CNN with 3 convolutional blocks
2. **ImprovedCNN**: CNN with BatchNorm and 3 convolutional blocks
3. **CustomResNet**: Main model with residual blocks (ResNet-style architecture)

## Project Structure

```
.
├── Classification code.ipynb      # Main Jupyter notebook with all experiments
├── class_map.json                 # Mapping of class names to IDs
├── prediction_labels.json         # Test predictions output
├── train/                         # Training dataset
│   ├── basophil/
│   ├── eosinophil/
│   ├── erythroblast/
│   ├── ig/
│   ├── lymphocyte/
│   ├── monocyte/
│   ├── neutrophil/
│   └── platelet/
└── test/                          # Test dataset
    └── [test images]
```

## Key Features

- **Data Augmentation**: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
- **Multiple Architectures**: Comparing Simple CNN vs ResNet
- **Ablation Study**: Testing different configurations (with/without augmentation, different optimizers, etc.)
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and training curves

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=0.24.0
pandas>=1.1.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
Pillow>=8.0.0
tqdm>=4.50.0
```

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd SUbmission
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook "Classification code.ipynb"
```

## Usage

The main notebook `Classification code.ipynb` contains:
- Cell 1: Imports and setup
- Cell 2: Configuration and hyperparameters
- Cell 3-5: Data exploration and visualization
- Cell 6-9: Model definitions (SimpleCNN, ImprovedCNN, CustomResNet)
- Cell 10: Training and evaluation functions
- Cell 11-12: Model training experiments
- Cell 13: Ablation study
- Cell 14-17: Evaluation and visualization
- Cell 18: Test predictions generation

## Results

- **CustomResNet Best Validation Accuracy**: ~95%+ (depending on data)
- **SimpleCNN Best Validation Accuracy**: ~90%+ (depending on data)
- Training curves, confusion matrices, and detailed metrics are generated for each model

## Hyperparameters

- Image Size: 224x224
- Batch Size: 64
- Learning Rate: 0.001
- Number of Epochs: 15-50
- Optimizer: Adam with weight decay (1e-4)

## Outputs

- `best_custom_resnet.pth`: Best model checkpoint
- `best_simple_cnn.pth`: Baseline model checkpoint
- `prediction_labels.json`: Test set predictions
- `confusion_matrices.png`: Model evaluation visualization
- `training_comparison.png`: Training curves comparison

## Author
Waqar Ahmad

## License

MIT License
