# Iris Classification with PyTorch

A beginner-friendly implementation of a **multilayer perceptron (MLP)** for classifying the three Iris flower species using PyTorch.

The project walks through the complete machine-learning workflow: loading and preparing data, defining a neural network, training it with mini-batch gradient descent, evaluating it on held-out data, plotting the learning curves, and optionally visualizing the network architecture.

## Project Overview

The model classifies flowers into:

- **Iris setosa**
- **Iris versicolor**
- **Iris virginica**

It uses four numerical measurements:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

The dataset contains **150 observations**, with 50 samples from each class.

## Model Architecture

The notebook implements the following fully connected neural network:

```text
Input layer       4 features
      ↓
Hidden layer 1    4 neurons + ReLU
      ↓
Hidden layer 2    3 neurons + ReLU
      ↓
Output layer      3 logits
```

The final layer returns raw logits rather than probabilities because `nn.CrossEntropyLoss` internally applies the required log-softmax operation.

## Training Configuration

| Setting | Value |
|---|---:|
| Train/test split | 70% / 30% |
| Training samples | 105 |
| Test samples | 45 |
| Split strategy | Stratified |
| Random state for split | 4 |
| Batch size | 16 |
| Epochs | 100 |
| Optimizer | Stochastic Gradient Descent |
| Learning rate | 0.01 |
| Loss function | Cross-entropy loss |

The model is evaluated on the test set after every epoch. Training loss, test loss, and test accuracy are monitored during training.

## Project Structure

```text
.
├── Iris In PyTorch.ipynb
└── README.md
```

## Requirements

A recent Python 3 environment and the following packages are required:

- `torch`
- `scikit-learn`
- `matplotlib`
- `jupyter`
- `visualtorch` — optional, used only for architecture visualization

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install torch scikit-learn matplotlib jupyter
```

To run the optional architecture-visualization section:

```bash
pip install visualtorch
```

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Iris In PyTorch.ipynb
```

Run the cells from top to bottom.

Alternatively, launch it directly:

```bash
jupyter notebook "Iris In PyTorch.ipynb"
```

## Workflow

The notebook performs these steps:

1. Loads the Iris dataset from `sklearn.datasets`.
2. Splits the data into stratified training and test sets.
3. Converts NumPy arrays into PyTorch tensors.
4. Wraps the training tensors in a `TensorDataset`.
5. Uses a `DataLoader` to create shuffled mini-batches.
6. Defines an MLP by subclassing `torch.nn.Module`.
7. Trains the network with backpropagation and SGD.
8. Measures test loss and classification accuracy after each epoch.
9. Plots the training and test loss curves.
10. Optionally renders the neural-network architecture with `visualtorch`.

## Example Training Output

During training, the notebook prints progress every 10 epochs:

```text
Epoch 10 | Train Loss: ... | Test Loss: ... | Test Acc: ...%
Epoch 20 | Train Loss: ... | Test Loss: ... | Test Acc: ...%
...
Epoch 100 | Train Loss: ... | Test Loss: ... | Test Acc: ...%
```

Exact results can vary because the network weights and DataLoader shuffling are not assigned fixed PyTorch random seeds.

## Reproducibility

The train/test split is reproducible because it uses:

```python
random_state=4
```

For fully reproducible model initialization and batch ordering, add seeds before creating the model and DataLoader:

```python
import random
import numpy as np
import torch

seed = 4

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

For supported CUDA environments, additional deterministic settings may be required.

## What This Project Demonstrates

This notebook is designed to teach:

- Tensor creation and data types in PyTorch
- Train/test splitting for multiclass classification
- Mini-batch loading with `DataLoader`
- Creating custom models with `nn.Module`
- ReLU activation functions
- Logits and cross-entropy loss
- Gradient resetting, backpropagation, and optimizer updates
- Training and evaluation modes
- Disabling gradient tracking during evaluation
- Computing multiclass accuracy
- Visualizing loss curves
- Visualizing a network architecture

## Possible Improvements

Useful extensions include:

- Standardizing the input features
- Adding a validation set instead of repeatedly evaluating on the test set
- Setting random seeds for reproducibility
- Comparing SGD with Adam
- Testing different hidden-layer sizes and activation functions
- Adding early stopping
- Reporting a confusion matrix and per-class metrics
- Saving and loading the trained model
- Using k-fold cross-validation
- Moving training to a reusable function or Python module
- Adding automated tests and a `requirements.txt` file

## Notes

This is an educational project rather than a production training pipeline. In a formal experiment, the test set should generally be reserved for final evaluation rather than monitored after every epoch. A separate validation set can be used for model selection and training decisions.

## License

GNU General Public License v3.0
