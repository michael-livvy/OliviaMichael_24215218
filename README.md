# Machine Learning Labs

A collection of ML/DL implementations covering classification, regression, and image recognition using NumPy, TensorFlow, and PyTorch.

## Labs

### Lab 1 — Perceptron (NumPy)

Single-layer perceptron for binary classification (pass/fail prediction).

- Implements the perceptron learning rule from scratch
- Updates weights on misclassification
- Visualizes the decision boundary

### Lab 2 — MLP Classification (TensorFlow/Keras)

Multi-layer perceptron for loan approval prediction using 4 features.

- Feedforward neural network
- Sigmoid output for binary classification
- Model building and training
- Evaluation using a confusion matrix

### Lab 3 — MLP Classification (PyTorch)

Spam email classifier using 4 features:

- Number of links
- Capitalization ratio
- Number of recipients
- Number of spam-related words

Implements the same network architecture in PyTorch using an explicit training loop:

`Forward → Loss → Backward → Update`

This provides a comparison with the higher-level Keras training workflow.

### Lab 4 — Learning Rules Comparison (NumPy)

A single neuron trained using 5 different learning rules on the same dataset:

- **Hebbian** — Unsupervised, correlation-based learning
- **Perceptron** — Discrete error-based learning
- **Delta (Widrow-Hoff)** — Continuous gradient-based learning
- **Correlation** — Supervised Hebbian learning
- **Outstar** — Direct target-pull learning

Compares convergence speed and final accuracy across the different learning rules.

### Lab 5 — CNN Image Classification (TensorFlow/Keras)

Cats vs. Dogs image classifier using a convolutional neural network.

Architecture:

`Conv2D → MaxPool → Conv2D → MaxPool → Dense → Output`

Images are loaded directly from raw image directories using `ImageDataGenerator`.

### Lab 6 — MLP Regression (TensorFlow/Keras)

Regression using the California Housing dataset:

- **20,640 samples**
- **8 input features**
- Continuous target variable

Compares:

- **3 activation functions:** ReLU, Tanh, Sigmoid
- **3 loss functions:** MSE, MAE, Huber
- **5 different experiments**

Includes:

- Training/loss curves
- Predicted vs. actual plots
- Comparative analysis of model performance

## Tech Stack

| Library | Used For |
|---|---|
| NumPy | Raw mathematical operations, perceptron, learning rules |
| TensorFlow | Keras MLPs, CNN, regression |
| PyTorch | Explicit training-loop MLP |
| scikit-learn | Datasets, scaling, and evaluation metrics |
| Matplotlib | Visualizations |
| Pandas | Data manipulation and results tables |

## Installation

Install the required dependencies using:

```bash
pip install numpy tensorflow torch scikit-learn matplotlib pandas seaborn