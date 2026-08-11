# ANN Lab 3 - Feedforward Neural Network for Loan Approval Prediction

A practical implementation of a **Feedforward Neural Network (FNN)** using **TensorFlow/Keras** to predict whether a customer's loan application should be **Approved** or **Rejected** based on their financial profile.

---

## 📌 Objective
To build and train a multi-layer Feedforward Neural Network that learns to classify loan applications using key financial and employment features, and to understand how hidden layers and non-linear activation functions enable the network to model more complex decision boundaries than a single neuron.

---

## 🧠 What is a Feedforward Neural Network?

A **Feedforward Neural Network (FNN)**, also known as a **Multi-Layer Perceptron (MLP)**, is the simplest type of artificial neural network where information moves in only one direction — from input nodes, through hidden nodes, to output nodes. Unlike a single perceptron, an FNN contains **one or more hidden layers** between the input and output, allowing it to learn non-linear relationships and solve problems that are not linearly separable.

### Architecture Overview
```
Input Layer (3 neurons)  →  Hidden Layer 1 (8 neurons, ReLU)
                                    ↓
                         Hidden Layer 2 (4 neurons, ReLU)
                                    ↓
                         Output Layer (1 neuron, Sigmoid)
```

### Key Components
| Component | Description |
|:---|:---|
| **Input Layer** | Receives the raw feature values (Credit Score, Income, Employment) |
| **Hidden Layers** | Learn intermediate representations and non-linear patterns |
| **Output Layer** | Produces the final prediction (probability of approval) |
| **ReLU Activation** | `max(0, x)` — introduces non-linearity, helps network learn complex patterns |
| **Sigmoid Activation** | `1 / (1 + e^(-x))` — squashes output to a probability between 0 and 1 |
| **Adam Optimizer** | Adaptive learning rate optimizer for efficient convergence |
| **Binary Crossentropy** | Loss function for binary classification tasks |

---

## 🛠️ Technologies Used
- Python 3.x
- NumPy
- TensorFlow 2.x
- Keras (via TensorFlow)

---

## 📁 Project Structure
```
ANN-Lab-3/
│── loan_approval.ipynb       # Loan Approval Prediction (Jupyter/Colab)
│── README.md
└── requirements.txt
```

---

## 🚀 Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ANN-Lab-3.git
```

### 2. Open the Project
```bash
cd ANN-Lab-3
```

### 3. Create a Virtual Environment
**Windows**
```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment
**PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```
**Command Prompt**
```cmd
.venv\Scripts\activate
```

### 5. Install Dependencies
```bash
pip install numpy tensorflow
```

### 6. Run the Notebook
Open `loan_approval.ipynb` in **Jupyter Notebook**, **JupyterLab**, or **Google Colab** and run all cells.

---

## ⚙️ Program Description

### 🎯 Problem Statement
Banks and financial institutions receive thousands of loan applications daily. Manually reviewing each application is time-consuming and prone to human bias. A neural network can learn from historical approval/rejection data to automatically predict whether a new applicant is likely to repay the loan, helping banks make faster, data-driven decisions.

### 📊 Feature Engineering
We extract three key features from each loan application:

| Feature | Description | Range / Value |
|:---|:---|:---:|
| **Credit Score** | A numerical score representing creditworthiness | 300 – 900 |
| **Annual Income** | Yearly income in Lakhs (Indian Rupees) | 0 – 15+ |
| **Employment Status** | Whether the applicant has a stable salaried job | 1 = Salaried, 0 = Unemployed/Self-employed |

These features are chosen because they are the most critical factors banks evaluate:
- **Credit Score**: Reflects the applicant's credit history and repayment behavior. Higher scores indicate lower risk.
- **Annual Income**: Determines the applicant's ability to repay the loan. Higher income means better repayment capacity.
- **Employment Status**: Salaried employees have a stable, predictable income stream, making them lower-risk borrowers compared to unemployed or self-employed individuals.

### 📊 Training Dataset

| Credit Score | Annual Income (Lakhs) | Employment Status | Loan Approved? |
|:---:|:---:|:---:|:---:|
| 350 | 2.5 | 0 (Unemployed) | 0 (Rejected) |
| 720 | 8.0 | 1 (Salaried) | 1 (Approved) |
| 800 | 12.0 | 1 (Salaried) | 1 (Approved) |
| 400 | 3.0 | 0 (Unemployed) | 0 (Rejected) |
| 680 | 7.5 | 1 (Salaried) | 1 (Approved) |
| 450 | 4.0 | 0 (Unemployed) | 0 (Rejected) |
| 750 | 10.0 | 1 (Salaried) | 1 (Approved) |
| 380 | 2.8 | 0 (Unemployed) | 0 (Rejected) |

### 🔧 Preprocessing: Feature Normalization
Since features have vastly different scales (Credit Score: 300–900, Income: 2–12, Employment: 0–1), we normalize them to a 0–1 range to ensure the neural network learns efficiently without being dominated by larger-scale features.

```python
X[:,0] = X[:,0] / 900    # Credit Score normalized by max 900
X[:,1] = X[:,1] / 15     # Income normalized by assumed max 15 Lakhs
```

### 🔄 How the Neural Network Learns
1. **Forward Propagation**: Input features pass through hidden layers. Each neuron computes a weighted sum, adds a bias, and applies ReLU activation.
2. **Non-Linear Transformation**: Hidden layers transform the input into a higher-dimensional space where the classes become separable.
3. **Output Prediction**: The final layer uses Sigmoid to output a probability between 0 and 1.
4. **Loss Calculation**: Binary Crossentropy measures the difference between predicted probabilities and true labels.
5. **Backpropagation**: The Adam optimizer computes gradients and updates weights/biases to minimize loss.
6. **Convergence**: Training repeats for 100 epochs until the model achieves high accuracy.

### 🏗️ Model Architecture
```python
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),   # Hidden Layer 1: 8 neurons
    Dense(4, activation='relu'),                      # Hidden Layer 2: 4 neurons
    Dense(1, activation='sigmoid')                    # Output Layer: 1 neuron
])
```

| Layer | Neurons | Activation | Purpose |
|:---|:---:|:---:|:---|
| Input | 3 | — | Receives normalized features |
| Hidden 1 | 8 | ReLU | Learns basic patterns (e.g., high credit + salaried = likely approved) |
| Hidden 2 | 4 | ReLU | Combines patterns into higher-level decisions |
| Output | 1 | Sigmoid | Outputs approval probability |

### ✅ Expected Output
```
Accuracy: 1.0

Loan Approval Predictions:
Customer 1: Credit=350, Income=2.5L, Salaried=0 -> Rejected (Confidence: 0.023)
Customer 2: Credit=720, Income=8.0L, Salaried=1 -> Approved (Confidence: 0.987)
Customer 3: Credit=800, Income=12.0L, Salaried=1 -> Approved (Confidence: 0.995)
Customer 4: Credit=400, Income=3.0L, Salaried=0 -> Rejected (Confidence: 0.015)
Customer 5: Credit=680, Income=7.5L, Salaried=1 -> Approved (Confidence: 0.978)
Customer 6: Credit=450, Income=4.0L, Salaried=0 -> Rejected (Confidence: 0.031)
Customer 7: Credit=750, Income=10.0L, Salaried=1 -> Approved (Confidence: 0.991)
Customer 8: Credit=380, Income=2.8L, Salaried=0 -> Rejected (Confidence: 0.019)
```
> *Note: Exact confidence values may vary slightly due to random weight initialization, but the final classification (Approved/Rejected) should remain consistent.*

---

## 🌍 Real-World Applications of Feedforward Neural Networks

| Domain | Application |
|---|---|
| **Banking & Finance** | Loan approval, credit scoring, fraud detection, risk assessment |
| **Healthcare** | Disease diagnosis, patient risk stratification, medical image analysis |
| **E-Commerce** | Customer churn prediction, recommendation systems, sales forecasting |
| **Marketing** | Lead scoring, ad click prediction, customer segmentation |
| **Manufacturing** | Quality control, defect detection, predictive maintenance |
| **Education** | Student performance prediction, dropout risk assessment |
| **Cybersecurity** | Intrusion detection, malware classification, anomaly detection |

---

## ⚠️ Limitations
- **Small Dataset**: This demo uses only 8 samples. Real-world models require thousands of samples for robust generalization.
- **No Train/Test Split**: For simplicity, we evaluate on training data. In practice, always split data into training and testing sets.
- **Feature Simplicity**: Real loan approval systems use 20+ features (debt-to-income ratio, loan amount, employment history, etc.).
- **Black Box Nature**: Neural networks are less interpretable than decision trees or logistic regression — banks often need explainable AI for regulatory compliance.
- **Overfitting Risk**: With small data and many parameters, the model may memorize rather than generalize.

> These limitations motivate advanced techniques like **Regularization (Dropout, L2)**, **Cross-Validation**, **Larger Datasets**, and **Explainable AI (XAI)** methods like SHAP and LIME.

---

## 📚 Concepts Covered
- Feedforward Neural Network (FNN)
- Multi-Layer Perceptron (MLP)
- Hidden Layers
- ReLU Activation Function
- Sigmoid Activation Function
- Binary Crossentropy Loss
- Adam Optimizer
- Feature Normalization / Scaling
- Forward Propagation
- Backpropagation
- Binary Classification
- TensorFlow / Keras

---

## 🔖 References
- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/

---

## 👤 Author
**Divyansh**
BSc Data Science and AI, Christ University
Deep Learning Laboratory — BDA404-5N
