# Lab 3 : Implement a Feedforward Neural Network using TensorFlow (Keras)

## 🎯 Aim

To implement a **Feedforward Neural Network (FNN)** using TensorFlow's Keras API and train it to classify the **AND logic gate**. This experiment demonstrates how multiple artificial neurons work together to learn patterns from data through forward propagation and backpropagation.

---

## 📚 Main Topics Covered

- Artificial Neural Networks (ANN)
- Feedforward Neural Network (FNN)
- TensorFlow
- Keras Sequential API
- Dense Layers
- Hidden Layer
- ReLU Activation Function
- Sigmoid Activation Function
- Binary Crossentropy Loss
- Adam Optimizer
- Backpropagation
- Binary Classification

---

## 📖 Theory

A **Feedforward Neural Network (FNN)** is one of the most fundamental architectures in Deep Learning. Unlike a single neuron or perceptron, an FNN contains multiple layers of interconnected neurons that can learn complex relationships within data.

The architecture consists of three main layers:

- **Input Layer** – Receives the input features.
- **Hidden Layer(s)** – Learns meaningful patterns and relationships from the data.
- **Output Layer** – Produces the final prediction.

During training, data flows from the input layer to the output layer (forward propagation). The prediction error is then calculated using a loss function, and the network updates its weights using **backpropagation** and an optimization algorithm such as Adam.

This learning process enables the network to continuously improve its predictions over multiple epochs.

---

## ⚙️ Code Concept

The implementation follows these steps:

1. Import NumPy and TensorFlow.
2. Create the AND gate dataset.
3. Define the Feedforward Neural Network using the Sequential API.
4. Add a hidden Dense layer with the ReLU activation function.
5. Add an output Dense layer with the Sigmoid activation function.
6. Compile the model using:
   - Adam Optimizer
   - Binary Crossentropy Loss
7. Train the model for multiple epochs.
8. Evaluate the model's accuracy.
9. Predict outputs for all possible AND gate inputs.

TensorFlow automatically performs forward propagation, computes the loss, applies backpropagation, and updates the network weights during training.

---

## 🖥️ Sample Output

```text
Training Accuracy: 100%

Predictions

Input: [0 0]
Output Probability: 0.0000
Prediction: 0

Input: [0 1]
Output Probability: 0.0001
Prediction: 0

Input: [1 0]
Output Probability: 0.0038
Prediction: 0

Input: [1 1]
Output Probability: 0.9998
Prediction: 1
```

---

## 🔍 Observations

- The Feedforward Neural Network successfully learned the AND gate with **100% training accuracy**.
- The hidden layer enabled the model to learn more efficiently than a single perceptron.
- The ReLU activation function accelerated the learning process by introducing non-linearity.
- The Sigmoid activation function produced probability values between 0 and 1 for binary classification.
- TensorFlow simplified model creation, training, and evaluation through its high-level Keras API.
- The Adam optimizer automatically adjusted the network weights during training, allowing the model to converge quickly.

---

## 🌍 Real-World Application

### Handwritten Digit Recognition

One of the most common applications of Feedforward Neural Networks is **handwritten digit recognition**, where computers automatically identify digits written by humans.

Consider an image of a handwritten number. Before being processed, the image is converted into numerical pixel values. These pixel values become the input features for the neural network.

The workflow is as follows:

1. The **input layer** receives the pixel values of the image.
2. The **hidden layer** learns important visual patterns such as edges, curves, and shapes.
3. The **output layer** predicts which digit (0–9) is present in the image.

For example, if a handwritten image of the digit **"7"** is given as input, the network analyzes the learned patterns and assigns the highest probability to the output neuron representing the digit **7**.

This technology is widely used in:

- Bank cheque processing
- Postal code recognition
- Optical Character Recognition (OCR)
- Automated form processing
- Educational examination systems
- Digitizing handwritten records

Although modern image recognition systems primarily use **Convolutional Neural Networks (CNNs)** for higher accuracy, Feedforward Neural Networks provide the fundamental understanding of how neural networks learn patterns from data. Many advanced deep learning architectures are built upon the same principles demonstrated in this experiment.

---

## 🚀 Advantages

- Learns complex patterns from data.
- Supports multiple hidden layers.
- Easily scalable to larger datasets.
- Provides high accuracy for classification tasks.
- Forms the foundation of modern Deep Learning.

---

## ⚠️ Limitations

- Requires more computational resources than a single neuron.
- Training time increases with dataset size.
- May overfit on very small datasets if not properly regularized.
- Feedforward Neural Networks are less effective for image data compared to CNNs and for sequential data compared to RNNs or Transformers.

---

## ✅ Conclusion

The Feedforward Neural Network was successfully implemented using TensorFlow and Keras. The model achieved **100% training accuracy** on the AND gate dataset while demonstrating the complete neural network workflow, including forward propagation, loss computation, backpropagation, and weight optimization.

This experiment extends the concepts learned from the Perceptron and Single Neuron practicals by introducing hidden layers and automated optimization techniques. It serves as an excellent introduction to modern Deep Learning and provides the conceptual foundation for advanced architectures used in computer vision, natural language processing, speech recognition, recommendation systems, and many other Artificial Intelligence applications.

---

## 🛠️ Technologies Used

- Python
- NumPy
- TensorFlow
- Keras
- Jupyter Notebook

---

## 🎓 Key Learning Outcomes

After completing this practical, the following concepts were understood:

- Feedforward Neural Network Architecture
- Input, Hidden, and Output Layers
- Dense Layers
- ReLU Activation Function
- Sigmoid Activation Function
- Binary Crossentropy Loss
- Adam Optimizer
- Forward Propagation
- Backpropagation
- Model Training and Evaluation
- Binary Classification using Deep Learning
