#  Neural Networks from Scratch (Python & NumPy)

This repository serves as a fundamental exploration into the mathematics of Deep Learning. It contains pure Python implementations of various Neural Network architectures—built without the aid of high-level frameworks like TensorFlow or PyTorch—to demonstrate the core algorithms driving modern AI.

##  Project Modules

### 1. Multi-Layer Perceptrons (MLP)
*File: `BasicNetwork.py`*
A collection of dense network architectures demonstrating the evolution of learning capacity:
* **Single Layer:** Implements basic linear separation logic.
* **Double Layer:** Introduces hidden layers and non-linear activation functions (Sigmoid) to solve more complex patterns.
* **Triple Layer:** A deep network implementation featuring full backpropagation and error calculation.

### 2. Recurrent Neural Networks (RNN) & NLP
*Files: `TextNatureRNN.ipynb`, `data.py`*
A custom RNN implementation designed for Natural Language Processing tasks, specifically sentiment analysis.
* **Architecture:** Manages hidden states (`h`) and time-step unrolling manually.
* **Forward Pass:** Uses Tanh activation for state updates and Softmax for output probabilities.
* **Backpropagation Through Time (BPTT):** Calculates gradients across time steps to learn sequence dependencies.
* **Data:** A custom labeled dataset of natural language phrases (e.g., *"i am very happy"*) mapped to sentiment booleans.

### 3. The Neuron Primitive
*File: `Neuron.py`*
The atomic unit of the network. A class-based implementation of a single neuron handling weights, biases, and feedforward calculations.

### 4. Training & Visualization
*File: `NeuralNet.ipynb`*
A Jupyter Notebook environment used to visualize the training process. It includes:
* Real-time loss calculation (Mean Squared Error).
* Epoch-by-epoch accuracy tracking.
* Dynamic user input for inference testing.

---

##  Tech Stack
* **Language:** Python 3.x
* **Core Math:** NumPy (Matrix multiplication, Dot products, Exponentials)
* **Visualization:** Matplotlib / Pandas

##  How to Run

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib
    ```

2.  **Train the MLP Models:**
    ```bash
    python BasicNetwork.py
    ```

3.  **Run the RNN Sentiment Analyzer:**
    Open `TextNatureRNN.ipynb` in Jupyter Notebook or VS Code to train the model on the text dataset and see it predict sentiment on new sentences.

---
*Built from first principles.*
