# 3D MNIST Classification Project

This repository contains the implementation and optimization of **3D Convolutional Neural Networks (Conv3D)** for classifying 3D digit data by benchmarking different architectural changes and regularization and successfully able to 

---

## 📊 Performance Summary
* [cite_start]**Baseline Accuracy:** 75.67%[cite: 2].
* [cite_start]**Optimized Accuracy:** 77.35%[cite: 2].
* [cite_start]**Key Improvements:** Transitioned to deeper filter blocks and implemented L2 regularization[cite: 2, 4].

---

## 🏗️ Model Evolution

### 1. Baseline Architecture
[cite_start]The initial model used a three-block structure with increasing filter sizes (16, 32, 64)[cite: 1]:
* [cite_start]**Conv3D Blocks:** 3x3x3 kernels with ReLU activation and 'same' padding[cite: 1].
* [cite_start]**Normalization:** `BatchNormalization()` after every convolution[cite: 1].
* [cite_start]**Downsampling:** `MaxPooling3D(2)`[cite: 1].
* [cite_start]**Head:** Flatten followed by a Dense layer (128 units) and **Dropout (0.5)**[cite: 1].

### 2. Optimized High-Capacity Model
[cite_start]To improve accuracy, the filter density was increased (64, 128, 256) and the Dense layer was expanded to 256 units[cite: 2]:
* [cite_start]**Increased Filters:** Captured more complex spatial features[cite: 2].
* [cite_start]**Expanded Dense Layer:** 256 units for better classification power[cite: 2].

---

## 🛠️ Regularization & Training Strategy
To combat the overfitting observed in early loss curves, the following techniques were applied:

* [cite_start]**L2 Regularization:** Added `kernel_regularizer=l2(1e-4)` to the Conv3D layers to penalize weight complexity[cite: 3, 4].
* [cite_start]**Early Stopping:** Implemented to halt training when validation loss plateaued[cite: 3].
* [cite_start]**Patience Tuning:** Tested patience levels of **10** and **5** to find the optimal balance between training time and performance[cite: 3, 5].
* [cite_start]**Hyperparameter Consistency:** Verified performance by testing different architectures under the same hyperparameter settings[cite: 6].

---

## 📈 Results and Visualization
The project used the following metrics to evaluate success:
* [cite_start]**Training vs. Validation Loss:** Monitored to ensure the model wasn't just memorizing data[cite: 1, 4, 6].
* [cite_start]**Confusion Matrix:** Analyzed to identify which 3D digits were most frequently confused by the network[cite: 1, 6].
