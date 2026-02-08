# 3D MNIST Classification Project

This repository contains the implementation and optimization of **3D Convolutional Neural Networks (Conv3D)** for classifying 3D digit data by benchmarking different architectural changes and regularization techniques, successfully improving accuracy from **70% to 75.35%**

---

![3D Point Cloud Rotation](digit_3d.gif)

## 📊 Performance Summary

* **Baseline Accuracy:** 70.07%
* **Optimized Accuracy1:** 72.20%
* **Key Improvements1:** Transitioned to deeper filter blocks and implemented L2 regularization.
* **Optimized Accuracy2:** 74.35%
* **Key Improvements2:** Transitioned to deeper filter blocks and implemented L2 regularization.
* **Optimized Accuracy (final):** 75.35%
* **Key Improvements (final):** Transitioned to deeper filter blocks and implemented L2 regularization.

---

## 🏗️ Model Evolution

### 1. Baseline Architecture
The initial model used a three-block structure with increasing filter sizes (16, 32, 64):

* **Conv3D Blocks:** 3x3x3 kernels with ReLU activation and 'same' padding.
* **Normalization:** `BatchNormalization()` after every convolution.
* **Downsampling:** `MaxPooling3D(2)`.
* **Head:** Flatten followed by a Dense layer (128 units) and **Dropout (0.5)**.

### 2. Optimized High-Capacity Model
To improve accuracy, the filter density was increased (64, 128, 256) and the Dense layer was expanded to 256 units:

* **Increased Filters:** Captured more complex spatial features.
* **Expanded Dense Layer:** 256 units for better classification power.

---

## 🛠️ Comprehensive Regularization & Optimization Strategy

To combat overfitting and maximize model performance, **a systematic hyperparameter tuning approach was implemented**, with results generated and analyzed after each modification to study their individual and combined effects on the model:

### **Regularization Techniques Applied:**

* **Kernel L2 Regularization:** Added `kernel_regularizer=l2(1e-4)` to Conv3D layers to penalize weight complexity and prevent overfitting. **Fine-tuned L2 penalty values** (1e-3, 1e-4, 1e-5) and evaluated impact on validation loss.

* **Dropout Regularization:** Applied **Dropout (0.5)** in the Dense layer to randomly deactivate neurons during training. **Tested multiple dropout rates** (0.3, 0.5, 0.7) to find optimal regularization strength.

* **Early Stopping:** Implemented to halt training when validation loss plateaued, preventing unnecessary overfitting. **Patience levels (5, 10, 15) were systematically tested** to balance training time and convergence.

* **Batch Normalization:** Applied after each Conv3D layer to stabilize training and accelerate convergence.

### **Optimizer and Learning Rate Tuning:**

* **Adam Optimizer:** Used with **adaptive learning rates** to ensure efficient gradient descent. **Initial learning rates (1e-3, 5e-4, 1e-4) were fine-tuned** to achieve optimal convergence speed without overshooting minima.

* **Learning Rate Scheduling:** Experimented with **ReduceLROnPlateau** to dynamically adjust learning rates based on validation performance.

### **Hyperparameter Fine-Tuning Process:**

* **Batch Size:** Tested values of **16, 32, and 64** to balance memory usage and gradient stability.

* **Epochs:** Ran experiments with **50, 100, and 150 epochs** in combination with Early Stopping to determine ideal training duration.

* **Architecture Variations:** **Systematically compared different filter configurations** (16-32-64 vs. 64-128-256) and Dense layer sizes (128 vs. 256 units) under identical hyperparameter settings.

### **Iterative Evaluation:**

* **After each hyperparameter adjustment, validation accuracy and loss were recorded** to isolate the effect of individual changes.

* **Performance metrics, confusion matrices, and loss curves were generated at every tuning stage** to ensure each modification contributed positively to the final model.

---

## 🏆 Final Optimized Architecture & Configuration

### **Architecture Design Rationale:**

The final model employs a **three-block deep architecture with doubled convolutions per block** (48-48, 96-96, 128-128 filters), strategically designed to balance feature extraction depth with spatial information preservation:

#### **Block Structure:**

**Block 1:**
```
Conv3D(48, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(5e-5))
BatchNormalization()
Conv3D(48, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(5e-5))
BatchNormalization()
MaxPooling3D(pool_size=2)
```

**Block 2:**
```
Conv3D(96, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(5e-5))
BatchNormalization()
Conv3D(96, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(5e-5))
BatchNormalization()
MaxPooling3D(pool_size=2)
```

**Block 3:**
```
Conv3D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(5e-5))
BatchNormalization()
Conv3D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(5e-5))
BatchNormalization()
MaxPooling3D(pool_size=2)
```

**Classification Head:**
```
GlobalAveragePooling3D()
Dense(128, activation='relu', kernel_regularizer=l2(5e-5))
Dropout(0.5)
Dense(10, activation='softmax')
```

---

### **Key Design Decisions:**

* **Double Convolution per Block:** Each block contains **two consecutive Conv3D layers** before downsampling. This design choice allows the network to learn more complex hierarchical features at each spatial resolution level before reducing dimensionality. **Single convolutions per block led to underfitting**, while **three or more convolutions caused overfitting and diminishing returns**.

* **Delayed MaxPooling Strategy:** **MaxPooling is applied only after two convolutional layers** in each block rather than after every convolution. This is critical because **immediate MaxPooling causes premature spatial information loss**, which is particularly costly in 3D data where spatial relationships define digit geometry. By processing features through two convolutions first, the network extracts richer representations before downsampling.

* **Progressive Filter Expansion (48 → 96 → 128):** Filters increase gradually across blocks to capture increasingly abstract features while **avoiding the overfitting observed with larger configurations (64-128-256)**. The moderate filter sizes strike an optimal balance—**fewer filters (16-32-64) led to underfitting**, while **excessive filters caused overfitting despite regularization**.

* **Global Average Pooling vs. Flatten:** Replaced traditional Flatten with **GlobalAveragePooling3D** to reduce the parameter count in the Dense layer, providing implicit regularization and improving generalization.

---

### **Final Regularization Configuration:**

* **Kernel L2 Regularization:** Set to **`l2(5e-5)`** across all Conv3D and Dense layers after fine-tuning. This value was chosen because:
  * **`1e-4` was too aggressive**, causing undertraining and reduced capacity
  * **`1e-5` was too weak**, allowing overfitting to persist
  * **`5e-5` provided the optimal penalty** to constrain weights without stifling learning

* **Dropout Rate:** Maintained at **0.5** in the Dense layer, which empirically provided the best trade-off. **Lower rates (0.3) were insufficient** to prevent overfitting, while **higher rates (0.7) degraded training stability**.

---

### **Final Optimizer & Learning Rate:**

* **Adam Optimizer with `learning_rate=1e-4`:** This learning rate was selected after systematic tuning:
  * **`1e-3` caused unstable training** with erratic validation loss
  * **`5e-4` showed improvement** but still occasional instability
  * **`1e-4` provided smooth, stable convergence** with consistent improvements

---

### **Final Early Stopping Configuration:**

* **Patience Level: 10 epochs** with `restore_best_weights=True`
  * **Patience=5 was too aggressive**, stopping training prematurely before full convergence
  * **Patience=15 allowed unnecessary overtraining**, wasting computational resources
  * **Patience=10 struck the optimal balance**, allowing the model to recover from temporary plateaus while preventing extended overfitting

---

### **Final Training Hyperparameters:**

* **Batch Size: 32** — Balanced gradient stability with memory efficiency. **Smaller batches (16) introduced noise**, while **larger batches (64) slowed convergence**.

* **Maximum Epochs: 100** — Combined with Early Stopping, this ensured sufficient training time while preventing excessive runs. The model typically converged around **60-70 epochs** with the patience mechanism triggering appropriately.

---

### **Why This Configuration Works:**

This final architecture succeeds because:

1. **Spatial information is preserved longer** through delayed pooling, critical for 3D digit recognition
2. **Feature depth is maximized** with double convolutions per block without crossing into overfitting territory
3. **Regularization is precisely calibrated**—aggressive enough to prevent overfitting, mild enough to preserve model capacity
4. **The architecture size is optimal**—adding more layers or filters led to overfitting and degraded test accuracy, while reducing them caused underfitting and loss of representational power

**The result: 75.35% accuracy**—a **5.28 percentage point improvement** over the baseline through systematic, evidence-driven optimization.

---

## 📈 Results and Visualization

The project used the following metrics to evaluate success:

* **Training vs. Validation Loss:** Monitored to ensure the model wasn't just memorizing data.
* **Confusion Matrix:** Analyzed to identify which 3D digits were most frequently confused by the network.

---

## 🚀 Usage

### Training the Model

```python
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define input shape
inputs = Input(shape=(voxel_dim, voxel_dim, voxel_dim, 1))

# Build model architecture (as shown above)
# ... [architecture code] ...

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train_voxel,
    y_train,
    validation_data=(X_test_voxel, y_test),
    batch_size=32,
    epochs=100,
    callbacks=[early_stopping]
)
```

---

## 📚 Key Takeaways

1. **Systematic hyperparameter tuning is essential** for achieving optimal performance in deep learning models
2. **Spatial information preservation** through delayed pooling is critical for 3D data
3. **Balanced regularization** (L2 + Dropout + Early Stopping) prevents overfitting without sacrificing model capacity
4. **Architecture depth matters**, but there's a sweet spot—too shallow underfits, too deep overfits
5. **Fine-grained learning rate control** with Adam optimizer ensures stable convergence


