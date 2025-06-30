

# EMG Signal Classification for Hand/Arm Movements
![img](emf.jpg)


## 🔍 Overview

This project classifies electromyography (EMG) signals to predict arm movements (e.g., **elbow bending**, **hand closing**) using a **CNN-LSTM hybrid deep learning model**. The dataset includes EMG signals from the **biceps**, **triceps**, and **middle** muscles, labeled with six different movement classes.

---

## 📊 Dataset

* **Source:** `/kaggle/input/emgsen/emg_dataset.csv`
* **Shape:** 11,595 rows × 4 columns
* **Features:**

  * `biceps` (float64)
  * `triceps`, `middle` (int64)
  * `label` (target: object)

### ✅ Data Cleaning

* **Missing values:** 189 in `biceps` → imputed via **KNN (k=5)**
* **Duplicates:** 9,109 removed → final dataset: **2,486 rows**

---

## ⚙️ Preprocessing Pipeline

* **Standardization:** `StandardScaler` for zero mean, unit variance
* **Outlier Treatment:**

  * Clipping (1st–99th, then 5th–95th percentiles)
  * `PowerTransformer` (Yeo-Johnson)
* **Label Encoding:** `LabelEncoder` for categorical targets
* **Sequence Creation:** Transformed into **2,476 sequences** of shape **(10 timesteps × 3 features)** for time-series modeling

---

## 📈 Exploratory Data Analysis (EDA)

* **Boxplots:** Before/after standardization for outlier visualization
* **Scatter Plots:** Feature vs. label analysis
* **Correlation:** Weak linear correlation with target (`biceps: -0.057`, etc.) → complex models needed

---

## 🧠 Model Architecture

A **hybrid CNN-LSTM** model implemented with TensorFlow/Keras:

```
Input: (10, 3)
→ Conv1D (64 filters, kernel=3, ReLU)
→ MaxPooling1D (pool=2)
→ Dropout(0.3)
→ LSTM (128, return_sequences=True)
→ Dropout(0.5)
→ LSTM (64)
→ Dropout(0.3)
→ BatchNormalization
→ Dense (64, ReLU)
→ Dropout(0.3)
→ Dense (6, softmax)
```

* **Optimizer:** Adam (lr=0.0005, clipnorm=1.0)
* **Loss:** Categorical cross-entropy
* **Metric:** Accuracy

---

## 🏋️‍♂️ Training Configuration

* **Data Split:** 80% train / 20% test (with stratification)
* **Epochs:** 300 (early stopping at 97% val accuracy)
* **Batch Size:** 32
* **Callback:** Custom early stop at 97% validation accuracy

### 🏁 Results

* Stopped at **epoch 250**
* **Validation Accuracy:** 97.98%
* **Validation Loss:** 0.0766
* **Training Accuracy:** \~93.24%

---

## 📊 Visualizations

* Accuracy and loss plots showed **steady learning** and **convergence**
* Validation metrics outperformed training, indicating good generalization

---

## 📂 Outputs

* **Preprocessed Data:** `preprosesseddata.csv`
* **Trained Model:** `model.h5`










