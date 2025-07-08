# CIFAR-10 Image Classification using Artificial Neural Network (ANN)

This project implements a fully connected Artificial Neural Network (ANN) using **TensorFlow** and **Keras** to classify images from the **CIFAR-10** dataset.

> ⚠️ Convolutional layers are **not used** — this project focuses purely on ANN performance on image data.

---

## 🧠 Model Architecture

The model consists of a series of Dense layers with ReLU activations and dropout regularization:

Input: Flattened 32x32x3 images → shape = (3072,)

Dense(512, activation='relu') → Dropout(0.2)
Dense(256, activation='relu') → Dropout(0.2)
Dense(128, activation='relu') → Dropout(0.2)
Dense(64, activation='relu') → Dropout(0.2)
Dense(10, activation='softmax')


---

## 📚 Dataset

- **CIFAR-10** is a collection of 60,000 32×32 RGB images in 10 classes:
  - `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- **Train/Test Split**: 50,000 training and 10,000 testing images

---

## ⚙️ Preprocessing

- **Flattening**: Images reshaped from (32, 32, 3) to (3072,)
- **Normalization**: Pixel values scaled to [0, 1]
- **One-hot Encoding**: Labels converted to categorical format

---

## 🛠️ Training Configuration

| Setting         | Value            |
|----------------|------------------|
| Optimizer       | Adam             |
| Learning Rate   | 0.0001           |
| Loss Function   | Categorical Crossentropy |
| Epochs          | 50               |
| Batch Size      | 64               |

---

## 📈 Results

| Metric              | Accuracy       |
|---------------------|----------------|
| **Training Accuracy** | 52.66%         |
| **Test Accuracy**     | 52.57%         |

---

## 📊 Evaluation
Check "Evulation" folder for loss, accuracy and confusion matrix plots

### ✅ Classification Report
              precision    recall  f1-score   support

    airplane       0.25      0.75      0.37      1000
  automobile       0.67      0.36      0.46      1000
        bird       0.42      0.01      0.02      1000
         cat       0.26      0.03      0.06      1000
        deer       0.00      0.00      0.00      1000
         dog       0.18      0.75      0.29      1000
        frog       0.75      0.00      0.01      1000
       horse       0.47      0.45      0.46      1000
        ship       0.64      0.39      0.49      1000
       truck       0.63      0.34      0.44      1000

    accuracy                           0.31     10000
   macro avg       0.43      0.31      0.26     10000
weighted avg       0.43      0.31      0.26     10000


🧩 Limitations
ANN lacks spatial understanding, so results are limited on image data like CIFAR-10.

For higher accuracy (>75%), CNNs are preferred.

⭐ Future Work
Add Convolutional layers (CNN) to improve accuracy

Experiment with learning rate schedules and regularization techniques

Add hyperparameter tuning and visualization dashboards

👨‍💻 Author
Muhammad Muaaz
