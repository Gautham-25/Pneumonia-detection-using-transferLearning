# 🩺 Pneumonia Detection

## 📌 Project Overview

Pneumonia is a serious lung infection that can be life‑threatening if not detected early. This project focuses on building an **AI‑based pneumonia detection system** using **Chest X‑ray images** and **Deep Learning**.

Instead of training a model from scratch, this system uses **Transfer Learning**, where a **pre‑trained convolutional neural network (CNN)** (such as ResNet, VGG, or MobileNet) is adapted for medical image classification. This improves accuracy, reduces training time, and performs well even with limited medical datasets.

The system helps doctors and healthcare professionals by providing fast and accurate preliminary diagnosis support.

---

## 🎯 Objectives

* Detect pneumonia from chest X‑ray images automatically.
* Reduce manual diagnostic workload.
* Improve early detection accuracy using deep learning.
* Provide a simple interface for prediction.

---

## 🧠 Technologies Used

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * TensorFlow / Keras
  * OpenCV
  * NumPy
  * Matplotlib
  * Scikit‑learn
* **Deep Learning Approach:** Transfer Learning
* **Pretrained Models (Example):** ResNet50 / VGG16 / MobileNetV2
* **Dataset:** Chest X‑ray Pneumonia Dataset (Kaggle)


---

## 📊 Dataset Description

The dataset contains chest X‑ray images categorized into:

* **Normal** – Healthy lungs
* **Pneumonia** – Infected lungs

Images are divided into:

* Training set
* Validation set
* Test set

---

## ⚙️ Methodology

1. **Data Collection** – Chest X‑ray dataset obtained from Kaggle(https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. **Data Preprocessing**

   * Image resizing
   * Normalization
   * Data augmentation
3. **Model Building (Transfer Learning)**

   * Load pretrained CNN (ImageNet weights)
   * Freeze base layers
   * Add custom classification layers
   * Fine‑tune upper layers for medical image learning
4. **Model Training****

   * Loss Function: Binary Crossentropy
   * Optimizer: Adam
5. **Evaluation**

   * Accuracy
   * Precision
   * Recall
   * Confusion Matrix
6. **Prediction**

   * Upload X‑ray → Model predicts Normal or Pneumonia.

---

## 🏗️ Model Architecture (Transfer Learning)

* Pretrained CNN Backbone (ResNet50 / VGG16 / MobileNetV2)
* Frozen Feature Extraction Layers
* Global Average Pooling Layer
* Fully Connected Dense Layer
* Dropout (Regularization)
* Output Layer (Sigmoid for Binary Classification)


---

## 📈 Results

* Training Accuracy: ~91% (example)
* Validation Accuracy: ~88-90%
* Faster convergence due to Transfer Learning
* Improved feature extraction using pretrained ImageNet weights
* Model successfully distinguishes infected lungs from normal lungs.

---


## ⚠️ Limitations

* Model depends on dataset quality.
* Not a replacement for professional medical diagnosis.
* Requires further clinical validation.

---

## 👨‍💻 Authors

* Final Year Project Team

---

## 📜 License

This project is for academic and research purposes only.

---

## ⭐ Acknowledgements

* Kaggle Chest X‑ray Dataset
* TensorFlow & Open‑source community

---

**"AI assisting healthcare for faster and smarter diagnosis."**
