# HelmNet
Helmet detection using CNN and VGG-16 transfer learning to classify workers as with or without safety helmets for automated workplace safety monitoring.

# 🪖 HelmNet — Helmet Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![VGG-16](https://img.shields.io/badge/Transfer%20Learning-VGG--16-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-red)
![NumPy](https://img.shields.io/badge/NumPy-1.25.2-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Evaluation-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

Workplace safety in hazardous environments such as construction sites and industrial plants depends heavily on workers wearing safety helmets. Manual monitoring is error-prone and hard to scale across large operations. This project builds an **image classification system using CNNs and VGG-16 transfer learning** that automatically detects whether a worker is wearing a safety helmet.

Four models of increasing complexity were built and compared, from a basic CNN trained from scratch to a VGG-16 model with a deeper classifier and data augmentation, to find the most reliable model for real-world deployment.

---

## 🎯 Objective

> Build a deep learning model that classifies worker images as **With Helmet** or **Without Helmet**, enabling automated and scalable safety compliance monitoring.

- **Task Type:** Binary Image Classification
- **Target Variable:** Helmet Status (With Helmet = 1, Without Helmet = 0)
- **Primary Metric:** Accuracy (classes are near-balanced), with Recall, Precision and F1 Score also tracked

---

## 📂 Dataset

| Property | Details |
|---|---|
| Total Images | 631 |
| With Helmet (Label 1) | 311 |
| Without Helmet (Label 0) | 320 |
| Image Size | 200 x 200 pixels |
| Channels | RGB (3) |
| Image File | `images_proj.npy` (shape: 631 x 200 x 200 x 3) |
| Label File | `Labels_proj.csv` (single column: `Label`) |
| Environments | Construction sites, factories, industrial settings |
| Variations | Lighting, camera angles, worker postures and activities |

The dataset is near-balanced (~49% vs ~51%), so no resampling or class weighting was needed.

---

## 🔍 Data Preparation

- Loaded the image array and labels, and verified shapes
- Plotted sample images from each class to confirm label accuracy and image quality
- Checked class distribution with a count plot
- Converted images to **grayscale** using OpenCV for visual inspection (RGB retained for modelling)
- Applied a **stratified split** to preserve class ratios across all sets
- Normalised pixel values from 0–255 to 0–1 by dividing by 255

| Split | Images | Share |
|---|---|---|
| Training | 441 | 70% |
| Validation | 95 | 15% |
| Test | 95 | 15% |

---

## ⚙️ Modelling Pipeline

| Step | Details |
|---|---|
| Image Loading | NumPy (`.npy`) + Pandas (CSV labels) |
| Preprocessing | Pixel normalisation (÷255), stratified 70/15/15 split |
| Base Architectures | Simple CNN, VGG-16 (ImageNet weights, frozen) |
| Optimizer | Adam (learning rate = 0.001) |
| Loss Function | Binary Crossentropy |
| Output Layer | 1 neuron, Sigmoid activation |
| Epochs / Batch Size | 20 / 32 |
| Augmentation | Rotation, width/height shift, shear, zoom (Model 4 only) |
| Evaluation | Accuracy, Recall, Precision, F1 Score |

---

## 🧪 Models Built

| Model | Architecture | Total Params | Trainable Params |
|:-----:|:-------------|:------------:|:----------------:|
| **Model 1** | Simple CNN: 3 Conv2D (32 → 64 → 128), 2 MaxPooling (4x4), Dense(4), Sigmoid | 179,785 | 179,785 |
| **Model 2** | VGG-16 (frozen) → Flatten → Sigmoid | 14,733,121 | 18,433 |
| **Model 3** | VGG-16 (frozen) → Flatten → Dense(256) → Dropout(0.5) → Dense(128) → Sigmoid | 19,466,561 | 4,751,873 |
| **Model 4** | Same as Model 3, trained with **data augmentation** | 19,466,561 | 4,751,873 |

### Data Augmentation Settings (Model 4)

| Parameter | Value |
|---|---|
| Rotation range | 20° |
| Width shift | 0.2 |
| Height shift | 0.2 |
| Shear range | 0.2 |
| Zoom range | 0.2 |
| Fill mode | nearest |

---

## 📊 Evaluation Results

### Training Set Performance

| Model | Accuracy | Recall | Precision | F1 Score |
|:------|:--------:|:------:|:---------:|:--------:|
| Model 1 — Simple CNN | 1.000 | 1.000 | 1.000 | 1.000 |
| Model 2 — VGG-16 Base | 1.000 | 1.000 | 1.000 | 1.000 |
| Model 3 — VGG-16 + FFNN | 1.000 | 1.000 | 1.000 | 1.000 |
| Model 4 — VGG-16 + FFNN + Aug | 1.000 | 1.000 | 1.000 | 1.000 |

### Validation Set Performance

| Model | Accuracy | Recall | Precision | F1 Score |
|:------|:--------:|:------:|:---------:|:--------:|
| Model 1 — Simple CNN | 0.989 | 0.989 | 0.990 | 0.989 |
| Model 2 — VGG-16 Base | 1.000 | 1.000 | 1.000 | 1.000 |
| Model 3 — VGG-16 + FFNN | 0.979 | 0.979 | 0.980 | 0.979 |
| **Model 4 — VGG-16 + FFNN + Aug** | **1.000** | **1.000** | **1.000** | **1.000** |

### Train vs Validation Gap

| Model | Gap (Accuracy) |
|:------|:--------------:|
| Model 1 — Simple CNN | ~0.011 |
| Model 2 — VGG-16 Base | 0.000 |
| Model 3 — VGG-16 + FFNN | ~0.021 |
| Model 4 — VGG-16 + FFNN + Aug | 0.000 |

> Model 1 overfits slightly, Model 3 overfits a little more because of the larger classifier on a small dataset, and Models 2 and 4 show no train–validation gap.

---

## 🏆 Final Model — Model 4 (VGG-16 + FFNN + Data Augmentation)

**Why Model 4:**
- Matches Model 2 on all validation metrics with zero train–validation gap
- Trained on augmented images (rotations, shifts, shear, zoom), so it is better prepared for varied real-world camera angles and positions
- Keeps the richer classifier of Model 3 while fixing its overfitting
- Reliable performance matters most for a safety-critical application

**Test Set Performance:**

| Metric | Value |
|---|---|
| Accuracy | 100% |
| Recall | 100% |
| Precision | 100% |
| F1 Score | 100% |

Test performance matches validation performance, confirming strong generalisation. No non-compliant worker was missed (zero false negatives).

---

## 💡 Actionable Insights & Recommendations

- **Deploy Model 4** on existing CCTV infrastructure for real-time helmet detection at construction and industrial sites
- **Set up automated alerts** to notify site supervisors the moment a non-compliant worker is detected
- **Retrain periodically** with new site-specific images to keep accuracy high across different environments and lighting conditions
- **Expand the dataset** with edge cases such as partial helmets, night-time images, obstructed views and unusual angles
- **Validate on larger data** before production use, since the current dataset has only 631 images and the near-perfect scores should be confirmed on more varied footage

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn, OpenCV |
| Deep Learning | TensorFlow, Keras (VGG-16, ImageDataGenerator) |
| Evaluation | Scikit-learn |
| Environment | Google Colab (GPU) |

---

## 📁 Project Structure

```
HelmNet/
│
├── images_proj.npy                 # Image data (631 x 200 x 200 x 3)
├── Labels_proj.csv                 # Labels (0 = Without Helmet, 1 = With Helmet)
│
├── notebook/
│   └── HelmNet_Low_Code.ipynb      # Main project notebook
│
├── HelmNet_Presentation.pdf        # Business presentation (optional)
│
└── README.md                       # Project documentation
```

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/helmnet.git
cd helmnet
```

2. **Install dependencies**
```bash
pip install tensorflow numpy==1.25.2 pandas matplotlib seaborn opencv-python scikit-learn
```

3. **Add the data files**
- Place `images_proj.npy` and `Labels_proj.csv` in the root folder (or update the paths in the notebook)

4. **Run the notebook**
```bash
jupyter notebook notebook/HelmNet_Low_Code.ipynb
```

5. **Run all cells sequentially** from top to bottom

> 💡 Recommended: Run on **Google Colab** with a GPU runtime for faster training of the VGG-16 models.

---

## 📜 License

This project is for educational purposes as part of a deep learning / computer vision course.

---

## 🙌 Acknowledgements

- **Great Learning** for project guidance and structure
- **Keras / TensorFlow** for the deep learning framework and pretrained VGG-16 weights
- **ImageNet** for the dataset used to pretrain VGG-16
