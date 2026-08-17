# Animal_Image_Classification_CNN_Model

This project implements a Convolutional Neural Network (CNN) built with **TensorFlow / Keras** to classify images into **90 different animal categories**. The pipeline includes converting raw dataset images into pixel arrays saved in structured CSV format, followed by data preprocessing, normalization, and deep learning model evaluation.

---

## 📁 Repository Structure

```text
├── animals/                            # Dataset folder (Ignored by Git)
├── animal_pixels.csv                   # Processed pixel CSV file (Ignored by Git)
├── Making_animal_usable_file.ipynb     # Image preprocessing & CSV generation script
├── animal_image_classification_model.ipynb # Model architecture, training & evaluation script
├── .gitignore                          # Excluded files configuration
└── README.md                           # Project documentation
```

---

## 🛠️ Data Preprocessing Pipeline

1. **Image Resizing & Reshaping:** Images are read using OpenCV (`cv2`) in RGB format and resized to standard $32 \times 32$ pixels.
2. **Flattening:** Each $32 \times 32 \times 3$ image is flattened into a single feature array of 3,072 pixel intensity values.
3. **Data Storage:** Image features and associated target class labels are aggregated into a Pandas DataFrame and saved as `animal_pixels.csv`.
4. **Normalization:** Pixel values are scaled to the range $[0, 1]$ by dividing features by `255.0`.
5. **Label Encoding:** Target class labels are converted into one-hot encoded vectors across 90 output categories.

---

## 🏗️ Model Architecture

The deep learning model utilizes a 3-block Convolutional Neural Network architecture:

```text
Input (32x32x3) 
   │
   ├── Conv2D (32 filters, 3x3) ──> BatchNorm ──> MaxPool2D (2x2)
   ├── Conv2D (64 filters, 3x3) ──> BatchNorm ──> MaxPool2D (2x2)
   ├── Conv2D (128 filters, 3x3) ──> BatchNorm ──> MaxPool2D (2x2)
   │
   ├── Flatten
   ├── Dense (256 units, ReLU)
   ├── Dropout (0.5)
   └── Dense (90 units, Softmax) ──> Output Prediction
```

---

## 📊 Training & Performance Summary

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Epochs:** 50
* **Batch Size:** 32
* **Training Accuracy:** ~93%
* **Testing Accuracy:** ~67%

> **Note on Generalization:** The discrepancy between training and testing performance indicates overfitting. Because the dataset spans 90 distinct classes across ~5,400 total samples (an average of only ~60 images per class at $32 \times 32$ resolution), the feature capacity is relatively constrained for standard CNN training from scratch.

---

## 🚀 How to Run the Project

### Prerequisites

Ensure you have Python installed alongside the required packages:

```bash
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow
```

### Steps

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/animal-image-classification.git](https://github.com/YOUR_USERNAME/animal-image-classification.git)
   cd animal-image-classification
   ```

2. **Download Dataset:**
   * Download the **Animal Image Dataset** (90 classes) from Kaggle.
   * Place the extracted `animals` folder inside the project directory.

3. **Preprocess Images:**
   * Run `Making_animal_usable_file.ipynb` to construct `animal_pixels.csv`.

4. **Train and Evaluate:**
   * Run `animal_image_classification_model.ipynb` to train the CNN model and inspect the metrics.

---

## 💡 Future Improvements

* **Data Augmentation:** Apply random transformations (rotation, horizontal flips, zooming) to increase sample diversity and mitigate overfitting.
* **Transfer Learning:** Utilize pre-trained architectures such as **MobileNetV2** or **ResNet50** fine-tuned on ImageNet weights to boost test accuracy on high-class datasets.
* **Higher Resolution Inputs:** Increase spatial input dimensions beyond $32 \times 32$ to preserve fine visual features.
*
