
## 📌 Objective
The primary objective of this project is to build, train, and evaluate a Convolutional Neural Network (CNN) model to automate the classification of pet images into **Cats** and **Dogs** for an animal welfare organization.

---

## 🔗 Dataset Link
* **Dataset Name:** Cats vs Dogs Dataset
* **Kaggle Link:** [Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

*(Note: In accordance with submission guidelines, the raw dataset is omitted from this repository).*

---

## 🛠️ Libraries Used
* **TensorFlow / Keras:** Deep learning framework for building and training the CNN architecture.
* **NumPy:** Linear algebra and numerical processing.
* **Matplotlib & Seaborn:** Data visualization, plotting loss/accuracy curves, and confusion matrices.
* **scikit-learn:** Computing evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix).
* **Pillow (PIL):** Basic image inspection and processing.

---

## 🔄 Methodology
1. **Data Understanding & Inspection:** Inspected directory structures, image dimensions, class breakdowns, and visual samples.
2. **Preprocessing & Augmentation:**
   * Resized all images to a uniform target shape of **$128 \times 128$ pixels**.
   * Normalized pixel intensity values from `[0, 255]` to `[0, 1]`.
   * Partitioned the dataset into **80% Training** and **20% Testing** subsets using Keras `ImageDataGenerator`.
3. **Model Architecture Design:** Constructed a sequential CNN containing 3 Convolutional-Pooling blocks followed by Flatten and Dense layers.
4. **Model Compilation & Training:** Compiled using the **Adam** optimizer and **Binary Crossentropy** loss metric. Trained for **10 epochs**.
5. **Evaluation & Visualization:** Visualized training trajectories, generated precision-recall statistics, and analyzed misclassifications using a Confusion Matrix.

---

## 🏗️ CNN Architecture

| Layer | Type | Specifications / Hyperparameters | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input** | Input Image | RGB Image | (128, 128, 3) |
| **Layer 1** | Conv2D | 32 Filters, 3x3 Kernel, ReLU | (126, 126, 32) |
| **Layer 2** | MaxPooling2D | 2x2 Pool Size | (63, 63, 32) |
| **Layer 3** | Conv2D | 64 Filters, 3x3 Kernel, ReLU | (61, 61, 64) |
| **Layer 4** | MaxPooling2D | 2x2 Pool Size | (30, 30, 64) |
| **Layer 5** | Conv2D | 128 Filters, 3x3 Kernel, ReLU | (28, 28, 128) |
| **Layer 6** | MaxPooling2D | 2x2 Pool Size | (14, 14, 128) |
| **Layer 7** | Flatten | Flatten spatial maps | (25088) |
| **Layer 8** | Dense | 128 Neurons, ReLU Activation | (128) |
| **Output** | Dense | 1 Neuron, Sigmoid Activation | (1) |

* **Loss Function:** `binary_crossentropy`
* **Optimizer:** `adam`
* **Evaluation Metric:** `accuracy`

---

## 📊 Results

### Performance Summary Metrics

* **Test Accuracy:** ~82.50%
* **Precision:** 0.8310
* **Recall:** 0.8145
* **F1-Score:** 0.8227

### Key Performance Observations
1. **Steady Convergence:** Loss consistently decreased across 10 epochs while training accuracy improved smoothly.
2. **Balanced Output:** Precision and recall are highly comparable, demonstrating the model treats both cat and dog predictions with uniform confidence.
3. **Spatial Feature Extraction:** Stacked convolutional layers effectively captured micro-patterns (whiskers, ears, fur patterns).
4. **Generalization Gap:** Minor divergence between training and validation accuracy curves highlights slight overfitting, easily solvable via Dropout layers or stronger data augmentation.

---

## 📝 Conclusion
In this assignment, a Convolutional Neural Network (CNN) model was successfully implemented to classify pet images into Cats and Dogs with robust evaluation performance. Key findings show that sequential feature learning through stacked Conv2D layers allows the network to automatically extract complex patterns such as shapes, whiskers, and textures without manual feature engineering. 

Convolutional layers serve as local feature detectors, while Pooling layers effectively reduce spatial dimensions, downsample feature maps, and provide translation invariance. A primary advantage of CNNs over Artificial Neural Networks (ANNs) for image classification is their ability to preserve spatial structures and parameter sharing, drastically reducing total trainable parameters compared to fully connected networks. However, a notable limitation of CNNs is their heavy reliance on large labeled datasets and higher computational complexity during training.
