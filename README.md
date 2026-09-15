# Hand Gesture Recognition

A deep learning project for recognizing **14 different hand gestures** from images and webcam input using a custom Convolutional Neural Network (CNN). The project focuses not only on high accuracy on the original dataset, but also on improving performance on real-world webcam images by incorporating additional webcam training data.

## Highlights

- **14-class** hand gesture classification
- Custom CNN built with TensorFlow/Keras
- Image preprocessing and data augmentation
- Stratified train/validation/test splitting
- Evaluation using accuracy and classification metrics
- Collection of a separate webcam dataset
- Domain adaptation using additional webcam training images
- Final real-time webcam inference script

## Dataset

The project uses the **HG14 (HandGesture14)** dataset from Kaggle.

- 14 gesture classes (`Gesture_0` – `Gesture_13`)
- 14,000 images in total
- 1,000 images per class
- Original dataset split used in the project: 9,800 training / 2,100 validation / 2,100 test images
- Images are resized to **128 × 128** before being passed to the CNN

Dataset source: [HG14 – HandGesture14 Dataset](https://www.kaggle.com/datasets/gulerosman/hg14-handgesture14-dataset)

The dataset itself is **not included** in this repository.

## Approach

The project was developed in several stages:

### 1. Baseline CNN
A custom CNN was trained on the HG14 dataset. The initial model achieved approximately **89.8% test accuracy**.

### 2. Improved CNN
The architecture was strengthened using additional convolutional layers, Batch Normalization, and Global Average Pooling. This substantially improved performance on the original HG14 test set.

### 3. Webcam Dataset
A separate dataset was collected using a webcam, with **40 images per gesture class**. This exposed a significant domain gap between the original dataset and real webcam images.

### 4. Domain Adaptation
Webcam training images were combined with the original HG14 training data. Webcam samples were oversampled during training so that the model received more exposure to the visual characteristics of the webcam domain.

## Final Results

The selected final model is `models/final_gesture_model.keras`.

| Evaluation Set | Accuracy |
|---|---:|
| HG14 Test Set | **97.76%** |
| Webcam Test Set | **47.62%** |

The lower webcam accuracy demonstrates the remaining **domain gap** between the curated HG14 images and independently captured webcam images. Importantly, the final model substantially outperforms the original webcam baseline of **11.90%**.

## Model

The final model is a custom CNN consisting of:

- Convolutional layers
- Batch Normalization
- Max Pooling
- Global Average Pooling
- Fully connected layers
- Dropout
- 14-class Softmax output

Input size: **128 × 128 × 3**  
Output: **14 gesture classes**

## Repository Structure

```text
hand-gesture-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Hand_Gesture_Recognition.ipynb
├── models/
│   └── final_gesture_model.keras
├── demo/
│   └── webcam_demo.py
└── results/
    └── project_results.txt
```

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset and training

Open the notebook in `notebooks/` and provide the HG14 dataset in the expected local/Colab path. The notebook contains the complete preprocessing, training, evaluation, and webcam-data experiments.

### 3. Webcam Demo

The demo uses OpenCV to access the webcam and the saved Keras model to predict one of the 14 gestures.

From the `demo/` directory, run:

```bash
python webcam_demo.py
```

Press **Q** to close the webcam window.

> The demo expects the model at `../models/final_gesture_model.keras`.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- scikit-learn
- Google Colab

## Important Notes

- The HG14 dataset and collected webcam images are intentionally excluded from the repository.
- The model is evaluated separately on HG14 and webcam test sets to show the effect of domain shift.
- The gesture labels remain `Gesture_0` through `Gesture_13`, matching the dataset.
- The project uses image-level stratified splitting; the evaluation should therefore not be interpreted as person-independent testing.

## Author

**Kushagra Srivastava**

GitHub: [@kushagra734](https://github.com/kushagra734)
