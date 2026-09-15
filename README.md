# Hand Gesture Recognition

A deep learning project for classifying 14 different hand gestures using the HG14 (HandGesture14) dataset and a custom convolutional neural network.

## Project Overview

The project covers:
- Dataset preparation and stratified train/validation/test splitting
- Image preprocessing and augmentation
- Custom CNN development
- Model evaluation
- Webcam data collection
- Domain adaptation through additional webcam training data
- Final webcam inference

## Dataset

The project uses the **HG14 (HandGesture14)** dataset containing 14 gesture classes.

Dataset source: https://www.kaggle.com/datasets/gulerosman/hg14-handgesture14-dataset

The dataset is not included in this repository.

## Final Model

The selected model is `final_gesture_model.keras`.

| Dataset | Accuracy |
|---|---:|
| HG14 test set | **97.76%** |
| Webcam test set | **47.62%** |

The difference demonstrates the domain gap between the original dataset and images captured using a webcam.

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

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- scikit-learn
- Google Colab

## Notes

The original HG14 dataset and collected webcam images are intentionally not included in this repository.
