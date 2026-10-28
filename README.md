# 🩻 Chest X-ray Abnormality Detection and Classification Using Deep Learning

## 📌 Project Overview

This project focuses on identifying abnormalities in chest X-ray images using deep learning and computer vision. It combines image classification and object detection techniques to analyze chest radiographs and localize potential abnormalities.

The project uses the **VinBigData Chest X-ray Abnormalities Detection** dataset and explores transfer learning with pretrained convolutional neural networks.

Two complementary approaches are implemented:

- **Image Classification:** EfficientNet-B0 and ResNet50 to predict abnormality labels from chest X-ray images.
- **Object Detection:** YOLOv8n to identify and localize abnormalities using bounding boxes.

The project also explores model evaluation, bounding-box visualization, and interpretability techniques such as Grad-CAM.

> **Disclaimer:** This is an educational and research project, not a clinically validated diagnostic system. Predictions should not be used as a substitute for professional medical assessment.

## 🎯 Objectives

- Preprocess DICOM chest X-ray images for deep learning.
- Improve image quality using normalization, CLAHE, and filtering.
- Apply transfer learning for multi-label chest X-ray classification.
- Train YOLOv8n to localize abnormalities with bounding boxes.
- Evaluate models using appropriate classification and detection metrics.
- Visualize predictions and analyze model failure cases.
- Explore Grad-CAM heatmaps to understand image-classification predictions.

## 📂 Dataset

**Dataset:** [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

The dataset contains chest X-ray images in DICOM format and radiologist-provided annotations.

The annotations include:

- `image_id`: Image identifier
- `class_name`: Abnormality category
- `class_id`: Numeric category identifier
- `rad_id`: Radiologist identifier
- `x_min`, `y_min`, `x_max`, `y_max`: Bounding-box coordinates

The dataset contains multiple abnormality categories, along with `No finding` annotations.

For the initial experiments, 3,000 DICOM images were selected for preprocessing. Augmented versions were also generated for training experiments.

**Note:** The 3,000 original images and their augmented versions can produce approximately 6,000 image files. This does not mean that 6,000 independent original X-rays were collected.

The dataset is hosted on Kaggle and is not included in this repository.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Object Detection:** Ultralytics YOLOv8
- **Image Processing:** OpenCV, Pillow, pydicom
- **Data Processing:** NumPy, Pandas
- **Model Evaluation:** Scikit-learn
- **Visualization:** Matplotlib
- **Environment:** Kaggle Notebooks

## ⚙️ Image Preprocessing

The original DICOM images are converted into processed PNG images using the following pipeline:

1. **DICOM loading:** Read image pixel data using `pydicom`.
2. **Grayscale conversion:** Convert color images to grayscale when necessary.
3. **Resizing:** Resize images to 512 × 512 pixels.
4. **Min-max normalization:** Scale pixel values to a normalized range.
5. **CLAHE enhancement:** Improve local contrast.
6. **Gaussian blur:** Reduce high-frequency noise.
7. **Median filtering:** Reduce certain types of image noise.
8. **Border cropping and padding:** Crop a small border and restore the original processed dimensions.
9. **Data augmentation:** Apply random horizontal flipping, small rotations, and contrast variations.

The preprocessing pipeline aims to improve consistency across images while retaining useful anatomical features.

**Important:** For rigorous experiments, confirm that bounding-box coordinates are transformed consistently with the corresponding images. Also consider DICOM photometric interpretation and verify that preprocessing preserves clinically meaningful image details.

## 🧠 Model 1: Chest X-ray Classification

### EfficientNet-B0

EfficientNet-B0 is used as the primary classification architecture. It is initialized with pretrained ImageNet weights, and its final classification layer is replaced to match the target labels.

Transfer learning allows the model to reuse previously learned visual features and adapt them to chest X-ray analysis.

### Classification Pipeline

1. Load the processed images and their corresponding multi-label targets.
2. Split the data into training and validation sets.
3. Apply training-time augmentation.
4. Load the pretrained EfficientNet-B0 model.
5. Replace the final classification layer.
6. Train using `BCEWithLogitsLoss`.
7. Optimize parameters using AdamW.
8. Evaluate validation predictions using classification metrics.
9. Save the best-performing model checkpoint.

ResNet50 is also available as an alternative backbone for comparative experiments.

### Classification Metrics

- **Accuracy:** Measures the proportion of correctly predicted label decisions under a specified threshold.
- **Precision:** Measures how many positive predictions are correct.
- **Recall:** Measures how many actual positive labels are detected.
- **F1-score:** Balances precision and recall.
- **ROC-AUC:** Measures the ability to rank positive examples above negative examples.

Because this is a multi-label task, accuracy should be clearly defined—for example, per-label accuracy or exact-match accuracy. Macro F1-score is useful for examining performance across classes, particularly when class frequencies differ.

## 🎯 Model 2: Chest X-ray Object Detection

### YOLOv8n

YOLOv8n is used for object detection and bounding-box localization.

Unlike classification, which predicts image-level labels, object detection predicts both the class and spatial location of an abnormality.

### Detection Pipeline

1. Prepare the processed chest X-ray images.
2. Convert bounding-box annotations into YOLO format.
3. Organize images and labels into training and validation directories.
4. Configure the dataset using a YAML file.
5. Load pretrained YOLOv8n weights.
6. Train the detector at an image size of 512 × 512.
7. Evaluate the model on validation images.
8. Visualize predicted bounding boxes and confidence scores.

### Detection Metrics

- **Precision:** Proportion of predicted detections that are correct.
- **Recall:** Proportion of ground-truth objects successfully detected.
- **mAP@0.5:** Mean Average Precision at an IoU threshold of 0.5.
- **mAP@0.5:0.95:** Mean Average Precision averaged across IoU thresholds from 0.5 to 0.95.
- **Box loss:** Measures bounding-box localization error during training.
- **Classification loss:** Measures errors in predicted object classes.
- **DFL loss:** Distribution Focal Loss used for bounding-box regression in YOLOv8.

### Bounding-Box Visualization

The trained detector can be used to visualize:

- Predicted abnormality locations
- Predicted class names
- Detection confidence scores
- Correct detections
- Missed abnormalities and false-positive detections

The YOLO training labels must use the correct class mapping and valid bounding-box coordinates. `No finding` annotations do not represent actual bounding boxes and should not be treated as object detections.

## 🔍 Model Interpretability

Interpretability is an important part of the project because medical-image models should be examined beyond their headline metrics.

### Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) can generate heatmaps highlighting image regions that contribute to a selected classification prediction.

These heatmaps can be overlaid on chest X-rays to inspect whether the classifier focuses on plausible anatomical regions.

Grad-CAM provides qualitative evidence about model behavior; it does not establish clinical correctness or explain every aspect of a prediction.

### Detection Analysis

For YOLOv8n, predicted bounding boxes can be compared against ground-truth annotations to investigate:

- Correctly localized abnormalities
- False-positive detections
- Missed abnormalities
- Incorrect predicted classes
- Low-confidence predictions

## 📊 Experimental Results

Model performance should be reported using the results obtained from the final, reproducible validation run.

| Model | Task | Evaluation metrics |
|---|---|---|
|EfficientNet-B0|Multi-label classification| Accuracy, Precision, Recall, Macro F1, ROC-AUC |
| ResNet50 | Multi-label classification | Accuracy, Precision, Recall, Macro F1, ROC-AUC |
| YOLOv8n | Object detection | Precision, Recall, mAP@0.5, mAP@0.5:0.95 |

For classification, report the threshold used to convert predicted probabilities into binary labels. For detection, report the confidence threshold and IoU settings when comparing predictions.

Avoid comparing classification F1 directly with detection mAP, as they measure different tasks.

## 💾 Model Checkpoints

The trained models are saved locally in the Kaggle working directory.

Example checkpoint paths:

```text
/kaggle/working/best_model.pth
/kaggle/working/vindr_yolo/weights/best.pt
/kaggle/working/vindr_yolo/weights/last.pt
```

- `best_model.pth`: Classification checkpoint, if saved using this filename.
- `best.pt`: Best YOLOv8n checkpoint according to the training validation criterion.
- `last.pt`: Final YOLOv8n training checkpoint.

The classification checkpoint may contain only the model's state dictionary. To reload it, recreate the same model architecture and classification head before loading the saved weights.

Kaggle's working directory is temporary. Download or export important checkpoints before the notebook session ends.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/PG123-ops/Chest_X-ray_abnormalities.git
cd Chest_X-ray_abnormalities
```

### 2. Install Dependencies

```bash
pip install torch torchvision
pip install ultralytics
pip install pydicom opencv-python pillow
pip install pandas numpy scikit-learn matplotlib tqdm
```

### 3. Configure the Dataset

Download the dataset from Kaggle and configure the paths in your notebook or Python scripts.

Example:

```python
INPUT_DIR = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/"
OUTPUT_DIR = "/kaggle/working/pre_processed_3000/"
```

These paths are specific to the Kaggle environment and must be adjusted when running locally.

### 4. Run the Pipeline

Run the notebook cells in order:

1. Dataset loading and annotation inspection
2. DICOM preprocessing
3. Image augmentation
4. Classification dataset preparation
5. EfficientNet-B0 or ResNet50 training
6. Classification evaluation and checkpoint saving
7. YOLO dataset preparation
8. YOLOv8n training and evaluation
9. Prediction visualization and failure-case analysis
10. Grad-CAM visualization for classification


This is a suggested structure; adapt it to match the files actually present in your repository.

## 🔮 Future Improvements

- Fine-tune different depths of the pretrained classification backbone.
- Investigate class imbalance using weighted loss functions and threshold tuning.
- Compare ResNet50 with other additional architectures.
- Improve object detection performance through better annotation validation and hyperparameter tuning.
- Generate Grad-CAM heatmaps for correctly and incorrectly classified samples.
- Build a consolidated evaluation report for classification and detection.
- Develop an inference interface for educational demonstrations.
- Evaluate generalization on a separate, patient-independent test set.

## ⚠️ Limitations

- Initial experiments use a limited subset of the available dataset.
- Chest X-ray abnormalities may be visually subtle or overlapping.
- Class imbalance can substantially affect classification and detection performance.
- Radiologist annotations may vary and may not cover every visible abnormality.
- Validation results may not generalize to different hospitals, equipment, or patient populations.
- The models have not been established as safe or effective for clinical diagnosis.

## 👨‍💻 Author

**Puspendu Ghosh**

M.Sc. Data Science

**Keywords:** Chest X-ray, Medical Image Analysis, Deep Learning, PyTorch, EfficientNet-B0, ResNet50, YOLOv8n, Transfer Learning, Multi-label Classification, Object Detection, Grad-CAM, Computer Vision.
