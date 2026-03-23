# Face Mask Detection & Person Identification using CNN

## Overview

This project has two models:

1. **Face Mask Detection** — classifies whether a person in an image is wearing a mask or not, with bounding boxes drawn around each detected face.
2. **Person Identification** — identifies *who* a person is even when they are wearing a mask, using a MobileNetV2-based model trained on both masked and unmasked images of the same people.

---

## Dataset

**AFDB — Asian Face Database** (stored locally under `data/`)

```
data/
├── AFDB_face_dataset/        — 461 people, unmasked images (organized by person name)
└── AFDB_masked_face_dataset/ — 525 people, masked images (organized by person name)
```

- 442 people appear in both datasets (used for person identification)
- Only people with at least 3 masked images are used for training (210 people)
- Each person folder contains multiple images named e.g. `0_0_yangmi_0014.jpg`

---

## Project Files

| File | Description |
|------|-------------|
| `face_mask_detection_local.ipynb` | Detects faces and classifies each as mask / no mask |
| `person_identification.ipynb` | Identifies who the person is, with or without a mask |
| `person_identification_model.h5` | Saved trained MobileNetV2 model (210 people) |
| `person_label_classes.npy` | Label encoder classes for person identification |
| `haarcascade_frontalface_default.xml` | OpenCV face detector |

---

## Tech Stack

- Python 3.9
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- Scikit-learn
- Matplotlib

---

## Model 1 — Face Mask Detection (`face_mask_detection_local.ipynb`)

### How it works

1. Images from both dataset folders are loaded and resized to **128 × 128**
2. Converted to RGB, normalized to **[0, 1]**
3. Labels: `1` = mask, `0` = no mask
4. 80/20 train/test split

### Architecture (Keras Sequential)

```
Conv2D(32, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu) → MaxPooling2D
Flatten
Dense(128, relu) → Dropout(0.5)
Dense(64, relu)  → Dropout(0.5)
Dense(2, sigmoid)
```

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Epochs**: 5
- **Test Accuracy**: ~92%

### Prediction

- Detects all faces in an image using OpenCV Haar cascade
- Crops each face and runs it through the CNN
- Draws a **green box** (Mask) or **red box** (No Mask) around each face with confidence %
- Falls back to classifying the full image if no face is detected

---

## Model 2 — Person Identification (`person_identification.ipynb`)

### How it works

- Trained on both masked and unmasked images of the same person simultaneously
- The model learns to recognize each person regardless of whether they are wearing a mask
- Uses **MobileNetV2** pretrained on ImageNet as the feature extractor

### Architecture

```
MobileNetV2 (pretrained, ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(256, relu)
  → Dropout(0.4)
  → Dense(210, softmax)   ← 210 people
```

### Training

- **Phase 1** (5 epochs): MobileNetV2 base frozen, only top layers trained
- **Phase 2** (5 epochs): Top 30 layers of base unfrozen, fine-tuned at learning rate 1e-5
- **Batch size**: 32
- **Validation split**: 10%

### Prediction

- Detects face in the image using OpenCV Haar cascade
- Crops the face and passes it through MobileNetV2
- Outputs **top 3 predictions** with person name and confidence %
- Draws bounding box with predicted name on the image

---

## How to Run

### 1. Install dependencies

```bash
pip install numpy matplotlib opencv-python pillow scikit-learn tensorflow kaggle jupyter
```

### 2. Quick Predict (model already trained)

Open `person_identification.ipynb` in VSCode or Jupyter and run only the first 3 cells:

- Cell 1: loads the saved model
- Cell 2: defines the `identify_person` function
- Cell 3: asks for an image path and runs prediction

Example image path:
```
/full/path/to/data/AFDB_masked_face_dataset/yangmi/0_0_0.jpg
```

### 3. Retrain from scratch

Run all cells in `person_identification.ipynb` including the Training section at the bottom.

---

## Results

- **Mask Detection**: ~92% test accuracy
- **Person Identification**: correctly identifies the person as the top prediction even when wearing a mask (e.g. yangmi identified at 8.2% confidence under full mask coverage — low confidence is expected since the mask covers most facial features)
