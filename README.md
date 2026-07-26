# Skin Lesion Boundary Segmentation using U-Net

## Project Overview

This project implements a deep learning-based skin lesion segmentation system using a U-Net architecture. The objective is to accurately segment lesion boundaries from dermoscopy images by generating a binary mask for each input image.

This project was completed as part of the Image Analysis and Computer Vision course.

---

## Objective

- Perform automatic skin lesion boundary segmentation.
- Train a baseline U-Net model on dermoscopy images.
- Evaluate segmentation performance using Dice Coefficient and Intersection over Union (IoU).

---

## Dataset

The dataset consists of paired dermoscopy images and corresponding binary lesion masks.

Dataset Summary

- Total image-mask pairs: 2594
- Training samples: 2075
- Validation samples: 519
- Image size: 512 × 512

---

## Project Structure

```text
skin-lesion-segmentation/

├── AI_Log.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── unet.py
│
├── outputs/
│   ├── checkpoints/
│   ├── evaluation/
│   ├── plots/
│   └── predictions/
│
├── src/
│   ├── check_dataset.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
```

---

## Model

- Architecture: U-Net
- Input channels: 3
- Output channels: 1
- Loss Function: BCEWithLogitsLoss + Dice Loss
- Optimizer: Adam
- Learning Rate: 0.0001
- Batch Size: 2
- Epochs Trained: 15 (Early Stopping)
- Learning Rate Scheduler: ReduceLROnPlateau
- Early Stopping Patience: 5

---

## Training Results

Best Model

| Metric | Value |
|--------|-------|
| Best Epoch | 10 |
| Validation Loss | 0.6523 |
| Validation Dice | 0.7225 |
| Validation IoU | 0.6297 |

---

## Evaluation

The best checkpoint was evaluated on the validation dataset.

Final Evaluation Metrics

| Metric | Value |
|--------|-------|
| Validation Loss | 0.6523 |
| Validation Dice | 0.7225 |
| Validation IoU | 0.6297 |

Prediction examples are available in:

```
outputs/evaluation/predictions/
```

---

## Features

- Data preprocessing pipeline
- Custom PyTorch dataset loader
- U-Net implementation
- Combined BCE + Dice loss
- Model checkpointing
- Resume training support
- Learning rate scheduling
- Early stopping
- Validation metric tracking
- Automatic prediction visualization
- Evaluation pipeline

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py
```

### Evaluate the best model

```bash
python src/evaluate.py
```

---

## Results

The trained U-Net model successfully segments skin lesion boundaries and provides reliable baseline segmentation performance.

Final validation performance:

- Dice Coefficient: **0.7225**
- IoU: **0.6297**

---

## Future Improvements

Potential improvements include:

- Attention U-Net
- U-Net++
- Data augmentation
- Dice + Focal Loss
- Hyperparameter tuning
- Test-time augmentation

---

## Author

Preethika Lethakula

Image Analysis and Computer Vision