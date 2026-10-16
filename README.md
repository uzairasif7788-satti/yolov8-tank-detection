# 🚁 YOLOv8 Tank Detection

A specialized computer vision project for detecting military tanks in custom aerial and remote sensing imagery using the state-of-the-art **YOLOv8** object detection framework.

---

## 🎯 Objectives
- **Object Detection:** Accurately locate and bound military tanks in complex aerial environments.
- **Model Training & Execution:** Train a custom YOLOv8 model locally using custom-annotated datasets and workflow management.
- **Performance Evaluation:** Analyze metrics including Box Loss, Objectness/Classification Loss (`cls_loss`), and mAP50.

---

## 🛠️ Technologies Used
- **Python** (Core Scripting)
- **Ultralytics YOLOv8** (Object Detection Framework)
- **PyTorch** (Deep Learning Backend)
- **OpenCV** (Image Processing & Computer Vision)
- **Roboflow** (Dataset Annotation & Management)

---

## 📂 Project Workflow
1. **Dataset Preparation:** Curated, organized, and annotated custom aerial imagery datasets exclusively for tank detection.
2. **Dataset Configuration:** Configured `data.yaml` with appropriate paths and class mappings for training and validation splits.
3. **Model Training:** Conducted preliminary and secondary training iterations using local hardware execution to train the tank detection model.
4. **Error & Bias Analysis:** Evaluated detection errors, handled dataset imbalances, and optimized local processing workflows.
5. **Inference & Testing:** Tested unseen images and validated bounding box predictions using optimized weights (`best.pt`).

---

## 📊 Dataset Specifications
- **Target Class:**
  - `0`: `tank`
- **Format:** YOLO format (`.txt` files containing normalized bounding box coordinates)

---

## ⚙️ Model Architecture & Training Details
- **Architecture:** YOLOv8 Object Detection Framework
- **Loss Functions:** 
  - Box Regression Loss (CIoU)
  - Classification Loss (`cls_loss`)
  - Distribution Focal Loss (DFL)
- **Execution Environment:** Local hardware training with optimized batch workflows and checkpoint management.

---
