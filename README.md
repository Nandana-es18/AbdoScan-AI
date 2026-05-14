# AbdoScan-AI  
## Multimodal Abdominal Organ Segmentation and Anomaly Detection with Explainable AI

AbdoScan-AI is an advanced deep learning–based medical imaging framework developed for automated abdominal organ segmentation and anomaly detection from CT and MRI scans. The system integrates multimodal medical image analysis, volumetric segmentation, anomaly localization, and Explainable AI (XAI) into a unified clinical decision support platform.

The proposed framework utilizes a SegResNet-based architecture for accurate multi-organ segmentation and incorporates Grad-CAM–based explainability to improve transparency and interpretability of model predictions.

---

# Overview

Medical image segmentation plays a critical role in disease diagnosis, treatment planning, and radiological analysis. Manual segmentation of abdominal organs is labor-intensive, time-consuming, and highly dependent on expert radiologists.

AbdoScan-AI addresses these challenges by providing:
- Automated multi-organ segmentation
- Liver and kidney anomaly detection
- Explainable AI visualizations
- Interactive web-based medical image analysis

The framework supports both CT and MRI modalities, enabling robust and generalized abdominal image analysis across diverse imaging conditions.

---

# Key Features

- Multimodal abdominal image analysis (CT & MRI)
- Automated multi-organ segmentation
- Liver and kidney anomaly detection
- Explainable AI using Grad-CAM
- SegResNet-based volumetric segmentation
- Interactive web application interface
- Segmentation mask visualization
- Diagnostic report generation
- Downloadable analysis outputs
- Modular and scalable architecture

---

# Segmented Organs

The framework performs segmentation of the following abdominal organs:

- Liver
- Left Kidney
- Right Kidney
- Spleen
- Pancreas
- Gallbladder
- Stomach

---

# Problem Statement

Accurate abdominal organ segmentation is essential for clinical diagnosis and treatment planning. However, existing approaches face several limitations:

- Manual segmentation is time-consuming and operator-dependent
- Most systems support only single-modality imaging
- Many frameworks lack anomaly detection capabilities
- Limited explainability reduces clinical trust
- Existing solutions often lack deployable end-to-end systems

AbdoScan-AI aims to overcome these limitations by integrating multimodal segmentation, anomaly detection, and Explainable AI into a unified medical imaging framework.

---

# Objectives

The primary objectives of the project include:

- Automated segmentation of major abdominal organs
- Accurate liver and kidney anomaly detection
- Support for both CT and MRI modalities
- Integration of Explainable AI for interpretability
- Development of a web-based clinical visualization platform
- Generation of structured diagnostic outputs

---

# Datasets

The system is trained and evaluated using publicly available benchmark medical imaging datasets.

| Dataset | Modality | Purpose |
|----------|----------|----------|
| AMOS | CT & MRI | Multi-organ segmentation |
| LiTS | CT | Liver tumor segmentation |
| KiTS | CT | Kidney tumor segmentation |

### Dataset Highlights
- AMOS contains 600+ annotated CT and MRI scans
- LiTS provides liver and liver tumor annotations
- KiTS contains kidney and kidney tumor annotations
- Data is provided in NIfTI/DICOM formats

---

# System Architecture

The architecture of AbdoScan-AI is designed as a layered and modular pipeline to ensure scalability, maintainability, and efficient processing of volumetric medical images.

## 1. Client Tier
The frontend interface enables users to:
- Upload CT/MRI scans
- Select imaging modality
- Visualize segmentation outputs
- Access diagnostic reports

Technologies:
- HTML5
- CSS3
- JavaScript

---

## 2. API Tier
The FastAPI backend acts as the communication bridge between the frontend and AI modules.

Responsibilities:
- Request handling
- Routing and validation
- File management
- Asynchronous inference processing

Technologies:
- FastAPI
- Uvicorn

---

## 3. Processing Tier
This layer performs modality-aware preprocessing and image standardization.

Operations include:
- NIfTI/DICOM parsing
- Spatial resampling
- Image resizing
- Intensity normalization
- ROI extraction
- Temporary storage management

### CT Preprocessing
- Hounsfield Unit clipping and normalization

### MRI Preprocessing
- Z-score intensity normalization

---

## 4. AI Inference Tier
This is the core intelligence layer of the framework.

The preprocessed scan is passed through:
- SegResNet segmentation network
- Anomaly detection pipeline
- Explainability module

Outputs generated:
- Organ segmentation masks
- Tumor localization
- Grad-CAM heatmaps

---

## 5. Output & Reporting Tier
The final layer generates:
- Segmentation overlays
- Clinical-style reports
- Explainability visualizations
- Radiomics-based outputs

Results are returned to the frontend for interactive visualization and download.

---

# Model Architecture

The proposed framework utilizes **SegResNet**, a residual learning–based segmentation architecture optimized for volumetric medical imaging.

### Advantages of SegResNet
- Improved gradient flow
- Better feature propagation
- Stable training performance
- Efficient 3D medical segmentation
- Enhanced contextual feature extraction

The architecture combines:
- Encoder-decoder segmentation pipeline
- Residual learning blocks
- Multi-class segmentation using softmax activation

---

# Explainable AI (XAI)

To improve model interpretability, the framework integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM generates visual attention maps highlighting:
- Organ boundaries
- Clinically relevant regions
- Abnormal anatomical structures

### Benefits
- Improves clinical transparency
- Enhances trust in AI predictions
- Supports explainable diagnostic analysis

---

# Preprocessing Pipeline

All input scans undergo a standardized preprocessing pipeline before inference.

### Pipeline Steps
- File parsing
- Spatial normalization
- Intensity normalization
- Resampling
- Resizing
- ROI extraction
- Data augmentation

### Data Augmentation Techniques
- Random rotation
- Flipping
- Scaling
- Noise addition

These operations improve:
- Model robustness
- Generalization capability
- Cross-modality consistency

---

# Training Configuration

| Parameter | Value |
|-----------|------|
| Optimizer | AdamW |
| Learning Rate | 1 × 10⁻⁴ |
| Weight Decay | 1 × 10⁻⁵ |
| Loss Function | Dice + Cross Entropy |
| Batch Size | 1 |
| Epochs | 200 |
| Patch Size | 96 × 96 × 96 |

---

# Quantitative Results

The framework was evaluated using Dice Similarity Coefficient (DSC).

| Dataset | Dice Score |
|----------|------------|
| AMOS (CT + MRI) | 93.2% |
| LiTS (Liver) | 93.0% |
| KiTS (Kidney) | 92.0% |

### Performance Highlights
- Accurate multi-organ segmentation
- Reliable anomaly localization
- Strong multimodal generalization
- Stable convergence during training
- Clinically interpretable predictions

---

# Technology Stack

## Frontend
- HTML5
- CSS3
- JavaScript

## Backend
- Python
- FastAPI
- Uvicorn

## Deep Learning & Imaging
- PyTorch
- MONAI
- Scikit-Image
- NumPy
- NiBabel

## Data Formats
- NIfTI (.nii.gz)
- DICOM

---

# Web Application Features

- Drag-and-drop scan upload
- CT/MRI modality selection
- Interactive segmentation visualization
- Explainability heatmap rendering
- Organ-specific report generation
- Downloadable outputs

---

# Project Structure

```bash
ABDOSCAN-AI/
│
├── frontend/
│   ├── templates/
│   ├── static/
│
├── backend/
│   ├── app.py
│   ├── routes/
│   ├── preprocessing/
│
├── models/
│   ├── segresnet/
│   ├── trained_weights/
│
├── datasets/
│
├── outputs/
│   ├── segmentation_results/
│   ├── heatmaps/
│   ├── reports/
│
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Nandana-es18/ABDOSCAN-AI.git
cd ABDOSCAN-AI
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows
```bash
venv\Scripts\activate
```

### Linux / macOS
```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

```bash
uvicorn app:app --reload
```

Open in browser:

```bash
http://127.0.0.1:8000
```

---

# Future Scope

- Full-body medical image segmentation
- Additional organ support
- PACS integration
- Cloud deployment for hospital access
- Federated learning integration
- Real-time clinical deployment

---

# Research Contributions

- Multimodal CT & MRI segmentation framework
- SegResNet-based abdominal analysis
- Integrated anomaly detection
- Grad-CAM–based Explainable AI
- End-to-end web-based clinical platform

---

# Publications

### ICTEST 2026
3rd International Conference on Trends in Engineering Systems and Technologies

---


# References

- MONAI Framework
- SegResNet
- TransUNet
- Grad-CAM
- AMOS Dataset
- LiTS Dataset
- KiTS Dataset

---

# Conclusion

AbdoScan-AI presents a comprehensive multimodal medical imaging framework that combines:
- Automated abdominal organ segmentation
- Anomaly detection
- Explainable AI
- Interactive clinical visualization

The proposed system demonstrates strong segmentation accuracy, reliable anomaly localization, and improved interpretability, making it suitable for next-generation AI-assisted clinical decision support systems.

---
