# IN3060 Computer Vision Coursework - Age Group Detection Pipeline

> Individual coursework submission for **IN3060: Computer Vision** at **City, University of London**.  
> Instructed by **Dr. Giacomo Tarroni**.

---

## Project Title & Academic Context

This repository contains an Age Group Detection (AGD) computer vision pipeline for classifying cropped facial images into four numerical target groups:

| Class ID | Target Group | Coursework Age Range |
| --- | --- | --- |
| `0` | Child | `0-12` |
| `1` | Young | `13-33` |
| `2` | Middle-Aged | `40-55` |
| `3` | Senior | `56+` |

The project processes the supplied coursework dataset of **13,300 training images** and **850 testing images**. The pipeline compares classical feature engineering and learned representation approaches, including Histogram of Oriented Gradients (HOG), SIFT Bag of Visual Words (BoVW), Support Vector Machines (SVM), Multi-Layer Perceptrons (MLP), and a PyTorch convolutional neural network (CNN).

To reduce boundary ambiguity between adjacent age categories, the target definition excludes ages **34-39**, creating a deliberate separation between the `Young` and `Middle-Aged` groups.

---

## Pipeline Architecture & Quantitative Performance Metrics

The final notebook, [`copy_of_work.ipynb`](copy_of_work.ipynb), evaluates three selected model pipelines:

1. **HOG + Linear SVM**  
   Cropped face images are resized to `128x128`, converted to grayscale, represented using HOG descriptors, standardised with `StandardScaler`, and classified with `sklearn.svm.SVC`.

2. **SIFT/BoVW + MLP**  
   Images are resized to `128x128`, SIFT descriptors are extracted with OpenCV, descriptors are quantised into a `100`-cluster MiniBatch K-Means visual vocabulary, and BoVW histograms are classified with `sklearn.neural_network.MLPClassifier`.

3. **CNN Image Tensor Classifier**  
   Images are resized to `128x128`, converted into channel-first tensors, and classified with a PyTorch CNN containing three convolutional blocks, max pooling, dropout, and fully connected output layers.

### Final Model Comparison

| Rank | Pipeline | Feature Representation | Classifier | Key Configuration | Validation Accuracy | Test Accuracy | Model Artifact | Model Size | Classification Speed |
| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| **1** | **CNN Tensor Classifier** | RGB tensor, `3 x 128 x 128` | PyTorch CNN | Filters `(32, 64, 128)`, dropout `0.5`, `10` epochs, Adam optimiser | `70.00%` | **`72.94%`** | `Models/cnn_best.joblib` | `98 MB` | Not benchmarked in notebook; expected to be GPU-friendly and used for final personal-image inference |
| 2 | HOG + Linear SVM | HOG, `9` orientations, `8x8` pixels per cell | Linear SVM | `kernel="linear"`, `C=1`, standardised features | `63.08%` | `66.00%` | `Models/svm_best.joblib` | `106 MB` | Not benchmarked in notebook; HOG extraction plus SVM prediction on CPU |
| 3 | SIFT/BoVW + MLP | SIFT visual-word histogram | MLP | `100` visual clusters, hidden layer `(256,)`, `250` max iterations | `42.18%` | `41.06%` | `Models/mlp_best.joblib` | `856 KB` | Not benchmarked in notebook; SIFT/BoVW extraction is the dominant preprocessing stage |

### Best Performing Model

The best performing final model is the **CNN Tensor Classifier**, achieving **72.94% test accuracy** on the 850-image coursework test split.

The final personal-image prediction workflow is exposed through the `AgeDetection(path)` loop in `test_function.ipynb`. It loads images from the custom personal test bed, detects faces using OpenCV Haar cascades, crops detected face regions with padding, preprocesses them through the selected extractor, and displays predicted age-group labels on the input images.

---

## Repository Organization

```text
Age-Classification-Coursework/
├── Code/
│   ├── ageCNN.py                  # PyTorch CNN architecture, training, evaluation, prediction
│   ├── ageDetection.py            # Personal-image face detection and age-label visualisation
│   ├── ageMLP.py                  # MLP classifier wrapper with scaling and evaluation
│   ├── ageSVM.py                  # SVM classifier wrapper with scaling and evaluation
│   ├── dataset_analysis.py        # Dataset image-size analysis helpers
│   ├── feature_set_extractor.py   # HOG, SIFT/BoVW, and CNN preprocessing extractors
│   └── preprocessor.py            # Dataset loading, label parsing, split handling, feature extraction
├── CW_Dataset/
│   └── CW_Dataset.zip             # Coursework dataset archive
├── Models/
│   ├── cnn_best.joblib            # Best-performing CNN model artifact
│   ├── mlp_best.joblib            # Best MLP/BoVW model artifact
│   └── svm_best.joblib            # Best HOG/SVM model artifact
├── Personal_Dataset/
│   └── populateMe.md              # Placeholder for custom personal evaluation images
├── copy_of_work.ipynb             # Final training, tuning, and evaluation notebook
├── test_function.ipynb            # Validation and AgeDetection(path) execution environment
├── work.ipynb                     # Development notebook
├── requirements.txt               # Python dependency list
├── LICENSE                        # Current repository license file
└── README.md                      # Portfolio documentation
```

### Key Navigation Notes

- `Code/` contains the core model components, hyperparameter experiments, feature extraction classes, and image preprocessing scripts.
- `CW_Dataset/` contains the supplied facial data archive used for training and held-out testing.
- `Personal_Dataset/` is reserved for custom evaluation images used by the final `AgeDetection(path)` demonstration loop.
- `Models/` stores the fully trained, serialized model artifacts used for reproducible validation and inference.
- `test_function.ipynb` is the lightweight validation notebook for executing the submitted prediction function.

---

## Reproducibility

### Environment

The project was developed in Python using Jupyter/Google Colab style workflows. Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

Core dependencies:

```text
matplotlib
numpy
opencv-python
scikit-image
scikit-learn
torch
```

### Dataset Preparation

The expected extracted dataset structure is:

```text
CW_Dataset/
├── train/
│   ├── train_XXXX.jpg
│   └── train_labels.txt
└── test/
    ├── test_XXXX.jpg
    └── test_labels.txt
```

The notebook workflow extracts `CW_Dataset.zip`, reads label files, performs stratified training/validation splitting, and evaluates the final models on the full held-out test set.

---

## Open-Source Licensing Compliance Recommendation

This project uses widely adopted open-source scientific Python and computer vision libraries:

| Library | Usage in Project | Typical License Family |
| --- | --- | --- |
| NumPy | Array representation and numerical operations | BSD-style |
| Matplotlib | Visualisation and prediction display | Matplotlib license / PSF-style |
| OpenCV | Image loading, SIFT, Haar cascade face detection | Apache 2.0 |
| scikit-image | HOG feature extraction | BSD-style |
| scikit-learn | SVM, MLP, K-Means, scaling, metrics, splits | BSD-style |
| PyTorch | CNN model implementation and tensor inference | BSD-style |

For a public portfolio repository intended for recruiters and industry review, a highly permissive license is recommended. The **MIT License** is a strong fit because it allows use, copying, modification, distribution, sublicensing, and commercial use while preserving clear warranty and liability disclaimers.

### Recommended MIT License Summary

```text
MIT License Summary

Permission is granted to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of this software, provided that the copyright
notice and permission notice are included in all copies or substantial portions
of the software.

The software is provided "AS IS", without warranty of any kind, express or
implied, including but not limited to warranties of merchantability, fitness for
a particular purpose, and non-infringement. The authors are not liable for any
claim, damages, or other liability arising from use of the software.
```

---

## Academic Integrity & Generative AI Usage Log

All core vision pipeline modeling, feature descriptor logic, custom classification array definitions, and evaluation logic are my own original individual engineering work.

Generative AI tools were strictly limited to repository housekeeping, .gitignore optimization, and structuring Markdown formatting templates based on historical repository structures. No machine learning engine layers, image processing calculations, or model weighting configurations were generated or synthesized by AI.

This repository is published as an academic portfolio artifact. It is intended to document the completed coursework implementation, experimental process, and engineering decisions, while preserving the distinction between original coursework code and later repository presentation improvements.

---

## Author

**Eduard Teodor**  
GitHub: [EEEDWARD1](https://github.com/EEEDWARD1)
