# Age Classification Coursework

## Project Overview

This repository contains a machine learning and computer vision coursework project for age group classification from cropped face images. The project uses a labelled dataset of face images and implements multiple classification approaches, including traditional feature-based models and a convolutional neural network.

The task is formulated as a four-class classification problem:

| Label | Age Group | Age Range |
| --- | --- | --- |
| `0` | Child | 0-12 |
| `1` | Young | 13-33 |
| `2` | Middle-Aged | 40-55 |
| `3` | Senior | 56+ |

## Features

- Preprocessing pipeline for loading labelled train and test image datasets.
- Support for images with varying resolutions through resizing during feature extraction.
- HOG feature extraction for classical machine learning models.
- Bag of Visual Words (BoVW) feature extraction using SIFT descriptors and MiniBatch K-Means.
- CNN-compatible image preprocessing.
- Age classification models implemented with:
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
- Model evaluation using accuracy and classification reports.
- OpenCV Haar cascade face detection for age prediction on personal image folders.
- Pre-trained model files stored in the `Models/` directory.

## Dataset

The coursework dataset contains cropped face images labelled by age group.

Dataset contents:

- Training set: 13,300 images in `train/`
- Testing set: 850 images in `test/`

Expected dataset files:

```text
CW_Dataset/
├── train/
│   ├── train_XXXX.jpg
│   └── train_labels.txt
└── test/
    ├── test_XXXX.jpg
    └── test_labels.txt
```

The label files contain one label per image. Images in the dataset vary in resolution, so preprocessing should account for this before training or inference.

The repository includes `CW_Dataset/CW_Dataset.zip`. Extract it before running training or testing code:

```bash
cd CW_Dataset
unzip CW_Dataset.zip
cd ..
```

## Requirements and Dependencies

The project is written in Python and uses common machine learning and computer vision libraries.

Main dependencies:

- Python 3.x
- NumPy
- OpenCV
- scikit-image
- scikit-learn
- PyTorch
- Matplotlib
- joblib
- Jupyter Notebook or JupyterLab, if using the provided notebooks

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/EEEDWARD1/Age-Classification-Coursework.git
cd Age-Classification-Coursework
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install numpy opencv-python scikit-image scikit-learn torch matplotlib joblib notebook
```

If using a CUDA-enabled PyTorch installation, install PyTorch using the command recommended for your platform from the official PyTorch installation guide.

Extract the dataset:

```bash
cd CW_Dataset
unzip CW_Dataset.zip
cd ..
```

## Usage

The repository includes notebooks for experimentation and Python modules in the `Code/` directory for preprocessing, feature extraction, model training, evaluation, and age detection.

### Running the Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open one of the provided notebooks:

- `work.ipynb`
- `copy_of_work.ipynb`
- `test_function.ipynb`

### Training an SVM Model

The following example trains an SVM classifier using HOG features:

```bash
python - <<'PY'
import sys
sys.path.append("Code")

from preprocessor import AgeDatasetPreprocessor
from feature_set_extractor import HOGExtractor
from ageSVM import AgeSVM

extractor = HOGExtractor(img_size=(128, 128))
preprocessor = AgeDatasetPreprocessor("CW_Dataset", extractor)

X_train, X_val, y_train, y_val = preprocessor.load_train(validation_ratio=0.1)

model = AgeSVM(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)
model.evaluate(X_val, y_val)
PY
```

### Training an MLP Model

The following example trains an MLP classifier using HOG features:

```bash
python - <<'PY'
import sys
sys.path.append("Code")

from preprocessor import AgeDatasetPreprocessor
from feature_set_extractor import HOGExtractor
from ageMLP import AgeMLP

extractor = HOGExtractor(img_size=(128, 128))
preprocessor = AgeDatasetPreprocessor("CW_Dataset", extractor)

X_train, X_val, y_train, y_val = preprocessor.load_train(validation_ratio=0.1)

model = AgeMLP(hidden_layers=(256, 128), max_iter=30)
model.fit(X_train, y_train)
model.evaluate(X_val, y_val)
PY
```

### Training a CNN Model

The following example trains the CNN classifier using CNN-formatted image tensors:

```bash
python - <<'PY'
import sys
sys.path.append("Code")

from preprocessor import AgeDatasetPreprocessor
from feature_set_extractor import CNNExtractor
from ageCNN import AgeCNN

extractor = CNNExtractor(img_size=(128, 128))
preprocessor = AgeDatasetPreprocessor("CW_Dataset", extractor)

X_train, X_val, y_train, y_val = preprocessor.load_train(validation_ratio=0.1)

model = AgeCNN(epochs=10, lr=0.001, batch_size=32)
model.fit(X_train, y_train)
model.evaluate(X_val, y_val)
PY
```

### Running Inference or Testing

The following example evaluates an SVM model on the test split:

```bash
python - <<'PY'
import sys
sys.path.append("Code")

from preprocessor import AgeDatasetPreprocessor
from feature_set_extractor import HOGExtractor
from ageSVM import AgeSVM

extractor = HOGExtractor(img_size=(128, 128))
preprocessor = AgeDatasetPreprocessor("CW_Dataset", extractor)

X_train, X_val, y_train, y_val = preprocessor.load_train(validation_ratio=0.1)
X_test, y_test = preprocessor.load_test()

model = AgeSVM(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)
model.evaluate(X_test, y_test)
PY
```

The `AgeDetection` class can be used with a trained model and extractor to run age prediction on images stored in a folder, such as `Personal_Dataset/` after it has been populated with images.

## Training and Models

Implemented model classes:

- `AgeSVM` in `Code/ageSVM.py`
  - Uses `sklearn.svm.SVC`
  - Applies `StandardScaler` before training and prediction
- `AgeMLP` in `Code/ageMLP.py`
  - Uses `sklearn.neural_network.MLPClassifier`
  - Applies `StandardScaler` before training and prediction
- `AgeCNN` in `Code/ageCNN.py`
  - Uses PyTorch
  - Contains three convolutional layers, max pooling, dropout, and fully connected layers
  - Uses cross-entropy loss and the Adam optimiser

Feature extraction classes:

- `HOGExtractor` in `Code/feature_set_extractor.py`
- `BoVWExtractor` in `Code/feature_set_extractor.py`
- `CNNExtractor` in `Code/feature_set_extractor.py`

Saved model files are stored in:

```text
Models/
├── cnn_best.joblib
├── mlp_best.joblib
└── svm_best.joblib
```

## Results and Performance

The model classes report:

- Accuracy
- Classification report from scikit-learn

No fixed benchmark metrics are documented in the original README. Run the notebooks or the example commands above to reproduce evaluation results for the available dataset split.

## Project Structure

```text
Age-Classification-Coursework/
├── Code/
│   ├── ageCNN.py
│   ├── ageDetection.py
│   ├── ageMLP.py
│   ├── ageSVM.py
│   ├── dataset_analysis.py
│   ├── feature_set_extractor.py
│   └── preprocessor.py
├── CW_Dataset/
│   └── CW_Dataset.zip
├── Models/
│   ├── cnn_best.joblib
│   ├── mlp_best.joblib
│   └── svm_best.joblib
├── Personal_Dataset/
│   └── populateMe.md
├── copy_of_work.ipynb
├── test_function.ipynb
├── work.ipynb
├── LICENSE
└── README.md
```

## Future Improvements

- Add a `requirements.txt` file for reproducible dependency installation.
- Add command-line training and inference scripts for each model type.
- Document reproducible experiment settings and final evaluation metrics.
- Add saved preprocessing pipelines alongside trained model files.
- Include confusion matrices and per-class analysis for model comparison.
- Improve dataset setup instructions if the extracted folder structure changes.

## Licence

This project includes a `LICENSE` file. Please review it before using or distributing the code or model files.

## Author

Eduard Teodor

GitHub: [EEEDWARD1](https://github.com/EEEDWARD1)

## DISCLOSURE
- This project utilised Codex to generate the README.md and the requirements.txt file.
- All implementation decisions, testing, integration, and final submitted work were reviewed, modified, and validated by the author.
- The project content, design decisions, experimentation, and coursework understanding remain the author's own work.
