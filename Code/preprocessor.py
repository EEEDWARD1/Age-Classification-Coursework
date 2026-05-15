import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import random

random.seed(10)

class AgeDatasetPreprocessor:
  def __init__(self, base_path, extractor):
    self.base_path = base_path
    self.extractor = extractor

  def _load_labels(self, split):
    label_path = os.path.join(self.base_path, split, f"{split}_labels.txt")

    label_dict = {}

    with open(label_path, "r") as f:
      for line in f:
        filename, label = line.strip().split()
        label_dict[filename] = int(label)

    return label_dict

  def _load_dataset(self, split, limit_ratio=None):
    folder = os.path.join(self.base_path, split)
    label_dict = self._load_labels(split)

    items = list(label_dict.items())

    random.shuffle(items)

    if limit_ratio is not None:
      if limit_ratio > 1:
        limit_ratio = 1
        print("limit ratio cannot be greater than 1, defaulted to 1")
      subset_size = int(len(items) * limit_ratio)
      items = items[:subset_size]

    data = []
    labels = []

    if hasattr(self.extractor, "is_bovw"):
      descriptors_list = []

      for filename, label in items:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
          continue

        desc = self.extractor.extract_descriptors(img)
        descriptors_list.append(desc)
        labels.append(label)

      if split == "train":
          self.extractor.build_vocab(descriptors_list)

      for desc in descriptors_list:
          hist = self.extractor.to_histogram(desc)
          data.append(hist)
    else:
      for filename, label in items:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
          continue
        processed = self.extractor.extract(img)
        data.append(processed)
        labels.append(label)

    return np.array(data), np.array(labels)

  def load_train(self, validation_ratio=0.1, limit_ratio=None):
    X, y = self._load_dataset("train", limit_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=validation_ratio,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val

  def load_test(self, limit_ratio=None):
    return self._load_dataset("test", limit_ratio)