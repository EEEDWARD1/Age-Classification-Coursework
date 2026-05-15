import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans

class HOGExtractor:
  def __init__(self, img_size=(128,128)):
    self.img_size = img_size

  def extract(self, img):
    img = cv2.resize(img, self.img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return hog(
          gray,
          orientations=9,
          pixels_per_cell=(8,8),
          cells_per_block=(1,1)
      )

class BoVWExtractor:
  def __init__(self, img_size=(128,128), num_clusters=100):
    self.img_size = img_size
    self.sift = cv2.SIFT_create()
    self.kmeans = None
    self.num_clusters = num_clusters
    self.is_bovw = True

  def extract_descriptors(self, img):
    img = cv2.resize(img, self.img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, desc = self.sift.detectAndCompute(gray, None)
    return desc

  def build_vocab(self, descriptors_list):
    all_desc = np.vstack([d for d in descriptors_list if d is not None])

    self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters)
    self.kmeans.fit(all_desc)

  def to_histogram(self, descriptors):
    hist = np.zeros(self.num_clusters)

    if descriptors is None:
      return hist

    clusters = self.kmeans.predict(descriptors)

    for c in clusters:
      hist[c] += 1

    return hist

class CNNExtractor:
  def __init__(self, img_size=(128,128)):
    self.img_size = img_size

  def extract(self, img):
    img = cv2.resize(img, self.img_size)
    img / 255.0
    img = np.transpose(img, (2,0,1))
    return img