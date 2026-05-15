Age Group Detection - Coursework Dataset
=============================================

This dataset contains cropped face images labelled by age group.

Labels:
  0 = Child (ages 0-12)
  1 = Young (ages 13-33)
  2 = Middle-Aged (ages 40-55)
  3 = Senior (ages 56+)

Training set: 13300 images in train/
Testing set:  850 images in test/

Files:
  train/train_XXXX.jpg  - Training images
  train/train_labels.txt - One label (0-3) per line
  test/test_XXXX.jpg    - Testing images
  test/test_labels.txt  - One label (0-3) per line

Note: Images in this dataset vary in resolution.
You should account for this in your preprocessing pipeline.
