import os
import cv2

DATASET_PATH = '/content/CW_Dataset'

def analyse_image_sizes(split):
  folder = os.path.join(DATASET_PATH, split)

  sizes = []
  failed = 0

  for file in os.listdir(folder):
    if file.endswith(".jpg"):
      path = os.path.join(folder, file)

      img = cv2.imread(path)
      if img is None:
        failed += 1
        continue

      h, w = img.shape[:2]
      sizes.append((w,h))
  return sizes, failed

def print_stats(name, sizes, failed):
  print(f"\n==== {name} Set ====")
  print(f"Total images: {len(sizes)}")
  print(f"Failed {failed}")

  widths = [s[0] for s in sizes]
  heights = [s[1] for s in sizes]

  print("\n Min & MaX")
  print(f"Width: {min(widths)} -> {max(widths)}")
  print(f"Height: {min(heights)} -> {max(heights)}")