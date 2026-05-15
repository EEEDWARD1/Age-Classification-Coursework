import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

LABELS = {0: 'Child', 1: 'Young', 2: 'Middle', 3: 'Senior'}

class AgeDetection:
    def __init__(self, model, extractor):
        self.model = model
        self.extractor = extractor
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def __call__(self, path):
        images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        selected = random.sample(images, 4)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for ax, filename in zip(axes, selected):
            img = cv2.imread(os.path.join(path, filename))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8, minSize=(50,50))
            ax.imshow(img_rgb)

            if len(faces) == 0:
                processed = self.extractor.extract(img)
                label = LABELS[self.model.predict_image(processed)]
                ax.set_title(f"{label} (no face)", color='red', fontsize=12)
            else:
                for (x, y, w, h) in faces:
                    pad_x, pad_y = 50, 80
                    x1 = max(x - pad_x, 0)
                    y1 = max(y - pad_y, 0)
                    x2 = min(x + w + pad_x, img.shape[1])
                    y2 = min(y + h + pad_y, img.shape[0])

                    face = img[y1:y2, x1:x2]
                    processed = self.extractor.extract(face)
                    label = LABELS[self.model.predict_image(processed)]

                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                            linewidth=2,
                                            edgecolor='cyan',
                                            facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, label, color='cyan',
                            fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.show()