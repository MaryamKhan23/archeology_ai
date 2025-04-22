import cv2
import os
import numpy as np

def load_images_from_folder(folder, img_size=(224, 224)):
    images, labels = [], []
    for label in ['yes', 'no']:  # two classes
        class_folder = os.path.join(folder, label)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img / 255.0
                images.append(img)
                labels.append(1 if label == 'yes' else 0)
    return np.array(images), np.array(labels)
