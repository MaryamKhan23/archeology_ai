import numpy as np
import cv2

def add_noise(img, scale=0.1):
    noise = np.random.normal(loc=0, scale=scale, size=img.shape)
    return np.clip(img + noise, 0, 1)

def blur_image(img):
    return cv2.GaussianBlur(img, (11, 11), 0)

def occlude(img):
    h, w = img.shape[:2]
    x_start, y_start = h // 4, w // 4
    img[x_start:x_start+50, y_start:y_start+50] = 0  # Add a black square
    return img
