# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os, sys
import cv2
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vision.core.preprocess_input import prep_input


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    # auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


# color_mode One of "grayscale", "rgb", "rgba". Default: "rgb"
def load_images(DIRECTORY=None, CATEGORIES=None, target_size=(224, 224), mode="load_path", path_imgs=None, 
                lb_imgs=None, auto_brightness=False, color_mode='rgb', models_name='resnet'):
    print("[INFO] loading images...")
    data, labels = [], []
    if mode == "load_path":
        for category in CATEGORIES:
            path = os.path.join(DIRECTORY, category)
            for img in os.listdir(path):
                data.append(img)
                labels.append(category)
            
    elif mode == "load_imgs":
        for img, lb in zip(path_imgs, lb_imgs):
            
            path = os.path.join(DIRECTORY, lb)
            img_path = os.path.join(path, img)
            
            image = load_img(img_path, target_size=target_size, color_mode=color_mode)
            image = img_to_array(image)
            image = prep_input(models_name, image)

            if auto_brightness:
                image, _, _ = automatic_brightness_and_contrast(image)

            data.append(image)
            labels.append(lb)
    else:
        return "Error load images"
            
    if mode == "load_imgs":
        return np.array(data, dtype="float32"), np.array(labels)
    return np.array(data), np.array(labels)