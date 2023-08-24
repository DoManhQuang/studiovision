# import the necessary packages
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.image import rgb_to_grayscale
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import os, sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def load_images(DIRECTORY, CATEGORIES, target_size=(224, 224), mode="binary", grayscale=False):
    print("[INFO] loading images...")
    print("DIRECTORY : ", DIRECTORY)
    print("CATEGORIES : ", CATEGORIES)
    data = []
    labels = []
    imgs_name = []
    CATEGORIES = os.path.join(DIRECTORY)
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=target_size)
            image = img_to_array(image)
            if grayscale == True:
                image = rgb_to_grayscale(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(category)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    if mode != "binary":
        labels = to_categorical(labels)
    return np.array(data, dtype="float32"), np.array(labels)