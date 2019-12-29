import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

datadir = r'E:\py_file_pycharm\tf_keras\PetImages'
categories = ['Dog','Cat']

img_size = 50

training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                # print(e)
                pass

create_training_data()
print(len(training_data))

