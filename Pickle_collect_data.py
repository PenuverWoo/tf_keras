import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pickle
import matplotlib.pyplot as plt

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
                # print(new_array.shape)
                training_data.append([new_array, class_num])
            except Exception as e:
                # print(e)
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

X = np.array(x).reshape(-1, img_size, img_size,1)
Y = np.array(y)
# plt.imshow(X[0])
# plt.show()

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('Y.pickle', 'wb')
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
print(X[1])