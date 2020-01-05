import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import matplotlib.pyplot as plt
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(c))
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.6705)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
NAME = "Cats-vs-dogs-CNN"

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

x = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('Y.pickle', 'rb'))
print(y)
# print(type(x),type(y))
# print(x[0].shape)
# # plt.imshow(x[1], cmap=plt.get_cmap('binary'))
# # plt.show()
#
X = x/255.0
# #
# print(X.shape[1:])
# model = Sequential()
# model.add(Conv2D(256, (3, 3), input_shape = X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Flatten())
#
# model.add(Dense(64))
# model.add(Activation('relu'))
#
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1,
#           callbacks=[tensorboard])

# model.save('DOG_CAT.model')
new_model = tf.keras.models.load_model('DOG_CAT.model') # saved model
predictions = new_model.predict(X)
print(predictions)
# num_index = 0
# for i in range(100):
# #     num_index = i
#     print(np.argmax(predictions[num_index]))
#
    # plt.imshow(x_test[num_index],cmap=plt.get_cmap('binary'))
#     plt.show()

