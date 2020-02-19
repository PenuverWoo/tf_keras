import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)
# plt.imshow(x_train[0], cmap=plt.get_cmap('binary'))
# plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()                              # trainning processing

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
print(model.supports_masking )
model.add(tf.keras.layers.Dense(200, activation=tf.nn.relu))
print(model.supports_masking )
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
print(model.supports_masking )

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# model.save('epix_num_reader.model')

# new_model = tf.keras.models.load_model('epix_num_reader.model') # saved model
# predictions = new_model.predict([x_test])
# num_index = 0
# for i in range(100):
#     num_index = i
#     print(np.argmax(predictions[num_index]))
#
#     plt.imshow(x_test[num_index],cmap=plt.get_cmap('binary'))
#     plt.show()