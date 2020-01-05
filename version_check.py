import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_v2_behavior()
#%%
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
# a = tf.constant('conssss')
# if __name__ == '__main__':
#     print(device_lib.list_local_devices())
#     print(tf.__version__)
#     print(a.numpy())

# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
#
# with tf.Session() as sess:
#     print (sess.run(c))
print(tf.test.is_gpu_available())