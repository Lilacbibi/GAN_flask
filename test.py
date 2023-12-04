import numpy as np
import os
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from keras import layers
import tensorflow as tf

gan_model = keras.models.load_model('gan.h5', compile=False)

num_classes = 3
latent_dim =128

def noise_with_class_label(class_number):
    one_hot_labels = keras.utils.to_categorical(class_number, num_classes)[None,:]
    random_latent_vectors = tf.random.normal(shape=(1, latent_dim))
    random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1)
    return random_vector_labels

image = gan_model.predict(noise_with_class_label(1))    


plt.imshow(np.squeeze(image))
plt.show()