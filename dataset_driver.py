import tensorflow as tf

import tensorflow_datasets as tfds

import numpy as np

#ds = tfds.load('emnist', split='train', as_supervised=True)
#np.save('EMNIST', tfds.as_numpy(ds))

ds = tfds.load('kmnist', split='train')
np.save('KMNIST.npy', tfds.as_numpy(ds))

ds = tfds.load('fashion_mnist', split='train')
np.save('FMNIST.npy', tfds.as_numpy(ds))