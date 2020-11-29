# import tensorflow as tf
import numpy as np
# boston_housing = tf.keras.datasets.boston_housing
# (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
boston_housing = np.load(r'.\boston_housing.npz')
print(boston_housing.files)
