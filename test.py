#TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from time import time


# 数据加载
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_test.shape, y_test.shape)
print(x_test[0],y_test[0])
plt.imshow(x_test[0])
plt.show()

# 网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(36, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2)



# 测试
# loss, accuracy = model.evaluate(x_test[0:1000], y_test[0:1000])
# print('\ntest loss', loss)
# print('accuracy', accuracy)
print(model.predict(x_test[0:1]))
# t1 = time() - startTime1
# # 打印运行时间
# print('使用cpu花的时间：', t1)
