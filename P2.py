import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
 (x_train, y_train), (x_test, y_test) = mnist.load_data()
 x_train = x_train.reshape((60000, 28 * 28)).astype('float32') / 255
 x_test = x_test.reshape((10000, 28 * 28)).astype('float32') / 255
 y_train = tf.keras.utils.to_categorical(y_train, 10)
 y_test = tf.keras.utils.to_categorical(y_test, 10)
 model = models.Sequential()
 model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
 model.add(layers.Dense(64, activation='relu'))
 model.add(layers.Dense(10, activation='softmax'))
 model.compile(optimizer='adam',
 loss='categorical_crossentropy',
 metrics=['accuracy'])

random_index = np.random.randint(0 , x_test.shape[0])
random_image = x_test[random_index]
random_label = np.argmax(y_test[random_index])
random_image = random_image.reshape(1, 28 * 28)
predictions = model.predict(random_image)
predicted_class =np.argmax(predictions)

plt.imshow(random_image.reshape(28, 28), cmap= 'gray')
plt.title(f'True label: {random_label}, Predicted: {predicted_class}')
plt.axis('off')
plt.show()
