import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100

# Load CIFAR-10 dataset
(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
x_train_10, x_test_10 = x_train_10 / 255.0, x_test_10 / 255.0


# Build the CNN model
def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and evaluate on CIFAR-10
model_10 = create_model(10)
model_10.fit(x_train_10, y_train_10, epochs=10, validation_data=(x_test_10, y_test_10))

# Load CIFAR-100 dataset
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
x_train_100, x_test_100 = x_train_100 / 255.0, x_test_100 / 255.0

# Retrain the model on CIFAR-100
model_100 = create_model(100)
model_100.fit(x_train_100, y_train_100, epochs=10, validation_data=(x_test_100, y_test_100))
