import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images for the autoencoder
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Define the Stacked Autoencoder
def build_autoencoder(input_dim):
    # Encoder
    encoder = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu')
    ])
    
    # Decoder
    decoder = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(128,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # Autoencoder
    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Build and train the autoencoder
autoencoder, encoder = build_autoencoder(784)
history = autoencoder.fit(
    x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test)
)

# Test reconstruction
decoded_imgs = autoencoder.predict(x_test)

# Visualize original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()

# Print training history
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
