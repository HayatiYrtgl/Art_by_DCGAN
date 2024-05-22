# Art_by_DCGAN
This code sets up and trains a GAN to generate artwork images using Keras and TensorFlow, defining and compiling discriminator and generator networks, and saving results.

let's break down and analyze the provided code.

### 1. Importing Required Libraries
```python
import keras.src.backend
from keras.layers import *
from keras.models import *
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.utils import image_dataset_from_directory, img_to_array, set_random_seed
from keras.metrics import Mean
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```
This code imports necessary libraries from Keras and TensorFlow for building and training the GAN. It also imports utility functions for dataset handling, metrics for monitoring, and plotting.

### 2. Setting Latent Dimension
```python
# latent dim
latent_dim = 128
```
Defines the size of the latent space, which is the input to the generator. 

### 3. Loading and Preprocessing the Dataset
```python
# dataset
dataset = image_dataset_from_directory("../DATASETS/wikiart/wikiart", label_mode=None, image_size=(128, 128), batch_size=16)
dataset = dataset.map(lambda x: x / 255.0)
```
Loads the dataset from the specified directory and resizes images to 128x128 pixels. The images are normalized to the range [0, 1].

### 4. Defining the Discriminator Model
```python
# discriminator section
discriminator = Sequential()
# Input layer
discriminator.add(Input((128, 128, 3)))
# conv1
discriminator.add(Conv2D(filters=64, kernel_size=4, padding="same", strides=(2, 2)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.2))
# conv2
discriminator.add(Conv2D(filters=128, kernel_size=4, padding="same", strides=(2, 2)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.2))
# conv3
discriminator.add(Conv2D(256, 4, padding="same", strides=(2, 2)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.2))
# conv 4
discriminator.add(Conv2D(256, 4, padding="same", strides=(2, 2)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.2))
discriminator.add(Flatten())
discriminator.add(Dropout(0.2))
discriminator.add(Dense(1, activation="sigmoid"))
discriminator.summary()
```
Defines the discriminator as a CNN with multiple convolutional layers, LeakyReLU activations, and dropout for regularization. The final output layer uses a sigmoid activation to classify the input images as real or fake.

### 5. Defining the Generator Model
```python
# generator
generator = Sequential()
# input
generator.add(Input((latent_dim,)))
# generator add
generator.add(Dense(8 * 8 * 256))
# reshape
generator.add(Reshape((8, 8, 256)))
# convt 1
generator.add(Conv2DTranspose(64, kernel_size=4, padding="same", strides=2))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.2))
# convt 2
generator.add(Conv2DTranspose(128, kernel_size=4, padding="same", strides=2))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.2))
# convt3
generator.add(Conv2DTranspose(256, kernel_size=4, padding="same", strides=2))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.2))
generator.add(Conv2DTranspose(512, kernel_size=4, padding="same", strides=2))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.2))
# last
generator.add(Conv2D(3, kernel_size=4, padding="same", strides=1, activation="sigmoid"))
# generator summary
generator.summary()
```
Defines the generator as a neural network that starts with a dense layer followed by reshaping and multiple transposed convolutional layers. Each transposed convolutional layer is followed by batch normalization and LeakyReLU activation. The final layer uses a sigmoid activation to output images with pixel values in the range [0, 1].

### 6. Defining the GAN Class
```python
# class for gan model
class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = Mean(name="d_loss")
        self.g_loss_metric = Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vector = tf.random.normal([batch_size, latent_dim])
        generated_images = self.generator(random_latent_vector)
        combined = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        random_latent_vector = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vector))
            g_loss = self.loss_fn(misleading, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}
```
Defines the `GAN` class, inheriting from `keras.Model`. The class integrates the generator and discriminator, specifies optimizers and loss functions, and defines the training step. The `train_step` method includes generating fake images, combining them with real images, and computing the loss for both the discriminator and the generator.

### 7. Callback for Monitoring Training Progress
```python
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed_generator = set_random_seed(42)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = keras.src.backend.random_normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save("created_images/generated_img_%03d_%d.jpg" % (epoch, i))
```
Defines a callback to save generated images at the end of each epoch for monitoring the training progress.

### 8. Training the GAN
```python
epochs = 300  # In practice, use ~100 epochs

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=5, latent_dim=latent_dim)]
)
generator.save("../devam_eden_trains/generator_artist.h5")
discriminator.save("../devam_eden_trains/discriminator_artist.h5")
```
Compiles the GAN model, specifying the optimizers and loss function, and trains the GAN for 300 epochs. The `GANMonitor` callback is used to save generated images at the end of each epoch. Finally, the generator and discriminator models are saved.

### Summary

This code sets up and trains a GAN for generating artwork images. The discriminator and generator networks are defined and compiled into a GAN model, which is then trained using a dataset of artwork images. The progress is monitored using a callback that saves generated images at the end of each epoch.
