import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.datasets import cifar10

# Define the generator
def build_generator(latent_dim, img_shape):
    inputs = Input(shape=(latent_dim,))
    x = Dense(128 * 16 * 16, activation='relu')(inputs)
    x = Reshape((16, 16, 128))(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)
    generator = Model(inputs, x)
    return generator

# Define the discriminator
def build_discriminator(img_shape):
    inputs = Input(shape=img_shape)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs, outputs)
    return discriminator

# Define the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    return gan

# Load and preprocess the dataset
(train_images, _), (_, _) = cifar10.load_data()
train_images = (train_images - 127.5) / 127.5

# Define hyperparameters
latent_dim = 100
img_shape = train_images.shape[1:]

# Build and compile the models
generator = build_generator(latent_dim, img_shape)
discriminator = build_discriminator(img_shape)
gan = build_gan(generator, discriminator)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Training loop
epochs = 10000
batch_size = 64
for epoch in range(epochs):
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_images = train_images[idx]
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise)
    
    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} - D loss: {d_loss[0]}, G loss: {g_loss}")
