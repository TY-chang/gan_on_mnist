from keras import initializers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam


def make_generator_model(random_dim) -> Sequential:
    generator = Sequential()
    generator.add(Dense(32 * 5 * 5, input_dim=random_dim))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((32, 5, 5)))
    generator.add(
        Conv2DTranspose(
            32, 3, strides=(2, 2), padding="valid", output_padding=None, use_bias=False
        )
    )
    generator.add(LeakyReLU(0.2))
    generator.add(
        Conv2DTranspose(
            64, 5, strides=(2, 2), padding="valid", output_padding=None, use_bias=False
        )
    )
    generator.add(LeakyReLU(0.2))
    generator.add(
        Conv2DTranspose(
            1, 4, strides=(1, 1), padding="valid", output_padding=None, use_bias=False
        )
    )
    return generator


# discriminator building
def make_discriminator_model() -> Sequential:
    discriminator = Sequential()
    discriminator.add(
        Conv2D(
            64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            input_shape=(1, 28, 28),
            kernel_initializer=initializers.RandomNormal(stddev=0.02),
        )
    )
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation="sigmoid"))
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0003))
    return discriminator
