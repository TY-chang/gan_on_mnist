from keras.layers import Input
from keras.models import Model

from data.data_loader import load_data
from model.model import make_discriminator_model, make_generator_model
from trainer.train import train


def main():
    randomDim = 100
    x, y, _, _ = load_data()
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    ganInput = Input(shape=(randomDim,))
    discriminator.trainable = False
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0003))

    dLosses, gLosses = train(
        gan, generator, discriminator, x, randomDim, epochs=10, batch_size=16
    )


if __name__ == "__main__":

    main()
