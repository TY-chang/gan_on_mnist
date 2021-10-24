import numpy as np
from utils.utils import plot_generated_image


def train(gan, generator, discriminator, x, random_dim, epochs=10, batch_size=16):
    batch_count = x.shape[0] / batch_size
    dLosses = []
    gLosses = []
    for _ in range(1, epochs + 1):
        for _ in range(int(batch_count)):
            # step 1.1, generate the image from noise/ normal distribution ---Terrible data
            noise_fake = np.random.normal(0, 1, size=[batch_size, random_dim])
            fake_image = generator.predict(noise_fake)
            # step 1.2, get real image from training data --- Good data
            real_image = x[np.random.randint(0, x.shape[0], size=batch_size)]

            # mix data from step1, and label them respectively
            X = np.concatenate([real_image, fake_image])
            Y = np.zeros(2 * batch_size)
            Y[:batch_size] = 0.9

            # Step 2, train the discriminator alone
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, Y)
            # fit(X, Y, batch_size = batch_size , epochs = 4, shuffle = True)

            # Step 3, train generator AND  discriminator non-update
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            Y = np.ones(batch_size)
            discriminator.trainable = False
            gloss = gan.train_on_batch(
                noise, Y
            )  # (noise, Y, batch_size = batch_size , epochs = 4, shuffle = True)
            plot_generated_image()
        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)
        return dLosses, gLosses
