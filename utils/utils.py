import matplotlib.pyplot as plt
import numpy as np


def plot_generated_image(generator, x, random_dim):
    generated_image = generator.predict(
        np.random.randint(0, x.shape[0], size=[1, random_dim])
    )
    dims = generated_image.shape
    plt.pcolor(generated_image.reshape(dims[2], dims[3]), cmap=plt.cm.gray)
    plt.save("plot.png")
