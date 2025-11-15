
import numpy as np
#import jax.numpy as jnp
import matplotlib.pyplot as plt



def fig_init(shape=(1, 1), dims=(8.0, 7.0), fontsize=7.0, squeeze=True, **kwargs):
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    plt.rc('font', size=fontsize)
    cm_per_in = 2.54
    fig, ax = plt.subplots(shape[0], shape[1], squeeze=squeeze,  # nrows, ncols
                           figsize=(shape[1]*dims[0]/cm_per_in, shape[0]*dims[1]/cm_per_in),
                           layout='constrained', **kwargs)
    return fig, ax



def complex_to_rgb(z, vec_real=(0, 1, 1), vec_imag=(1, 0, 1), normalize_each=True):
    z = np.asarray(z + 0j)
    vec_real = np.asarray(vec_real)
    vec_imag = np.asarray(vec_imag)

    re, im = z.real, z.imag

    if normalize_each:
        re = (re - re.min()) / (re.max() - re.min() + 1e-12)
        im = (im - im.min()) / (im.max() - im.min() + 1e-12)
    else:
        # Joint normalization to a common scale
        m = max(abs(re).max(), abs(im).max())
        re = 0.5*re/m + 0.5
        im = 0.5*im/m + 0.5

    # RGB [shape(x), 3]
    rgb = np.multiply.outer(re, vec_real) + np.multiply.outer(im, vec_imag)
    return np.clip(rgb, 0., 1.)
