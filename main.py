import numpy as np
from scipy.signal import convolve2d as conv2
from skimage import color, restoration, io
import PIL.Image as image

rng = np.random.default_rng()

img = io.imread('sample.png')[:,:,:3]
img[:, :, 1] = 0
img[:, :, 2] = 0
astro = color.rgb2gray(img)
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

deconvolved[:, :] *= 256
new = image.fromarray(deconvolved)
new = new.convert("L")
new.save('red.png')

img = io.imread('sample.png')[:,:,:3]
img[:, :, 0] = 0
img[:, :, 2] = 0
astro = color.rgb2gray(img)
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

deconvolved[:, :] *= 256
new = image.fromarray(deconvolved)
new = new.convert("L")
new.save('green.png')

img = io.imread('sample.png')[:,:,:3]
img[:, :, 0] = 0
img[:, :, 1] = 0
astro = color.rgb2gray(img)
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

deconvolved[:, :] *= 256
new = image.fromarray(deconvolved)
new = new.convert("L")
new.save('blue.png')