import numpy as np
from scipy.signal import convolve2d as conv2
from skimage import color, restoration, io
from PIL import Image
import numpy as np

rng = np.random.default_rng()

img = io.imread('sample.png')[:,:,:3]
img[:, :, 1] = 0
img[:, :, 2] = 0
astro = color.rgb2gray(img)
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)
newr = deconvolved

img = io.imread('sample.png')[:,:,:3]
img[:, :, 0] = 0
img[:, :, 2] = 0
astro = color.rgb2gray(img)
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)
newg = deconvolved

img = io.imread('sample.png')[:,:,:3]
img[:, :, 0] = 0
img[:, :, 1] = 0
astro = color.rgb2gray(img)
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)
newb = deconvolved

rgbArray = np.zeros((img.shape[0],img.shape[1],3), 'uint8')
rgbArray[..., 0] = newr*256
rgbArray[..., 1] = newg*256
rgbArray[..., 2] = newb*256
img = Image.fromarray(rgbArray)
img.save('enhanced.jpeg')