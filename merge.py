from PIL import Image
import numpy as np

r = np.asarray(Image.open('red.png'))
g = np.asarray(Image.open('green.png'))
b = np.asarray(Image.open('blue.png'))

combined = np.zeros(shape=(r.shape[0], r.shape[1], 3), dtype=r.dtype)
combined[:, :, 0] = r
combined[:, :, 1] = g
combined[:, :, 2] = b

new = Image.fromarray(np.uint8(combined))
new.save('sample_better.png')