import numpy as np
from PIL import Image
import math
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

sigma = 1 / (2 * math.log(2))
image = Image.open("lena.png")
image = np.array(image)
ans = gaussian_filter(image, sigma = 1 / (2 * sigma ** 2))

plt.imshow(ans)
plt.show()

