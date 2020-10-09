import numpy as np
from PIL import Image
import math
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

sigma = 1 / (2 * math.log(2))
image = Image.open("lena.png")
image = np.array(image)


#==================================================================
#3-1

# ans = gaussian_filter(image, sigma = 1 / (2 * sigma ** 2))
# plt.subplot(1, 2, 1)
# plt.title('Original')
# plt.imshow(image, cmap = 'gray')
# plt.subplot(1, 2, 2)
# plt.title('Gaussian filter')
# plt.imshow(ans, cmap = 'gray')
# plt.show()
#==================================================================
#3-2

# plt.subplot(1, 3, 1)
# plt.title('Original')
# plt.imshow(image, cmap = 'gray')

# I_x = []

# for row in image:
#     i_x = np.convolve(row,[1,0,-1],'same')
#     I_x.append(i_x)
# I_x = np.array(I_x)

# plt.subplot(1, 3, 2)
# plt.title('Ix')
# plt.imshow(I_x.reshape(512,512), cmap = 'gray')



# image_y = image.transpose()

# I_y = []

# for row in image_y:
#     i_y = np.convolve(row,[1,0,-1],'same')
#     I_y.append(i_y)
# I_y = np.array(I_y)
# I_y = I_y.transpose()

# plt.subplot(1, 3, 3)
# plt.title('Iy')
# plt.imshow(I_y.reshape(512,512), cmap = 'gray')
# plt.show()

#==================================================================

#3-3

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap = 'gray')

def one_conv(image):
    I_x = []

    for row in image:
        i_x = np.convolve(row,[1,0,-1],'same')
        I_x.append(i_x)
    I_x = np.array(I_x)

    image_y = image.transpose()
    I_y = []

    for row in image_y:
        i_y = np.convolve(row,[1,0,-1],'same')
        I_y.append(i_y)
    I_y = np.array(I_y)
    I_y = I_y.transpose()

    return I_x.reshape(512, 512), I_y.reshape(512, 512)

ans = gaussian_filter(image, sigma = 1 / (2 * sigma ** 2))

gaussian_x, gaussian_y = one_conv(ans)
lena_x, lena_y = one_conv(image)

gaussian = np.sqrt(np.square(gaussian_x) + np.square(gaussian_y))
lena = np.sqrt(np.square(lena_x) + np.square(lena_y))

plt.subplot(1, 3, 2)
plt.title('lena_gradient')
plt.imshow(lena, cmap = 'gray')

plt.subplot(1, 3, 3)
plt.title('Gaussian_gradient')
plt.imshow(gaussian, cmap = 'gray')

plt.show()
