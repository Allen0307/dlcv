import numpy as np
from PIL import Image
import math
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

sigma = 1 / (2 * math.log(2))
image = Image.open("dog.jpg")


plt.subplot(1,4,1)
plt.title('Original')
plt.imshow(image)

image = np.array(image)
print(image.shape)

filter_ans = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])


ans = np.zeros((1024,1024,3))
for rgb in range(3):
    for row in range(1,1022):
        for column in range(1,1022):
            temp = 0
            for row1 in range(3):
                for column1 in range(3):
                    temp+=int(filter_ans[row1][column1]) * int(image[row + row1 - 1][column + column1 - 1][rgb])
            ans[row][column][rgb] = temp

plt.subplot(1,4,2)
plt.title('edge detection')
plt.imshow(ans)


filter_ans = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
ans = np.zeros((253,450,3))
for rgb in range(3):
    for row in range(1,251):
        for column in range(1,448):
            temp = 0
            for row1 in range(3):
                for column1 in range(3):
                    temp+=int(filter_ans[row1][column1]) * int(image[row + row1 - 1][column + column1 - 1][rgb])
            ans[row][column][rgb] = temp
            
plt.subplot(1,4,3)
plt.title('sharpe')
plt.imshow(ans)


filter_ans = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
ans = np.zeros((253,450,3))
for rgb in range(3):
    for row in range(1,251):
        for column in range(1,448):
            temp = 0
            for row1 in range(3):
                for column1 in range(3):
                    temp+=int(filter_ans[row1][column1]) * int(image[row + row1 - 1][column + column1 - 1][rgb])
            ans[row][column][rgb] = temp
            
plt.subplot(1,4,4)
plt.title('top sobel')
plt.imshow(ans)
plt.show()

# filter_ans = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
# ans = np.zeros((253,450,3))
# for rgb in range(3):
#     for row in range(1,251):
#         for column in range(1,448):
#             temp = 0
#             for row1 in range(3):
#                 for column1 in range(3):
#                     temp+=int(filter_ans[row1][column1]) * int(image[row + row1 - 1][column + column1 - 1][rgb])
#             ans[row][column][rgb] = temp
            
# plt.subplot(2,4,5)
# plt.title('emboss')
# plt.imshow(ans)

# filter_ans = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
# ans = np.zeros((253,450,3))
# for rgb in range(3):
#     for row in range(1,251):
#         for column in range(1,448):
#             temp = 0
#             for row1 in range(3):
#                 for column1 in range(3):
#                     temp+=int(filter_ans[row1][column1]) * int(image[row + row1 - 1][column + column1 - 1][rgb])
#             ans[row][column][rgb] = temp
            
# plt.subplot(2,4,6)
# plt.title('line detection(水平)')
# plt.imshow(ans)

# filter_ans = np.array([[0,-0.5,0],[-0.5,3,-0.5],[0,-0.5,0]])
# ans = np.zeros((253,450,3))
# for rgb in range(3):
#     for row in range(1,251):
#         for column in range(1,448):
#             temp = 0
#             for row1 in range(3):
#                 for column1 in range(3):
#                     temp+=int(filter_ans[row1][column1]) * int(image[row + row1 - 1][column + column1 - 1][rgb])
#             ans[row][column][rgb] = temp
            
# plt.subplot(2,4,7)
# plt.title('sharpen_v2')
# plt.imshow(ans)

# filter_ans = np.array([[0,0,0],[0,1,0],[0,0,-1]])
# ans = np.zeros((253,450,3))
# for rgb in range(3):
#     for row in range(1,251):
#         for column in range(1,448):
#             temp = 0
#             for row1 in range(3):
#                 for column1 in range(3):
#                     temp+=int(filter_ans[row1][column1]) * int(image[row + row1 - 1][column + column1 - 1][rgb])
#             ans[row][column][rgb] = temp
            
# plt.subplot(2,4,8)
# plt.title('shift and subtract')
# plt.imshow(ans)
# plt.show()

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

# plt.subplot(1, 3, 1)
# plt.title('Original')
# plt.imshow(image, cmap = 'gray')

# def one_conv(image):
#     I_x = []

#     for row in image:
#         i_x = np.convolve(row,[1,0,-1],'same')
#         I_x.append(i_x)
#     I_x = np.array(I_x)

#     image_y = image.transpose()
#     I_y = []

#     for row in image_y:
#         i_y = np.convolve(row,[1,0,-1],'same')
#         I_y.append(i_y)
#     I_y = np.array(I_y)
#     I_y = I_y.transpose()

#     return I_x.reshape(512, 512), I_y.reshape(512, 512)

# ans = gaussian_filter(image, sigma = 1 / (2 * sigma ** 2))

# gaussian_x, gaussian_y = one_conv(ans)
# lena_x, lena_y = one_conv(image)

# gaussian = np.sqrt(np.square(gaussian_x) + np.square(gaussian_y))
# lena = np.sqrt(np.square(lena_x) + np.square(lena_y))

# plt.subplot(1, 3, 2)
# plt.title('lena_gradient')
# plt.imshow(lena, cmap = 'gray')

# plt.subplot(1, 3, 3)
# plt.title('Gaussian_gradient')
# plt.imshow(gaussian, cmap = 'gray')

# plt.show()
