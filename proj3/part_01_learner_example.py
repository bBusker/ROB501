import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread
from stereo_disparity_fast import stereo_disparity_fast
from stereo_disparity_score import stereo_disparity_score
import timeit

# Load the stereo images.
Il = imread("images/cones_image_02.png", as_gray = True)
Ir = imread("images/cones_image_06.png", as_gray = True)
It = imread("images/cones_disp_02.png", as_gray = True)

# Load the appropriate bounding box.
bboxes = loadmat("images/bboxes.mat")
bbox = np.array(bboxes["cones_02"]["bbox"])

Id = stereo_disparity_fast(Il, Ir, bbox, 52)
plt.imshow(Id, cmap = "gray")
plt.show()
# plt.imshow(Il[50:150, 20:40], cmap='gray')
# plt.show()
print("Score: {}".format(stereo_disparity_score(It, Id, bbox)))