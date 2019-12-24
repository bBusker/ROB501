import numpy as np
import matplotlib.pyplot as plt
import time
from mat4py import loadmat
from imageio import imread
from stereo_disparity_fast import stereo_disparity_fast
from stereo_disparity_best import stereo_disparity_best
from stereo_disparity_score import stereo_disparity_score

# Load the stereo images.
Il = imread("images/cones_image_02.png", as_gray = True)
Ir = imread("images/cones_image_06.png", as_gray = True)
It = imread("images/cones_disp_02.png", as_gray = True)

# Load the appropriate bounding box.
bboxes = loadmat("images/bboxes.mat")
bbox = np.array(bboxes["cones_02"]["bbox"])

# Id = stereo_disparity_fast(Il, Ir, bbox, 52)
# plt.imshow(Id, cmap = "gray")
# plt.show()
# N_val, RMS, p_bad = stereo_disparity_score(It, Id, bbox)
# print("Fast Score | N_val: {}, RMS: {}, p_bad: {}".format(N_val, RMS, p_bad))
# Fast Score | N_val: 133346, RMS: 5.127145421780357, p_bad: 0.1060924212199841

start = time.time()
Id = stereo_disparity_best(Il, Ir, bbox, 52)
print("Time taken: {}".format(time.time()-start))
plt.imshow(Id, cmap = "gray")
plt.show()
N_val, RMS, p_bad = stereo_disparity_score(It, Id, bbox)
print("Best Score | N_val: {}, RMS: {}, p_bad: {}".format(N_val, RMS, p_bad))
'''
Time taken: 92.81156492233276
Best Score | N_val: 133346, RMS: 4.453907885995353, p_bad: 0.13170998755118263

Best Score | N_val: 133346, RMS: 4.590562115934025, p_bad: 0.12105349991750784

Best Score | N_val: 133346, RMS: 6.336677395872395, p_bad: 0.09802318779715927

Best Score | N_val: 133346, RMS: 5.8277453660463845, p_bad: 0.09550342717441843

Best Score | N_val: 133346, RMS: 4.799839981284234, p_bad: 0.10443507866752659

Time taken: 186.6153907775879
Best Score | N_val: 133346, RMS: 3.8836435398632108, p_bad: 0.08642928921752434

Time taken: 171.59394574165344
Best Score | N_val: 133346, RMS: 3.9002118395059853, p_bad: 0.08621930916562927

Time taken: 140.12429642677307
Best Score | N_val: 133346, RMS: 3.9002118395059853, p_bad: 0.08621930916562927
'''