import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    # Image access convention is I[y][x]

    # Settings and general checks
    window_size = 5
    assert Il.shape == Ir.shape

    # Create image arrays. Pad with 0 by window_size so the convolution
    # later doesnt go out of bounds
    Id = np.zeros_like(Il)
    Ir = np.pad(Ir, window_size, 'constant', constant_values=0)
    Il = np.pad(Il, window_size, 'constant', constant_values=0)

    # Loop through each pixel and check SAD score between the pixel and
    # others in its row from disparity = -maxd to 0 (pixel correspondence between
    # left and right image can only be in negative direction)
    for y in range(bbox[1][0]+window_size, bbox[1][1]+window_size):
        for x in range(bbox[0][0]+window_size, bbox[0][1]+window_size):
            best_sad_score = float('inf')
            best_disparity = 0
            for d in range(-maxd, 1):
                # Safety check to make sure we are not out of bounds
                if x+d-window_size < 0 or x+d+window_size+1 > Il.shape[1]:
                    continue
                # Compute SAD score
                w_Il = Il[y-window_size:y+window_size+1, x-window_size:x+window_size+1]
                w_Ir = Ir[y-window_size:y+window_size+1, x+d-window_size:x+d+window_size+1]
                sad = np.sum(np.absolute(w_Il.flatten() - w_Ir.flatten()))
                # Save best SAD score and corresponding disparity
                if sad < best_sad_score:
                    best_sad_score = sad
                    best_disparity = abs(d)
            Id[y-window_size][x-window_size] = best_disparity
    return Id
