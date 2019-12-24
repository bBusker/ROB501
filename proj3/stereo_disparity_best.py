import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    # Settings and general checks
    window_size = 4
    assert Il.shape == Ir.shape

    # Create image arrays. Pad with 0 by window_size so the convolution
    # later doesnt go out of bounds
    # Add sobel filter to check for gradients in x and y direction
    # for calculation of SAD to improve stereo matching
    l_padded = np.pad(Il, window_size, 'edge')
    r_padded = np.pad(Ir, window_size, 'edge')
    left_imgs = [l_padded,
                 sobel(l_padded, axis=0),
                 sobel(l_padded, axis=1)]
    right_imgs = [r_padded,
                  sobel(r_padded, axis=0),
                  sobel(r_padded, axis=1)]
    Id = np.zeros_like(Il)


    # Loop through each pixel and check SAD score between the pixel and
    # others in its row from disparity = x-maxd to 0 (pixel correspondence between
    # left and right image can only be in negative direction)
    for y in range(bbox[1][0]+window_size, bbox[1][1]+window_size):
        for x in range(bbox[0][0]+window_size, bbox[0][1]+window_size):
            best_sad_score = float('inf')
            w_Il = (left_imgs[0])[y - window_size:y + window_size + 1, x - window_size:x + window_size + 1]
            ud_Il = (left_imgs[1])[y - window_size:y + window_size + 1, x - window_size:x + window_size + 1]
            lr_Il = (left_imgs[2])[y - window_size:y + window_size + 1, x - window_size:x + window_size + 1]

            for xr in range(max(x-maxd, window_size), x+1):
                # Compute SAD score
                w_Ir = (right_imgs[0])[y-window_size:y+window_size+1, xr-window_size:xr+window_size+1]
                ud_Ir = (right_imgs[1])[y-window_size:y+window_size+1, xr-window_size:xr+window_size+1]
                lr_Ir = (right_imgs[2])[y-window_size:y+window_size+1, xr-window_size:xr+window_size+1]
                sad = np.sum(np.abs(w_Il - w_Ir)) + np.sum(np.abs(ud_Il - ud_Ir)) + np.sum(np.abs(lr_Il - lr_Ir))
                # Save best SAD score and corresponding disparity
                if sad < best_sad_score:
                    best_sad_score = sad
                    Id[y-window_size][x-window_size] = x-xr
    Id = median_filter(Id, 10)
    return Id