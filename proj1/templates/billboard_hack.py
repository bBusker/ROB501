# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Extract bounding box min and max
    xmin = 404
    xmax = 490
    ymin = 38
    ymax = 354

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    # Get images
    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')
    Ihack = Iyd

    Ist = np.asarray(Ist)

    # Let's do the histogram equalization first.
    Ist = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Generate bounds in the Yonge-Dundas image for where we want to replace
    bound_pts = []
    for i in range(4):
        bound_pts += [(Iyd_pts[0][i], Iyd_pts[1][i])]
    bounds = Path(bound_pts)

    # Main for loop to calculate pixel values. Iterate through all bounding box pts
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            # If the point is not in the bound for the ad, ignore it
            if not bounds.contains_point((x, y)):
                continue

            # If point is in the bound for the ad, calculate the corresponding
            # point in the soldiers tower image using homography
            pt_yd = np.array((x, y, 1))
            pt_st = np.matmul(H, pt_yd)
            pt_st /= pt_st[2]

            # Use bilinear interpolation to find the pixel value we should put
            # in the Yonge-Dundas image (replace each channel with the value)
            for channel in range(3):
                Ihack[y][x][channel] = bilinear_interp(Ist, np.reshape(pt_st[:-1], (2,1)))

    return Ihack

if __name__ == "__main__":
    billboard_hack()