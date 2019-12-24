import numpy as np
from scipy.ndimage import gaussian_filter
from saddle_point import saddle_point

def main():
    # Build non-smooth but noise-free test patch.
    Il = np.hstack((np.ones((10, 8)), np.zeros((10, 10))))
    Ir = np.hstack((np.zeros((10, 8)), np.ones((10, 10))))
    I = np.vstack((Il, Ir))

    filtered_I = gaussian_filter(I, (1,1))

    pt = saddle_point(I)
    print('Saddle point is at: (%.2f, %.2f)' % (pt[0], pt[1]))

if __name__ == "__main__":
    main()