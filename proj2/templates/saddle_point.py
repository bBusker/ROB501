import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    # Set constants
    m, n = I.shape
    num_pix = m*n
    lat_offset = 0.5
    long_offset = 0.5

    # Create A and b matrix for linear least squares
    A = np.zeros((num_pix, 6))
    for x in range(m):
        for y in range(n):
            x_o = ((x+0.5)/m) - lat_offset
            y_o = ((y+0.5)/n) - long_offset
            arr = np.array((x_o**2, x_o*y_o, y_o**2, x_o, y_o, 1))
            A[n*x+y] = arr

    b = np.zeros((num_pix, 1))
    for x in range(m):
        for y in range(n):
            b[n*x+y] = I[x][y]

    # Solve linear least squares using numpy
    curve = lstsq(A, b)

    # Extract linear least squares output
    alpha = curve[0][0]
    beta = curve[0][1]
    gamma = curve[0][2]
    delta = curve[0][3]
    epsilon = curve[0][4]
    xi = curve[0][5]

    # Create matricies for finding saddle point
    intersect_A = np.array([[2*alpha, beta], [beta, 2*gamma]]).reshape((2,2))
    intersect_A = -inv(intersect_A)
    intersect_b = np.array([[delta], [epsilon]]).reshape((2, 1))

    # Calculate saddle point
    pt = np.matmul(intersect_A, intersect_b)
    x_t = pt[0]*m + m//2
    y_t = pt[1]*n + n//2

    # Swapping x and y to comply with image convention
    pt[1] = x_t
    pt[0] = y_t

    return pt
