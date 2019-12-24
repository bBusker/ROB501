import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # Initialize points
    x = pt[0].item()
    y = pt[1].item()
    x_1 = int(np.floor(x))
    x_2 = int(np.ceil(x))
    y_1 = int(np.floor(y))
    y_2 = int(np.ceil(y))

    # Generate A and b matricies for bilinear interpolation coefficient calculation
    A = np.array(((1, x_1, y_1, x_1*y_1),
                  (1, x_1, y_2, x_1*y_2),
                  (1, x_2, y_1, x_2*y_1),
                  (1, x_2, y_2, x_2*y_2)))
    b = np.array((1, x, y, x*y))

    # Calculate coefficients using (A^-1)^T * b formula
    coeffs = np.matmul(inv(A).transpose(), b)

    # Find results of bilinear interpolation by multiplying pixel intensities with corresponding coeffs
    res = coeffs[0]*I[y_1][x_1] + coeffs[1]*I[y_2][x_1] + coeffs[2]*I[y_1][x_2] + coeffs[3]*I[y_2][x_2]

    # Round result to return whole number
    res = round(res.item(), 0)

    return res
