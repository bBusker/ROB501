import numpy as np
from scipy.ndimage.filters import *
from PIL import Image, ImageDraw

def cross_junctions(I, bounds, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts    - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of I. These should be floating-point values.
    """
    # Initialize matrices and constants
    Ipts = np.zeros((2, 48))
    num_crosses = Wpts.shape[1]
    around_size = 10

    # Find corners of bounding box in world coordinates
    upleft_world = (Wpts[:,0] + np.array([-(0.028+0.0635), -(0.011+0.0635), 1])).reshape(3,1)
    upright_world = (Wpts[:,7] + np.array([0.025+0.0635, -(0.012+0.0635), 1])).reshape(3,1)
    botleft_world = (Wpts[:,40] + np.array([-(0.025+0.0635), 0.012+0.0635, 1])).reshape(3,1)
    botright_world = (Wpts[:,47] + np.array([0.025+0.0635, 0.012+0.0635, 1])).reshape(3,1)
    world_corners = np.concatenate((upleft_world, upright_world, botright_world, botleft_world), axis=1)

    # Compute homography between world and image
    H, A = dlt_homography(world_corners, bounds)

    # Find corresponding image point for each world point
    # using homography
    img_pts = np.zeros_like(Wpts)
    for i in range(num_crosses):
        img_pt = np.matmul(H, Wpts[:, i] + np.array([0,0,1]))
        img_pt /= img_pt[2]
        img_pts[:, i] = img_pt

    # Use saddle point locator to refine guess
    for i in range(num_crosses):
        curr_pt = img_pts[:2, i]

        # Get image patch to run saddle point locator on
        min_x = max(int(curr_pt[0])-around_size, 0)
        max_x = min(int(curr_pt[0])+around_size, I.shape[1])
        min_y = max(int(curr_pt[1])-around_size, 0)
        max_y = min(int(curr_pt[1])+around_size, I.shape[0])
        cropped_img = I[min_y:max_y, min_x:max_x]

        # Get saddle point location on crop and add offset between
        # crop and overall image
        saddle_pt_cropped_coords = saddle_point(cropped_img)
        saddle_pt_img_coords = saddle_pt_cropped_coords + np.array([[min_x], [min_y]])
        Ipts[:, i] = saddle_pt_img_coords.reshape(2,)

    return Ipts

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """

    # Variable initialization
    num_points = 4
    A_list = []

    # Generate A matrix by generating A_i matrix for each pair of points and concatenating
    for i in range(num_points):
        x = I1pts[0][i]
        y = I1pts[1][i]
        u = I2pts[0][i]
        v = I2pts[1][i]
        A_i = np.array(((-x, -y, -1, 0, 0, 0, u*x, u*y, u), # From Dubrofsky's M.A.Sc. thesis
                        (0, 0, 0, -x, -y, -1, v*x, v*y, v)))
        A_list.append(A_i)
    A = np.concatenate(A_list)

    # Find H matrix as the null space of A
    H = nullspace(A)
    # Normalize and reshape to desired output
    H = H*(1/H[-1])
    H = np.reshape(H, (3,3))

    return H, A

def nullspace(A, atol=1e-13, rtol=0):
    # Calculate null space for a matrix A
    # Check dim
    A = np.atleast_2d(A)
    # Get svd of matrix
    u, s, vh = np.linalg.svd(A)
    # Set tolerance based on bounds/precision
    tol = max(atol, rtol * s[0])
    # Calculate null space vectors
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

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
    curve = np.linalg.lstsq(A, b)

    # Extract linear least squares output
    alpha = curve[0][0]
    beta = curve[0][1]
    gamma = curve[0][2]
    delta = curve[0][3]
    epsilon = curve[0][4]
    xi = curve[0][5]

    # Create matricies for finding saddle point
    intersect_A = np.array([[2*alpha, beta], [beta, 2*gamma]]).reshape((2,2))
    intersect_A = -np.linalg.inv(intersect_A)
    intersect_b = np.array([[delta], [epsilon]]).reshape((2, 1))

    # Calculate saddle point
    pt = np.matmul(intersect_A, intersect_b)
    x_t = pt[0]*m + m//2
    y_t = pt[1]*n + n//2
    pt[1] = x_t
    pt[0] = y_t

    return pt
