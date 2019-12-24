import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline.

    Parameters:
    -----------
    Kl   - 3 x 3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3 x 3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    # Initialize arrays for position and rotation
    rot_l = Twl[:3, :3]
    rot_r = Twr[:3, :3]
    pos_l = Twl[:3, 3]
    pos_r = Twr[:3, 3]

    # Compute baseline (right camera translation minus left camera translation).
    b = pos_r - pos_l

    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    rayl_unnorm = rot_l @ inv(Kl) @ np.vstack((pl, [1]))
    rayr_unnorm = rot_r @ inv(Kr) @ np.vstack((pr, [1]))

    rayl = rayl_unnorm/norm(rayl_unnorm)
    rayr = rayr_unnorm/norm(rayr_unnorm)

    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    ml = (b.dot(rayl) - (b.dot(rayr))*((rayl.T).dot(rayr))) / (1-((rayl.T).dot(rayr)**2))
    mr = ((rayl.T).dot(rayr))*ml - b.dot(rayr)
    
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    Pl = pos_l.reshape(3,1) + rayl*ml
    Pr = pos_r.reshape(3,1) + rayr*mr

    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    # Calculate derivative of unit vector rays.
    # First get derivative of left and right rays (rl and rr) with respect
    # to u and v
    drl_du = (rot_l @ inv(Kl) @ np.array([[1, 0, 0]]).T).reshape((3,1))
    drl_dv = (rot_l @ inv(Kl) @ np.array([[0, 1, 0]]).T).reshape((3,1))
    drr_du = (rot_r @ inv(Kr) @ np.array([[1, 0, 0]]).T).reshape((3,1))
    drr_dv = (rot_r @ inv(Kr) @ np.array([[0, 1, 0]]).T).reshape((3,1))

    # Get magnitude of ray for denominator value in derivative
    magl = norm(rayl_unnorm)
    magr = norm(rayr_unnorm)

    # Calculate derivative of norm(rayl) and norm(rayr) (nrl and nrr)
    # with respect to u and v
    dnrl_du = drl_du.T @ rayl_unnorm / magl
    dnrl_dv = drl_dv.T @ rayl_unnorm / magl
    dnrr_du = drr_du.T @ rayr_unnorm / magr
    dnrr_dv = drr_dv.T @ rayr_unnorm / magr

    # Calculate the final derivative using chain rule
    drayl[:, 0] = ((drl_du*magl - rayl*dnrl_du) / magl**2).reshape((3,))
    drayl[:, 1] = ((drl_dv*magl - rayl*dnrl_dv) / magl**2).reshape((3,))
    drayr[:, 2] = ((drr_du*magr - rayr*dnrr_du) / magr**2).reshape((3,))
    drayr[:, 3] = ((drr_dv*magr - rayr*dnrr_dv) / magr**2).reshape((3,))

    #------------------

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    # 3D point.
    P = (Pl+Pr)/2
    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    S_im = np.zeros((4,4))
    S_im[0:2, 0:2] = Sl
    S_im[2:4, 2:4] = Sr
    S = JP @ S_im @ JP.T

    return Pl, Pr, P, S