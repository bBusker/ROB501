import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    # Run for maxIters
    for iter in range(maxIters):
        # Calculate residuals
        for i in range(tp):
            dY[2*i:2*i+2, :] = img_pt_from_world_pt(K, Twcg, Wpts[:, i].reshape((3, 1))) - Ipts[:, i].reshape(2,1)

        # Calculate jacobian
        for i in range(tp):
            J[2*i:2*i+2, :] = find_jacobian(K, Twcg, Wpts[:, i].reshape((3, 1)))

        # Get update step
        step = np.linalg.inv(J.T @ J) @ J.T @ dY

        # Update parameters
        curr_E = epose_from_hpose(Twcg)
        updated_E = curr_E - step
        Twcg = hpose_from_epose(updated_E)

    Twc = Twcg
    return Twc

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def img_pt_from_world_pt(K, Twc, Wpt):
    # Rotation Matrix
    R_wc = Twc[:3, :3]

    # Translation Vector
    t_wc = Twc[:3, -1:]

    # Find world point on image plane
    u = Wpt - t_wc
    I_pt = K.dot(R_wc.T).dot(u)
    I_pt /= I_pt[2]
    I_pt = I_pt[0:2]

    return I_pt
