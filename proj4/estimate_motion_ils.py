import numpy as np
from numpy.linalg import det, inv, svd
from rpy_from_dcm import rpy_from_dcm
from dcm_from_rpy import dcm_from_rpy
from estimate_motion_ls import estimate_motion_ls

def estimate_motion_ils(Pi, Pf, Si, Sf, iters):
    """
    Estimate motion from 3D correspondences.
  
    The function estimates the 6-DOF motion of a boby, given a series 
    of 3D point correspondences. This method relies on NLS.
    
    Arrays Pi and Pf store corresponding landmark points before and after 
    a change in pose.  Covariance matrices for the points are stored in Si 
    and Sf.

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Outputs:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """
    # Initial guess...
    Tfi = estimate_motion_ls(Pi, Pf, Si, Sf)

    # Initialize arrays for nls
    C = Tfi[:3, :3]
    t = Tfi[:3, 3].reshape(3,1)
    I = np.eye(3)
    rpy = rpy_from_dcm(C).reshape(3, 1)

    # Variable parameters
    step_size = 1

    # Iterate for each step
    for _ in np.arange(iters):
        # Reset incremental update arrays
        drpy = np.zeros((3,1))
        dt = np.zeros((3,1))

        # Iterate through each point
        for i in np.arange(Pi.shape[1]):
            # Get current points
            curr_Pi = Pi[:, i]
            curr_Pf = Pf[:, i]

            # Find residual
            est_Pf = Tfi @ np.hstack((curr_Pi.T, [1])).reshape(4,1)
            r = est_Pf[:3,:] - curr_Pf.reshape(3,1)

            # Construct jacobian
            # Rotation jacobian from function, position jacobian is identity
            dRdr, dRdp, dRdy = dcm_jacob_rpy(C)
            J = np.zeros((3,6))
            J[:, 0:3] = np.array((dRdr@curr_Pi, dRdp@curr_Pi, dRdy@curr_Pi))
            J[:, 3:6] = I

            # Construct weighting matrix based on inverse covariance
            Sj = Sf[:,:,i] + C @ Si[:,:,i] @ C.T

            # Contruct A and B matricies
            A = (J.T)@Sj@J
            B = (J.T)@Sj@r

            # Check for A being singular
            # (This sometimes happens when residuals are too small)
            if np.linalg.matrix_rank(A) != 6:
                continue

            # Calculate nls result for current point and add to sum
            theta = inv(A) @ B
            drpy += theta[0:3].reshape(3, 1)
            dt += theta[3:6].reshape(3, 1)

        # Calculate update step
        rpy += drpy * step_size
        t += dt * step_size
        C = dcm_from_rpy(rpy)
        Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))

    return Tfi

def dcm_jacob_rpy(C):
     # Rotation - convenient!
    cp = np.sqrt(1 - C[2, 0]*C[2, 0])
    cy = C[0, 0]/cp
    sy = C[1, 0]/cp

    dRdr = C@np.array([[ 0,   0,   0],
                       [ 0,   0,  -1],
                       [ 0,   1,   0]])

    dRdp = np.array([[ 0,    0, cy],
                     [ 0,    0, sy],
                     [-cy, -sy,  0]])@C

    dRdy = np.array([[ 0,  -1,  0],
                     [ 1,   0,  0],
                     [ 0,   0,  0]])@C
    
    return dRdr, dRdp, dRdy