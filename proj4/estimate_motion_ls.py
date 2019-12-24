import numpy as np
from numpy.linalg import det, svd

def estimate_motion_ls(Pi, Pf, Si, Sf):
    """
    Estimate motion from 3D correspondences.

    The function estimates the 6-DOF motion of a boby, given a series 
    of 3D point correspondences. This method (which avoids using Lagrange 
    multipliers directly to constrain C) is described in (Matthies, PhD 
    thesis, CMU, 1989).

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Returns:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """
    # Weights are inverse of determinants of covariance matrices.
    w = 0
    Qi = np.zeros((3, 1))
    Qf = np.zeros((3, 1))
    A  = np.zeros((3, 3))

    for i in np.arange(Pi.shape[1]):
        # Set weights as sum of inverse of determinants of initial and final point
        wj = det(np.linalg.inv(Si[:,:,i])) + det(np.linalg.inv(Sf[:,:,i]))

        w  += wj
        Qi += wj*Pi[:, i].reshape(3, 1)
        Qf += wj*Pf[:, i].reshape(3, 1)
        A  += wj*(Pf[:, i].reshape(3, 1))@(Pi[:, i].reshape(1, 3))

    E = A - 1/w*Qf.reshape(3, 1)@Qi.reshape(1, 3)

    # Compute SVD of E, then use that to find R, T.
    U, S, Vh = svd(E)
    M = np.eye(3)
    M[2, 2] = det(U)*det(Vh)
    C = U@M@Vh
    t = 1/w*(Qf - C@Qi)

    Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))

    return Tfi
