import numpy as np

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The
    projection model is the simple pinhole model.

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose.
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
    """
    # Rotation Matrix
    R_wc = Twc[:3, :3]

    # Translation Vector
    t_wc = Twc[:3, -1:]

    # Find world point on image plane
    u = Wpt - t_wc
    I_pt = K.dot(R_wc.T).dot(u)

    # Calculate positional derivatives (xyz) as
    # dI_pt/dx = K*R_wc*-I
    df = np.zeros([3, 6])
    df[:, :3] = K.dot(R_wc.T).dot((-np.eye(3)))

    # Calculate rotational derivatives (rpy) as
    # the formulas found on lecture 8 slides
    # Define constants
    rpy = rpy_from_dcm(R_wc)
    cr = np.cos(rpy[0])
    sr = np.sin(rpy[0])
    cp = np.cos(rpy[1])
    sp = np.sin(rpy[1])
    cy = np.cos(rpy[2])
    sy = np.sin(rpy[2])

    # Get C matrices
    C_1 = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    C_2 = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    C_3 = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    # Calculate matrix multiplication formula
    # multiply by R_wc^-1 on either side to account
    # for finding the derivative of R_cw
    inv = np.linalg.inv
    dy = -(inv(R_wc)).dot(matrix_cross(np.array([0,0,1]))).dot(C_3).dot(C_2).dot(C_1).dot(inv(R_wc)).dot(u)
    dp = -(inv(R_wc)).dot(C_3).dot(matrix_cross(np.array([0,1,0]))).dot(C_2).dot(C_1).dot(inv(R_wc)).dot(u)
    dr = -(inv(R_wc)).dot(C_3).dot(C_2).dot(matrix_cross(np.array([1,0,0]))).dot(C_1).dot(inv(R_wc)).dot(u)

    # Set corresponding values in derivative
    df[:, 3:4] = K.dot(dr)
    df[:, 4:5] = K.dot(dp)
    df[:, 5:6] = K.dot(dy)

    # Get derivative wrt z for division chain rule
    dfz = df[-1:, :]

    # Calculate jacobian using division chain rule
    # (columns are tx, ty, tz, r, p, q)
    J = np.divide((I_pt[-1, 0] * df - I_pt.dot(dfz)), I_pt[-1, 0] * I_pt[-1, 0])[:-1, :]

    return J


def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])


def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if np.abs(cp) > 1e-15:
        rpy[1] = np.arctan2(sp, cp)
    else:
        # Gimbal lock...
        rpy[1] = np.pi / 2

        if sp < 0:
            rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

def matrix_cross(array):
    return np.array([
        [0, -array[2], array[1]],
        [array[2], 0, -array[0]],
        [-array[1], array[0], 0]
    ])