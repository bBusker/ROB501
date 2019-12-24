import numpy as np
from numpy.linalg import inv
from dcm_from_rpy import dcm_from_rpy
from estimate_motion_ls import estimate_motion_ls
from estimate_motion_ils import estimate_motion_ils

# Generate initial and transformed points.
C  = dcm_from_rpy(np.array([10, -8, 12])*np.pi/180)
t  = np.array([[0.5], [-0.8], [1.7]])

#Pi = np.array([[1, 2, 3, 4], [7, 3, 4, 8], [9, 11, 6, 3]])
Pi = np.random.rand(3, 10)
Pf = C@Pi + t
Si = np.dstack((1*np.eye(3),)*10)
Sf = np.dstack((1*np.eye(3),)*10)

Tfi_est = estimate_motion_ls(Pi, Pf, Si, Sf)
Tfi_est_nl = estimate_motion_ils(Pi, Pf, Si, Sf, 100)

# Check that the transforms match...
Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))
print(Tfi - Tfi_est)

# Now try with iteration.
#Tfi_est = estimate_motion_ils(Pi, Pf, Si, Sf, 10)
#print(Tfi - Tfi_est)