import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread
from mat4py import loadmat
from triangulate import triangulate
from estimate_motion_ils import estimate_motion_ils
from vo_setup import VOSetup

def visual_odometry(setup):
    """
    Runs full VO pipeline, visualizes the resulting motion estimate and
    returns the pose change (start-to-end).

    Parameters:
    -----------
    setup  - VO setup object, all required details for the VO algorithm.

    Returns:
    --------
    Tba  - 4x4 np.array, homogeneous pose matrix, change in robot pose
           from 'after' frame to 'before' frame.
    """
    matchDir  = setup.projectRoot + setup.matchDir
    trackDir  = setup.projectRoot + setup.trackDir
    ransacDir = setup.projectRoot + setup.ransacDir

    # Set up stereo camera matrices.
    Kl = setup.Kl
    Kr = setup.Kr 
    Tl = np.eye(4)
    Tr = np.eye(4)
    Tr[0, 3] = setup.baseline

    Tba = np.eye(4)
    trans = np.zeros((3, 1))

    # Set up figure windows, if we want them.
    fig_imagept = None
    fig_traject = None

    if setup.plotPoints:
        fig_imagept = plt.figure()
    
    if setup.plotTrajectory:
        fig_traject = plt.figure()

    # Main loop, estimating incremental motion.
    for i in range(setup.startInd, setup.stopInd + 1):
        # Report progress.
        print("At frame %d of %d." % (i, setup.stopInd))

        # Load points file.
        try:
            points = \
                loadmat("%s/%s%04d.mat" % (matchDir, setup.interPrefix, i))
            ransac = \
                loadmat("%s/%s%04d.mat" % (ransacDir, setup.ransacPrefix, i))
            # Inlying matches / stereo points pairs.
            idxs = np.array(ransac["inliers"])
            points = np.array(points["common"])[idxs - 1, :]
        except:
            # No such file, so skip it and move on to the next one.
            continue

        # Typically a RANSAC loop would run here, but we have preprocessed the
        # data in advance.

        # Build covariance matrices - these are matrices of image plane variances
        # derived from the epipolar matching function. They provide information on
        # the quality of the match.
        Sil = np.empty((2, 2, points.shape[0]))
        Sfl = np.empty((2, 2, points.shape[0]))

        for j in range(points.shape[0]):
            Sil[:, :, j] = np.array([[points[j,  5], 0], [0, points[j,  6]]])
            Sfl[:, :, j] = np.array([[points[j, 11], 0], [0, points[j, 12]]])
    
        Sir = Sil.copy()
        Sfr = Sfl.copy()

        # Triangulate points before and after - then use the triangulated points
        # for motion estimation. Note that the array Sb is 3x3xn, and contains a
        # 3x3 covariance matrix for each landmark point in Pb.
        Pi = np.empty((3, points.shape[0]))
        Pf = np.empty((3, points.shape[0]))
        Si = np.empty((3, 3, points.shape[0]))
        Sf = np.empty((3, 3, points.shape[0]))

        for j in range(points.shape[0]):
            # Initial
            _, _, Pi[:, [j]], Si[:, :, j] = \
                triangulate(Kl, Kr, Tl, Tr, 
                            points[j, 1:3].reshape(2, 1),
                            points[j, 3:5].reshape(2, 1),
                            Sil[:, :, j], Sir[:, :, j])

            # Final
            _, _, Pf[:, [j]], Sf[:, :, j] = \
                triangulate(Kl, Kr, Tl, Tr, 
                            points[j,  7:9].reshape(2, 1),
                            points[j, 9:11].reshape(2, 1),
                            Sfl[:, :, j], Sfr[:, :, j])
            #print(Pi[:, j])
            #print(Pf[:, j])

        # Estimate incremental motion.

        Tfi = estimate_motion_ils(Pi, Pf, Si, Sf, 100)

        Tba = Tba@Tfi

        #print(Tfi)
        #print()
        #print(Tba)
        #print()

        # Append new increment to path for plotting.
        trans = np.append(trans, Tba[0:3, 3].reshape((3, 1)), axis = 1)

        # Show trajectory.
        if setup.plotTrajectory:
            plt.figure(fig_traject.number)
            plt.clf()
            ax = fig_traject.add_subplot(111, projection = '3d')
            ax.plot(trans[0, :], trans[1, :], trans[2, :], 'b-')
            #ax.set_aspect("equal")
            plt.grid()
            plt.pause(0.01)

        # Overlay motion on image.
        if setup.plotPoints:
            I = plt.imread("%s/l-rect-%08d_track.ppm" % (trackDir, i))
            plt.figure(fig_imagept.number)
            plt.clf()
            plt.imshow(I)
    
            # Overlay tracks on image.
            for k in range(points.shape[0]):
                plt.plot([points[k, 1], points[k, 7]], \
                         [points[k, 2], points[k, 8]] , c = 'b')
            plt.scatter(points[:, 1], points[:, 2], s = 20, c = 'r', marker = 'x')
            plt.scatter(points[:, 7], points[:, 8], s = 20, c = 'g', marker = 'o')
            plt.pause(0.01)
    
    return Tba

if __name__ == "__main__":
    # Set all required parameters.
    setup = VOSetup()
    Tba = visual_odometry(setup)
    print(Tba)