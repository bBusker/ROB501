import numpy as np

class VOSetup():
    # Project root directory.
    def __init__(self):
        self.projectRoot = "./"

        # Match directory - change to specify desired dataset.
        self.featsDir  = ""
        self.matchDir  = "dataset/matches"
        self.ransacDir = "dataset/ransac"

        # Imagery directories - change to specify desired images.
        self.imageDir = "dataset/images"
        self.trackDir = "dataset/tracks"

        self.leftPrefix  = "l-"        # Prefix for left image of stereo pair.
        self.rightPrefix = "r-"        # Prefix for right image of stereo pair.

        self.intraPrefix = "intra_"    # Intra-frame matches within one stereo pair.
        self.interPrefix = "inter_"    # Inter-frame matches between stereo pairs.
        self.ransacPrefix = "ransac_"  # Pre-processed RANSAC filtered matches.

        self.startInd =   4            # Index of first image in dataset.
        self.stopInd  = 319            # Index of last image in dataset.

        # Camera parameters.
        self.Kl = np.array([[1057.808428,           0, 519.190040],
                            [          0, 1052.916356, 388.633721],
                            [          0,           0,          1]])
        self.Kr = np.array([[1057.808428,           0, 519.190040],
                            [          0, 1052.916356, 388.633721],
                            [          0,           0,          1]])
        self.baseline = 0.256857

        # Estimation parameters.
        self.numIters = 50             # Max. number of ILS iterations.

        # Plotting flags.
        self.plotPoints = True
        self.plotTrajectory = True