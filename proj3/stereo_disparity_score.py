import numpy as np
import matplotlib.pyplot as plt

def stereo_disparity_score(It, Id, bbox):
    """
    Evaluate accuracy of disparity image.

    This function computes the RMS error between a true (known) disparity
    map and a map produced by a stereo matching algorithm. There are many
    possible metrics for stereo accuracy: we use the RMS error and the 
    percentage of incorrect disparity values (where we allow one unit
    of 'wiggle room').

    Note that pixels in the grouth truth disparity image with a value of
    zero are ignored (these are deemed to be invalid pixels).

    Parameters:
    -----------
    It  - Ground truth disparity image, m x n pixel np.array, greyscale.
    Id  - Computed disparity image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).

    Returns:
    --------
    N      - Number of valid depth measurements in It image.
    rms    - Test score, RMS error between Id and It.
    p_bad  - Percentage of incorrect depth values (for valid pixels).
    """
    # Ignore points where ground truth is unknown.
    mask = It != 0

    # Cut down the mask to only consider pixels in the box...
    mask[:, :bbox[0, 0]] = 0
    mask[:, bbox[0, 1] + 1:] = 0
    mask[:bbox[1, 0], :] = 0
    mask[bbox[1, 1] + 1:, :] = 0
    #plt.imshow(mask, cmap = "gray")
    #plt.show()

    N = np.sum(mask)  # Total number of valid pixels.
    rms = np.sqrt(np.sum(np.square(Id[mask] - It[mask]))/N)
    p_bad = np.sum((Id[mask] - It[mask]) > 1)/N

    return N, rms, p_bad