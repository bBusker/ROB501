import numpy as np
# from PIL import Image

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # Initialize constants
    total_pixels = I.shape[0] * I.shape[1]

    # Construct pixel intensity histogram
    histogram = np.zeros((256))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            histogram[I[i][j]] += 1

    # Create mapping array
    mapping = np.zeros((256))
    mapping[0] = histogram[0] / total_pixels * 255
    for i in range(1, mapping.shape[0]):
        mapping[i] = mapping[i-1] + histogram[i] / total_pixels * 255

    # Remap pixels into new image
    J = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            J[i][j] = round(mapping[I[i][j]], 0)
    #
    # img = Image.fromarray(J)
    # img.save("temp_histeq.png")

    return J
