# importing necessary packages
import numpy as np
from skimage import feature


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # initialising number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def compute(self, image, eps=1e-7):
        # computing the Local Binary Pattern representation of the image
        # Using LBP representation to build histogram of patterns

        lbp = feature.local_binary_pattern(
            image, self.numPoints, self.radius, method="uniform"
        )
        (hist, _) = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2),
        )

        # normalize the histogram
        hist = hist.astype("float")
        hist /= hist.sum() + eps

        # return the histogram of Local Binary Patterns
        return hist
