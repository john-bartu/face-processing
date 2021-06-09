#
#  Use k-means clustering to find the most-common colors in an image
#
import cv2
import numpy as np
from sklearn.cluster import KMeans


def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


# START HERE
img = cv2.imread('sample1-uv.png')
height, width, _ = np.shape(img)
# reshape the image to be a simple list of RGB pixels
image = img.reshape((height * width, 3))

# we'll pick the 5 most common colors
num_clusters = 6
clusters = KMeans(n_clusters=num_clusters)
clusters.fit(image)

# count the dominant colors and put them in "buckets"
histogram = make_histogram(clusters)
# then sort them, most-common first
combined = zip(histogram, clusters.cluster_centers_)
combined = sorted(combined, key=lambda x: x[0], reverse=True)

# finally, we'll output a graphic showing the colors in order
bars = []
hsv_values = []
for rows in combined:
    bgr = rows[1]
    print(f'  RGB values: {bgr}')
