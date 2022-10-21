import cv2
import numpy as np
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt

import cv2
from sklearn.cluster import KMeans


class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, img, clusters=3):
        self.CLUSTERS = clusters
        self.img = img.copy()

    def dominantColors(self):

        # convert to rgb from bgr
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))

        # save image after operations
        self.IMAGE = img

        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS, n_init=1, random_state=1, max_iter=10)
        kmeans.fit(img)

        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        # save labels
        self.LABELS = kmeans.labels_

        labels, counts = np.unique(kmeans.labels_, return_counts=True)

        # returning after converting to integer from float
        return self.COLORS[np.argsort(counts)].astype(int)

    def plotHistogram(self):

        # labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS + 1)

        # create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.axis("off")
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        ax1.imshow(np.array(image))

        # display chart
        # ax2.figure()
        ax2.axis("off")
        ax2.imshow(chart)
        plt.show()


# filename = "images_line_detection/image_001.png"
# frame = cv2.imread(filename)
#
# clusters = 5
## dc = DominantColors(frame, clusters)
## colors = dc.dominantColors()
## dc.plotHistogram()
## print(colors)
#
# width = 1920
# height = 1080
#
# filename = Path(filename)
# yolo_file = f"images_line_detection_labels/{filename.stem}.txt"
#
#
# data = np.loadtxt(yolo_file)
#
# if __name__ == "__main__":
#    for i in range(data.shape[0]):
#        pt1, pt2 = yolobbox2bbox(
#            data[i, 1],
#            data[i, 2],
#            data[i, 3],
#            data[i, 4],
#            width,
#            height,
#        )
#        bbox = (
#            pt1[0],
#            pt1[1],
#            pt2[0] - pt1[0],
#            pt2[1] - pt1[1],
#        )
#
#        # Draw bounding box
#        sub_img = frame[pt1[1] : pt2[1], pt1[0] : pt2[0]]
#        dc = DominantColors(sub_img, 2)
#        colors = dc.dominantColors()
#        color = (int(colors[0][2]), int(colors[0][1]), int(colors[0][0]))
#        # dc.plotHistogram()
#
#        cv2.rectangle(frame, pt1, pt2, color, 2, 1)
#
#    # Display result
#    cv2.imshow("Tracking", frame)
#    cv2.waitKey(0)
#
#    cv2.destroyAllWindows()
#
