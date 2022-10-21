"""
This module defines the class DominantColors
and provides a very basic tracking algo to keep tracking of the ball.
The DominantColors assign a color to each player, most probably his t-shirt color
"""

from pathlib import Path
from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class DominantColors:
    """
    Find second most important color of a subimage.
    First color is arguably green (color of the soccer pitch)
    """

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, img, clusters=3):
        self.CLUSTERS = clusters
        self.img = img.copy()

    def dominant_colors(self):
        """
        Perform K-means on the RGB space to find 2 main colors
        """
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

    def plot_histogram(self):
        """
        Plot color histogram for debuggging only
        """

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


def yolobbox2bbox(x, y, w, h, img_width, img_height):
    """
    Transform yolo bbox in xy-widht-height convention
    to bottom_left and top_right coordinates
    """
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    x1, x2 = int(x1 * img_width), int(x2 * img_width)
    y1, y2 = int(y1 * img_height), int(y2 * img_height)
    return (x1, y1), (x2, y2)


def basic_ball_tracker(labels):
    """
    Basic tracker of the ball: loop over ball detections.
    If no detection, select previous known position
    """

    labels = sorted(labels.glob("**/*"))

    # Ball position tracker (use previous position if no known position)
    prev_pos = np.zeros(4)
    ball_positions = np.zeros((750, 4))
    i = 0
    for i, yolo_file in enumerate(labels):
        if not yolo_file.exists():
            raise FileExistsError

        data = np.loadtxt(yolo_file)
        ball = data[data[:, 0] == 1]
        if len(ball) == 0:
            ball_positions[i, :] = prev_pos
        else:
            ball_positions[i, :] = ball[0, 1:5]
            prev_pos = ball[0, 1:5]

    return ball_positions


def draw_colored_players(yolo_file, img):
    """
    Draw colored players with yolo bbox and DominanColors
    """

    img_copy = img.copy()
    data = np.loadtxt(yolo_file)
    width = img.shape[1]
    height = img.shape[0]

    for k in range(data.shape[0]):
        # Skip ball with class id 1
        if data[k, 0] == 1:
            continue

        # Skip detections with ridiculous values
        if data[k, 3] < 0.01:
            continue

        # Transform yolo bounding box to opencv convention
        pt1, pt2 = yolobbox2bbox(
            data[k, 1], data[k, 2], data[k, 3], data[k, 4], width, height
        )

        # Make a sub image and find shirt color
        sub_img = img[pt1[1] : pt2[1], pt1[0] : pt2[0]]
        dc = DominantColors(sub_img, 2)
        colors = dc.dominant_colors()
        color = (int(colors[0][2]), int(colors[0][1]), int(colors[0][0]))

        # Draw bounding box with found color
        cv2.rectangle(img_copy, pt1, pt2, color=color, thickness=3)

    return img_copy


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Perform basic ball tracking and find shirt color of each player"
    )
    parser.add_argument("input", type=str, help="Image path or folder")
    parser.add_argument("labels", type=str, help="Folder of yolov5 detections")
    parser.add_argument(
        "--out",
        action="store_const",
        const=True,
        default=False,
        help="Output detection images with ball tracking",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileExistsError

    images = sorted(Path(args.input).glob("**/*"))

    ball_positions = basic_ball_tracker(Path(args.labels))

    for i, filename in enumerate(images):
        if not filename.exists():
            continue

        print(str(filename))
        img = cv2.imread(str(filename))

        filename = Path(filename)
        yolo_file = Path(args.labels).joinpath(Path(f"{filename.stem}.txt"))

        width = img.shape[1]
        height = img.shape[0]

        img_copy = draw_colored_players(yolo_file, img)

        # Draw ball position
        ball_bl, ball_tr = yolobbox2bbox(
            ball_positions[i, 0],
            ball_positions[i, 1],
            ball_positions[i, 2],
            ball_positions[i, 3],
            width,
            height,
        )
        cv2.rectangle(img_copy, ball_bl, ball_tr, color=(255, 255, 0), thickness=3)

        # Display result
        if not args.out:
            cv2.imshow("Tracking", img_copy)
            k = cv2.waitKey(0)
            if k == 27:
                break
        else:
            out_path = Path("out").joinpath(Path(filename).name)
            print(f"Writing image to {str(out_path)}")
            cv2.imwrite(str(out_path), img_copy)
