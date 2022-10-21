"""
Main script to use the whole algorithms.
First detect key points on the soccer pitch and find the
camera intrinsic/extrinsic calibration matrices
Then use yolov5 detector to locate players and balls.
Finally reproject those detections to a top view image of the soccer field
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

# from common import draw_point, yolobbox2bbox
from camera_pose_estimation.main import calibrate_from_image
from team_assigner_with_ball_tracking import basic_ball_tracker, yolobbox2bbox

from team_assigner import DominantColors


def from_world_to_field(x, z):
    """Map world coordinates to the soccer_field.png image"""

    center_field = [826, 520]
    scale = (1585 - 68) / 105  # 105 meters

    u = int(center_field[0] + x * scale)
    v = int(center_field[1] - z * scale)
    return (u, v)


def unproject_to_ground(K, to_device_from_world, u, v):
    """
    Unproject pixel point (u, v) and find its world coordinates,
    assuming that its altitude is zero
    """
    # First to K^-1 * (u, v, 1)
    homog_uv = np.ones((3, 1))
    homog_uv[0, 0] = u
    homog_uv[1, 0] = v

    K_inv_uv = np.dot(np.linalg.inv(K), homog_uv)
    alpha = K_inv_uv[0, 0]
    beta = K_inv_uv[1, 0]

    b = np.zeros((2, 1))
    r00 = to_device_from_world[0, 0]
    r02 = to_device_from_world[0, 2]
    r10 = to_device_from_world[1, 0]
    r12 = to_device_from_world[1, 2]
    r20 = to_device_from_world[2, 0]
    r22 = to_device_from_world[2, 2]
    tx = to_device_from_world[0, 3]
    ty = to_device_from_world[1, 3]
    tz = to_device_from_world[2, 3]
    b[0, 0] = alpha * tz - tx
    b[1, 0] = beta * tz - ty

    M = np.zeros((2, 2))
    M[0, 0] = r00 - alpha * r20
    M[0, 1] = r02 - alpha * r22
    M[1, 0] = r10 - beta * r20
    M[1, 1] = r12 - beta * r22

    final = np.dot(np.linalg.inv(M), b)
    X = final[0, 0]
    Z = final[1, 0]

    return X, Z


def is_in_field(X, Z):
    """Check if point (X, Z) is located inside the soccer field boundaries"""
    return -52.5 < X < 52.5 and -34 < Z < 34


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Main script to use the whole algorithms + display top view of the soccer pitch"
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

    field = cv2.imread("soccer_field.png")
    field_copy = field.copy()

    ball_positions = basic_ball_tracker(Path(args.labels))

    # Default value of our focal length to start with
    # Dont forget to change this value if you are using different sizes of images
    guess_fx = 2000
    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)

    for i, filename in enumerate(images):
        if not filename.exists():
            continue

        print(str(filename))
        img = cv2.imread(str(filename))

        # Find intrinsic/extrinsic camera matrices
        K, to_device_from_world, rot, trans, img = calibrate_from_image(
            img, guess_fx, guess_rot, guess_trans
        )

        # Load yolov5 inference results
        filename = Path(filename)
        yolo_file = Path(args.labels).joinpath(Path(f"{filename.stem}.txt"))

        width = img.shape[1]
        height = img.shape[0]
        data = np.loadtxt(yolo_file)

        field = field_copy.copy()

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

            if to_device_from_world is None:
                continue

            foot_point = int((pt1[0] + pt2[0]) / 2)

            X, Z = unproject_to_ground(K, to_device_from_world, foot_point, pt2[1])

            if not is_in_field(X, Z):
                continue

            # Mark the player with a circle on the soccer pitch
            field = cv2.circle(
                field,
                from_world_to_field(X, Z),
                radius=10,
                color=color,
                thickness=3,
            )

        ball_bl, ball_tr = yolobbox2bbox(
            ball_positions[i, 0],
            ball_positions[i, 1],
            ball_positions[i, 2],
            ball_positions[i, 3],
            width,
            height,
        )
        if to_device_from_world is not None:
            foot_point = int((ball_bl[0] + ball_tr[0]) / 2)

            X, Z = unproject_to_ground(K, to_device_from_world, foot_point, ball_tr[1])

            # Mark the ball with green-blue color
            if is_in_field(X, Z):
                field = cv2.circle(
                    field,
                    from_world_to_field(X, Z),
                    radius=10,
                    color=(255, 255, 0),
                    thickness=3,
                )

        # Modify current value of calibration matrices to get benefit
        # of this computation for next image
        guess_rot = (
            rot if to_device_from_world is not None else np.array([[0.25, 0, 0]])
        )
        guess_trans = trans if to_device_from_world is not None else (0, 0, 80)
        guess_fx = K[0, 0]

        # Display result
        if not args.out:
            cv2.imshow("field", field)
            cv2.imshow("img", img)
            k = cv2.waitKey(0)

            if k == 27:
                break
        else:
            out_path = Path("out").joinpath(Path(filename).name)
            print(f"Writing image to {str(out_path)}")
            cv2.imwrite(str(out_path), field)
