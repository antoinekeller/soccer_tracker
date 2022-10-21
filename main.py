import cv2
import numpy as np
import os
from argparse import ArgumentParser
from pathlib import Path

# from common import draw_point, yolobbox2bbox
from camera_pose_estimation.main import calibrate_from_image

from team_assigner import DominantColors


def from_world_to_field(x, z):
    center_field = [826, 520]
    scale = (1585 - 68) / 105  # 105 meters

    u = int(center_field[0] + x * scale)
    v = int(center_field[1] - z * scale)
    return (u, v)


def unproject_to_ground(K, to_device_from_world, u, v):
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
    return -52.5 < X < 52.5 and -34 < Z < 34


if __name__ == "__main__":
    # Default value of our focal length to start with
    # Dont forget to change this value if you are using different sizes of images
    guess_fx = 2000

    parser = ArgumentParser(
        description="Main script to key points on a soccer field image"
    )
    parser.add_argument("input", type=str, help="Image path or folder")
    parser.add_argument("labels", type=str, help="Folder of yolov5 detections")
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileExistsError

    images = sorted(Path(args.input).glob("**/*"))

    field = cv2.imread("soccer_field.png")
    field_copy = field.copy()
    field = cv2.circle(
        field, from_world_to_field(52.5, 30), radius=3, color=(0, 0, 255), thickness=3
    )
    cv2.imshow("field", field)
    cv2.waitKey(0)

    # Ball position tracker
    prev_pos = np.zeros(4)
    ball_positions = np.zeros((750, 4))
    i = 0
    for filename in images:
        filename = Path(filename)
        yolo_file = Path(args.labels).joinpath(Path(f"{filename.stem}.txt"))

        if not yolo_file.exists():
            raise FileExistsError

        data = np.loadtxt(yolo_file)
        ball = data[data[:, 0] == 1]
        if len(ball) == 0:
            ball_positions[i, :] = prev_pos
        else:
            ball_positions[i, :] = ball[0, 1:5]
            prev_pos = ball[0, 1:5]
        i += 1

    j = 0

    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)

    for k, filename in enumerate(images):
        if not filename.exists():
            continue

        print(str(filename))
        img = cv2.imread(str(filename))

        K, to_device_from_world, rot, trans, img = calibrate_from_image(
            img, guess_fx, guess_rot, guess_trans
        )
        # Exp

        filename = Path(filename)
        yolo_file = Path(args.labels).joinpath(Path(f"{filename.stem}.txt"))

        width = img.shape[1]
        height = img.shape[0]
        data = np.loadtxt(yolo_file)

        pt1, pt2 = yolobbox2bbox(
            ball_positions[k, 0],
            ball_positions[k, 1],
            ball_positions[k, 2],
            ball_positions[k, 3],
            width,
            height,
        )
        print(pt1, pt2)
        cv2.rectangle(img, pt1, pt2, color=(255, 255, 0), thickness=3)

        for i in range(data.shape[0]):
            if data[i, 0] == 1:
                continue

            if data[i, 3] < 0.01:
                continue

            pt1, pt2 = yolobbox2bbox(
                data[i, 1], data[i, 2], data[i, 3], data[i, 4], width, height
            )

            sub_img = img[pt1[1] : pt2[1], pt1[0] : pt2[0]]
            dc = DominantColors(sub_img, 2)
            colors = dc.dominant_colors()
            color = (int(colors[0][2]), int(colors[0][1]), int(colors[0][0]))

            cv2.rectangle(img, pt1, pt2, color=color, thickness=3)

            if to_device_from_world is None:
                continue

            foot_point = int((pt1[0] + pt2[0]) / 2)

            X, Z = unproject_to_ground(K, to_device_from_world, foot_point, pt2[1])

            if not is_in_field(X, Z):
                continue

            field = cv2.circle(
                field,
                from_world_to_field(X, Z),
                radius=10,
                color=color,
                thickness=3,
            )

        pt1, pt2 = yolobbox2bbox(
            ball_positions[j, 0],
            ball_positions[j, 1],
            ball_positions[j, 2],
            ball_positions[j, 3],
            width,
            height,
        )
        if to_device_from_world is not None:
            foot_point = int((pt1[0] + pt2[0]) / 2)

            X, Z = unproject_to_ground(K, to_device_from_world, foot_point, pt2[1])

            if is_in_field(X, Z):
                field = cv2.circle(
                    field,
                    from_world_to_field(X, Z),
                    radius=10,
                    color=(255, 255, 0),
                    thickness=3,
                )
        j += 1

        # cv2.rectangle(img, pt1, pt2, color=(255, 255, 0), thickness=3)
        # cv2.imshow("field", field)
        # cv2.waitKey(0)

        # cv2.imwrite(f"top_view/{filename}", field)

        cv2.imshow("Test", img)
        k = cv2.waitKey(0)
        if k == 27:
            break
