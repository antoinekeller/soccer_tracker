import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from pitch_tracker.common import intersect

from pitch_tracker.main import find_key_points
from projector import (
    draw_pitch_lines,
    project_to_screen,
)

from points_in_world import *


def find_extrinsic_intrinsic_matrices(img, fx, guess_rot, guess_trans, key_points):
    height, width = img.shape[0], img.shape[1]

    pixels, points_world = key_points.make_2d_3d_association_list()

    print(f"Solving PnP with {len(pixels)} points")

    # Build camera projection matrix
    est_fx = key_points.compute_fx()
    if est_fx is None:
        est_fx = fx

    K = np.array([[est_fx, 0, width / 2], [0, est_fx, height / 2], [0, 0, 1]])

    if pixels.shape[0] <= 3:
        print("Too few points to solve!")
        return None, K, guess_rot, guess_trans

    (ret, rotation_vector, translation_vector) = cv2.solvePnP(
        points_world,
        pixels,
        K,
        distCoeffs=None,
        rvec=guess_rot,
        tvec=guess_trans,
        useExtrinsicGuess=True,
    )

    assert ret

    if np.isnan(rotation_vector[0, 0]):
        print("PnP could not be solved correctly --> Skip")
        return None, None, guess_rot, guess_trans

    # in the reference world
    to_device_from_world_rot = cv2.Rodrigues(rotation_vector)[0]

    # to_world_from_device
    camera_position_in_world = -np.matrix(to_device_from_world_rot).T * np.matrix(
        translation_vector
    )

    print(
        f"Camera is located at {-camera_position_in_world[1,0]:.1f}m high and at {-camera_position_in_world[2,0]:.1f}m depth"
    )
    if est_fx is None:
        print("CRAZY VALUE!!!")
        return None, None, guess_rot, guess_trans

    dist_to_center = np.linalg.norm(camera_position_in_world)
    print(f"Final fx = {est_fx:.1f}. Distance to origin = {dist_to_center:.1f}m")
    if dist_to_center > 100.0:
        print("PnP: CRAZY VALUE!!!")
        return None, K, guess_rot, guess_trans

    to_device_from_world = np.identity(4)
    to_device_from_world[0:3, 0:3] = to_device_from_world_rot
    to_device_from_world[0:3, 3] = translation_vector.reshape((3,))

    return to_device_from_world, K, rotation_vector, translation_vector


def find_closer_point_on_line(point, line):
    rho = line[0]
    theta = line[1]
    point = np.array(point)

    pt_line_origin = np.array([0, rho / np.sin(theta)])
    a = point - pt_line_origin
    u = np.array([np.sin(theta), -np.cos(theta)])

    _lambda = np.dot(a, u)

    projected_point = pt_line_origin + _lambda * u

    projected_point = [int(projected_point[0]), int(projected_point[1])]

    return projected_point


def calibrate_from_image(img, guess_fx, guess_rot, guess_trans):

    key_points, key_lines = find_key_points(img)

    # cv2.imshow("Draw key points", img)
    # cv2.waitKey(0)

    assert not np.isnan(guess_rot[0, 0])

    to_device_from_world, K, guess_rot, guess_trans = find_extrinsic_intrinsic_matrices(
        img, guess_fx, guess_rot, guess_trans, key_points
    )

    if to_device_from_world is None:
        return K, to_device_from_world, guess_rot, guess_trans, img

    if key_points.corner_back_right is None and key_points.corner_back_left is None:
        pt = project_to_screen(K, to_device_from_world, corner_front_right_world)
        key_points.corner_front_right = find_closer_point_on_line(
            pt, key_lines.front_line
        )
        pt = project_to_screen(K, to_device_from_world, corner_front_left_world)
        key_points.corner_front_left = find_closer_point_on_line(
            pt, key_lines.front_line
        )
        pt = project_to_screen(K, to_device_from_world, corner_back_right_world)
        key_points.corner_back_right = find_closer_point_on_line(
            pt, key_lines.back_line
        )
        pt = project_to_screen(K, to_device_from_world, corner_back_left_world)
        key_points.corner_back_left = find_closer_point_on_line(pt, key_lines.back_line)
    # draw_line(img, right_goal_line)
    if (
        key_points.corner_back_right is not None
        and key_lines.right_goal_line is not None
    ):
        key_points.corner_front_right = intersect(
            key_lines.right_goal_line, key_lines.front_line
        )
    if key_points.corner_back_left is not None and key_lines.left_goal_line is not None:
        key_points.corner_front_left = intersect(
            key_lines.left_goal_line, key_lines.front_line
        )

    guess_fx = K[0, 0]
    # cv2.imshow("Test", img)
    # cv2.waitKey(0)
    to_device_from_world, K, found_rot, found_trans = find_extrinsic_intrinsic_matrices(
        img, guess_fx, guess_rot, guess_trans, key_points
    )

    return K, to_device_from_world, found_rot, found_trans, img


if __name__ == "__main__":
    guess_fx = 2000

    parser = ArgumentParser(
        description="Main script to key points on a soccer field image"
    )
    parser.add_argument("input", type=str, help="Image path or folder")
    parser.add_argument(
        "--out",
        action="store_const",
        const=True,
        default=False,
        help="Debug",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileExistsError

    images = (
        sorted(Path(args.input).glob("**/*"))
        if Path(args.input).is_dir()
        else [Path(args.input)]
    )

    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)

    for filename in images:
        if not filename.exists():
            continue

        print(str(filename))
        img = cv2.imread(str(filename))

        key_points, key_lines = find_key_points(img)
        img = key_points.draw(img)

        K, to_device_from_world, rot, trans, img = calibrate_from_image(
            img, guess_fx, guess_rot, guess_trans
        )

        if to_device_from_world is not None:
            img = draw_pitch_lines(K, to_device_from_world, img)

        guess_rot = (
            rot if to_device_from_world is not None else np.array([[0.25, 0, 0]])
        )
        guess_trans = trans if to_device_from_world is not None else (0, 0, 80)

        if not args.out:
            cv2.imshow("Test", img)
            k = cv2.waitKey(0)
            if k == 27:
                break
        else:
            out_path = Path("camera_pose_estimation/out").joinpath(Path(filename).name)
            print(f"Writing image to {str(out_path)}")
            cv2.imwrite(str(out_path), img)
