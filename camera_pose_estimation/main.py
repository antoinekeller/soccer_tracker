import cv2
import numpy as np
import os
from argparse import ArgumentParser

from pitch_tracker.common import intersect

from pitch_tracker.main import find_key_points


# Points in world with origin au centre, x a droite, y vers le fond, z en haut
central_world = [0, 0, 0]
right_circle_world = [9.15, 0, 0]
left_circle_world = [-9.15, 0, 0]
behind_circle_world = [0, 0, 9.15]
front_circle_world = [0, 0, -9.15]
front_middle_line_world = [0, 0, -34]
back_middle_line_world = [0, 0, 34]
test_point_world = [30, 0, 34]
help_point_world = [-30, 0, 34]

corner_back_left_world = [-52.5, 0, 34]
corner_front_left_world = [-52.5, 0, -34]
corner_back_right_world = [52.5, 0, 34]
corner_front_right_world = [52.5, 0, -34]


def find_extrinsic_intrinsic_matrices(img, fx, guess_rot, guess_trans, key_points):
    height, width = img.shape[0], img.shape[1]

    print(guess_rot, guess_trans)

    K = np.zeros((3, 3))

    K[0, 0] = fx
    K[1, 1] = fx
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    K[2, 2] = 1

    pixels = []
    points_world = []
    if key_points.right_circle is not None:
        pixels.append(key_points.right_circle)
        points_world.append(right_circle_world)
    if key_points.left_circle is not None:
        pixels.append(key_points.left_circle)
        points_world.append(left_circle_world)
    if key_points.behind_circle is not None:
        pixels.append(key_points.behind_circle)
        points_world.append(behind_circle_world)
    if key_points.front_circle is not None:
        pixels.append(key_points.front_circle)
        points_world.append(front_circle_world)
    if key_points.front_middle_line is not None:
        pixels.append(key_points.front_middle_line)
        points_world.append(front_middle_line_world)
    if key_points.back_middle_line is not None:
        pixels.append(key_points.back_middle_line)
        points_world.append(back_middle_line_world)
    if key_points.corner_front_left is not None:
        pixels.append(key_points.corner_front_left)
        points_world.append(corner_front_left_world)
    if key_points.corner_front_right is not None:
        pixels.append(key_points.corner_front_right)
        points_world.append(corner_front_right_world)
    if key_points.corner_back_left is not None:
        pixels.append(key_points.corner_back_left)
        points_world.append(corner_back_left_world)
    if key_points.corner_back_right is not None:
        pixels.append(key_points.corner_back_right)
        points_world.append(corner_back_right_world)

    print(f"Solving PnP with {len(pixels)} points")

    pixels = np.array(pixels, dtype=np.float32)

    if pixels.shape[0] <= 3:
        print("Too few points to solve!")
        return None, K, guess_rot, guess_trans

    points_world = np.array(points_world)

    rotation_vector = guess_rot
    translation_vector = guess_trans

    i = 30

    while i > 0:

        K[0, 0] = fx
        K[1, 1] = fx

        (ret, rotation_vector, translation_vector) = cv2.solvePnP(
            points_world,
            pixels,
            K,
            distCoeffs=None,
            rvec=rotation_vector,
            tvec=translation_vector,
            useExtrinsicGuess=True,
        )

        if rotation_vector[0][0] != rotation_vector[0][0]:
            print("BREAK")
            return None, None, guess_rot, guess_trans

        print(ret, rotation_vector, translation_vector)

        fx = key_points.compute_fx()
        i -= 1

    # in the reference world
    to_device_from_world_rot = cv2.Rodrigues(rotation_vector)[0]

    # to_world_from_device
    camera_position_in_world = -np.matrix(to_device_from_world_rot).T * np.matrix(
        translation_vector
    )

    print(rotation_vector, translation_vector)

    print(
        f"Camera is located at {-camera_position_in_world[1,0]:.1f}m high and at {-camera_position_in_world[2,0]:.1f}m depth"
    )
    if fx is None:
        print("CRAZY VALUE!!!")
        return None, None, guess_rot, guess_trans

    dist_to_center = np.linalg.norm(camera_position_in_world)
    print(f"Final fx = {fx:.1f}. Distance to origin = {dist_to_center:.1f}m")
    if dist_to_center > 100.0:
        print("CRAZY VALUE!!!")
        return None, K, guess_rot, guess_trans

    to_device_from_world = np.identity(4)
    to_device_from_world[0:3, 0:3] = to_device_from_world_rot
    to_device_from_world[0:3, 3] = translation_vector.reshape((3,))

    return to_device_from_world, K, rotation_vector, translation_vector


def project_to_screen(K, to_device_from_world, point_in_world):
    homog = np.ones((4, 1))
    homog[0:3, 0] = point_in_world
    point_in_device = np.dot(to_device_from_world, homog)
    point_in_device_divided_depth = (point_in_device / point_in_device[2, 0])[0:3]
    # print(point_in_device_divided_depth)
    point_projected = np.dot(K, point_in_device_divided_depth)
    return [int(point_projected[0]), int(point_projected[1])]


def draw_central_circle(K, to_device_from_world, img):
    # Draw circle
    res = 25
    circle_points_projected = np.zeros((res, 2), dtype=np.int32)
    for i in range(res):
        angle = i / res * np.pi * 2
        circle_points_world = np.array([np.cos(angle), 0, np.sin(angle)]) * 9.15
        circle_points_projected[i] = project_to_screen(
            K, to_device_from_world, circle_points_world
        )

    img = cv2.polylines(
        img, [circle_points_projected], isClosed=True, color=(0, 165, 255), thickness=3
    )

    return img


def draw_middle_line(K, to_device_from_world, img):
    # Draw middle line
    back_middle_line_projected = project_to_screen(
        K, to_device_from_world, np.array(back_middle_line_world)
    )
    front_middle_line_projected = project_to_screen(
        K, to_device_from_world, np.array(front_middle_line_world)
    )
    img = cv2.line(
        img,
        back_middle_line_projected,
        front_middle_line_projected,
        color=(0, 165, 255),
        thickness=3,
    )

    return img


def project_to_screen(K, to_device_from_world, point_in_world):
    homog = np.ones((4, 1))
    homog[0:3, 0] = point_in_world
    point_in_device = np.dot(to_device_from_world, homog)
    point_in_device_divided_depth = (point_in_device / point_in_device[2, 0])[0:3]
    # print(point_in_device_divided_depth)
    point_projected = np.dot(K, point_in_device_divided_depth)
    return [int(point_projected[0]), int(point_projected[1])]


def draw_central_circle(K, to_device_from_world, img):
    # Draw circle
    res = 25
    circle_points_projected = np.zeros((res, 2), dtype=np.int32)
    for i in range(res):
        angle = i / res * np.pi * 2
        circle_points_world = np.array([np.cos(angle), 0, np.sin(angle)]) * 9.15
        circle_points_projected[i] = project_to_screen(
            K, to_device_from_world, circle_points_world
        )

    img = cv2.polylines(
        img, [circle_points_projected], isClosed=True, color=(0, 165, 255), thickness=3
    )

    return img


def draw_lateral_lines(K, to_device_from_world, img):
    corner_back_left_projected = project_to_screen(
        K, to_device_from_world, np.array(corner_back_left_world)
    )
    corner_front_left_projected = project_to_screen(
        K, to_device_from_world, np.array(corner_front_left_world)
    )
    corner_back_right_projected = project_to_screen(
        K, to_device_from_world, np.array(corner_back_right_world)
    )
    corner_front_right_projected = project_to_screen(
        K, to_device_from_world, np.array(corner_front_right_world)
    )

    img = cv2.line(
        img,
        corner_front_left_projected,
        corner_front_right_projected,
        color=(0, 165, 255),
        thickness=3,
    )

    img = cv2.line(
        img,
        corner_front_right_projected,
        corner_back_right_projected,
        color=(0, 165, 255),
        thickness=3,
    )

    img = cv2.line(
        img,
        corner_back_right_projected,
        corner_back_left_projected,
        color=(0, 165, 255),
        thickness=3,
    )

    img = cv2.line(
        img,
        corner_back_left_projected,
        corner_front_left_projected,
        color=(0, 165, 255),
        thickness=3,
    )

    return img


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


def calibrate_from_image(img, guess_fx, guess_rot, guess_trans, debug=False, out=False):

    key_points, key_lines = find_key_points(img)

    # cv2.imshow("Draw key points", img)
    # cv2.waitKey(0)

    to_device_from_world, K, guess_rot, guess_trans = find_extrinsic_intrinsic_matrices(
        img, guess_fx, guess_rot, guess_trans, key_points
    )

    if to_device_from_world is None:
        img = key_points.draw(img)
        if not out:
            cv2.imshow("Test", img)
            cv2.waitKey(0)
        else:
            cv2.imwrite(os.path.join("images_line_detected/", filename), img)
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
    img = key_points.draw(img)

    guess_fx = K[0, 0]
    # cv2.imshow("Test", img)
    # cv2.waitKey(0)
    to_device_from_world, K, guess_rot, guess_trans = find_extrinsic_intrinsic_matrices(
        img, guess_fx, guess_rot, guess_trans, key_points
    )

    if to_device_from_world is not None:
        img = draw_central_circle(K, to_device_from_world, img)
        img = draw_middle_line(K, to_device_from_world, img)
        img = draw_lateral_lines(K, to_device_from_world, img)

    return K, to_device_from_world, guess_rot, guess_trans, img


if __name__ == "__main__":
    guess_fx = 2000

    parser = ArgumentParser(description="Main script to find K, extrinsic pose")
    parser.add_argument("--idx", type=int, help="Image index", nargs="*")
    parser.add_argument(
        "--debug",
        action="store_const",
        const=True,
        default=False,
        help="Debug",
    )
    parser.add_argument(
        "--out",
        action="store_const",
        const=True,
        default=False,
        help="Debug",
    )
    args = parser.parse_args()

    list_of_files = (
        sorted(os.listdir("../kaggle/images_line_detection/"))
        if args.idx is None
        else [f"image_{idx:03d}.png" for idx in args.idx]
    )

    guess_rot = (0.25, 0, 0)
    guess_trans = (0, 0, 80)

    for filename in list_of_files:
        print(" ")
        f = os.path.join("../kaggle/images_line_detection/", filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            img = cv2.imread(f)

            print(guess_rot)

            K, to_device_from_world, rot, trans, img = calibrate_from_image(
                img, guess_fx, guess_rot, guess_trans, args.debug, args.out
            )

            guess_rot = rot
            guess_trans = trans

            if not args.out:
                cv2.imshow("Test", img)
                k = cv2.waitKey(0)
                if k == 27:
                    break
            else:
                cv2.imwrite(os.path.join("images_line_detected/", filename), img)
