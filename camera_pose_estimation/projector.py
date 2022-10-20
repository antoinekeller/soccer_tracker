import numpy as np
import cv2

from pitch_tracker.key_points import (
    back_middle_line_world,
    front_middle_line_world,
    corner_front_right_world,
    corner_front_left_world,
    corner_back_right_world,
    corner_back_left_world,
)


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


def draw_pitch_lines(K, to_device_from_world, img):
    img = draw_central_circle(K, to_device_from_world, img)
    img = draw_middle_line(K, to_device_from_world, img)
    img = draw_lateral_lines(K, to_device_from_world, img)

    return img
