"""This module defines useful functions to manipulate images"""

import cv2
import numpy as np


def draw_point(img, point):
    """Draw key point with opencv on image"""
    if point is None:
        return img
    img = cv2.circle(
        img,
        center=(int(point[0]), int(point[1])),
        radius=5,
        color=(0, 0, 255),
        thickness=5,
    )

    return img


def draw_line(img, line, color="red"):
    """Draw unlimited line on image with some color
    Line is defined in polar coordinates
    """

    if line is None:
        return img

    rho = line[0]
    theta = line[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
    pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
    bgr = (255, 255, 255)
    if color == "red":
        bgr = (0, 0, 255)
    elif color == "blue":
        bgr = (255, 0, 0)
    elif color == "green":
        bgr = (0, 255, 0)
    cv2.line(img, pt1, pt2, bgr, 3, cv2.LINE_AA)

    return img


def intersect(line_1, line_2):
    """
    Find intersection of two lines
    """

    if line_1 is None or line_2 is None:
        return None

    rho_1 = line_1[0]
    theta_1 = line_1[1]
    rho_2 = line_2[0]
    theta_2 = line_2[1]

    if theta_1 == theta_2:
        return None

    u = (rho_1 * np.sin(theta_2) - rho_2 * np.sin(theta_1)) / np.sin(theta_2 - theta_1)
    if theta_1 != 0:
        v = (rho_1 - u * np.cos(theta_1)) / np.sin(theta_1)
    else:
        v = (rho_2 - u * np.cos(theta_2)) / np.sin(theta_2)

    return [int(u), int(v)]
