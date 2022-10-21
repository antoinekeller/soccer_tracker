"""
This module defines KeyPoints class and its correspondance in the 3D world
See the README.md to have a better understanding of the naming of the 3D points
"""

import numpy as np
from .common import draw_point

# Points in world with origin au centre, x going right, y to the foreground, z to the top
# Positions are based on official soccer dimensions
right_circle_world = [9.15, 0, 0]
left_circle_world = [-9.15, 0, 0]
behind_circle_world = [0, 0, 9.15]
front_circle_world = [0, 0, -9.15]
front_middle_line_world = [0, 0, -34]
back_middle_line_world = [0, 0, 34]

corner_back_left_world = [-52.5, 0, 34]
corner_front_left_world = [-52.5, 0, -34]
corner_back_right_world = [52.5, 0, 34]
corner_front_right_world = [52.5, 0, -34]


DIST_TO_CENTER = 77.0


class KeyPoints:
    """
    Class of key points to be used to solve the PnP
    and estimate the focal length
    """

    def __init__(self):
        self.right_circle = None
        self.left_circle = None
        self.behind_circle = None
        self.front_circle = None
        self.front_middle_line = None
        self.back_middle_line = None
        self.corner_back_left = None
        self.corner_back_right = None
        self.corner_front_left = None
        self.corner_front_right = None

    def draw(self, img):
        """Draw all key points on image"""
        img = draw_point(img, self.right_circle)
        img = draw_point(img, self.left_circle)
        img = draw_point(img, self.behind_circle)
        img = draw_point(img, self.front_circle)
        img = draw_point(img, self.front_middle_line)
        img = draw_point(img, self.back_middle_line)
        img = draw_point(img, self.corner_back_left)
        img = draw_point(img, self.corner_back_right)
        img = draw_point(img, self.corner_front_left)
        img = draw_point(img, self.corner_front_right)

        return img

    def __str__(self):
        return (
            f"Right circle: {self.right_circle}\nLeft circle: {self.left_circle}\n"
            f"Behing circle: {self.behind_circle}\nFront circle: {self.front_circle}\n"
            f"Back middle line: {self.back_middle_line}\n"
            f"Front middle line: {self.front_middle_line}\n"
            f"Corner back left: {self.corner_back_left}\n"
            f"Corner back right: {self.corner_back_right}\n"
            f"Corner front left: {self.corner_front_left}\n"
            f"Corner front right: {self.corner_front_right}\n"
        )

    def make_2d_3d_association_list(self):
        """
        Define set of pixels (2D) and its correspondance of points in world (3D)
        to feed the PnP solver.
        """
        pixels = []
        points_world = []

        if self.right_circle is not None:
            pixels.append(self.right_circle)
            points_world.append(right_circle_world)
        if self.left_circle is not None:
            pixels.append(self.left_circle)
            points_world.append(left_circle_world)
        if self.behind_circle is not None:
            pixels.append(self.behind_circle)
            points_world.append(behind_circle_world)
        if self.front_circle is not None:
            pixels.append(self.front_circle)
            points_world.append(front_circle_world)
        if self.front_middle_line is not None:
            pixels.append(self.front_middle_line)
            points_world.append(front_middle_line_world)
        if self.back_middle_line is not None:
            pixels.append(self.back_middle_line)
            points_world.append(back_middle_line_world)
        if self.corner_front_left is not None:
            pixels.append(self.corner_front_left)
            points_world.append(corner_front_left_world)
        if self.corner_front_right is not None:
            pixels.append(self.corner_front_right)
            points_world.append(corner_front_right_world)
        if self.corner_back_left is not None:
            pixels.append(self.corner_back_left)
            points_world.append(corner_back_left_world)
        if self.corner_back_right is not None:
            pixels.append(self.corner_back_right)
            points_world.append(corner_back_right_world)

        pixels = np.array(pixels, dtype=np.float32)
        points_world = np.array(points_world)

        return pixels, points_world

    def compute_focal_length(self, guess_fx):
        """
        Compute the focal length based on the central circle.
        If we cant spot the central circle, we return the default incoming value
        """
        if self.right_circle is None and self.left_circle is None:
            return guess_fx

        if self.right_circle is not None and self.left_circle is not None:
            fx = (
                (self.right_circle[0] - self.left_circle[0])
                * DIST_TO_CENTER
                / (right_circle_world[0] - left_circle_world[0])
            )
            return fx

        if self.behind_circle is None or self.front_circle is None:
            return guess_fx

        central = [
            int((self.behind_circle[0] + self.front_circle[0]) / 2),
            int((self.behind_circle[1] + self.front_circle[1]) / 2),
        ]
        if self.right_circle is None:
            fx = (
                (central[0] - self.left_circle[0])
                * DIST_TO_CENTER
                / (-left_circle_world[0])
            )
            return fx

        if self.left_circle is None:
            fx = (
                (self.right_circle[0] - central[0])
                * DIST_TO_CENTER
                / (right_circle_world[0])
            )
            return fx

        return fx
