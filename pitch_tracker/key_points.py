from .common import draw_point

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


DIST_TO_CENTER = 77.0


class KeyPoints:
    def __init__(self):
        self.central = None
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
        img = draw_point(img, self.central)
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
        str = f"Central: {self.central}\nRight circle: {self.right_circle}\nLeft circle: {self.left_circle}\nBehing circle: {self.behind_circle}\nFront circle: {self.front_circle}\nBack middle line: {self.back_middle_line}\nFront middle line: {self.front_middle_line}"
        return str

    def compute_fx(self):
        if self.right_circle is None and self.left_circle is None:
            return None

        if self.right_circle is not None and self.left_circle is not None:
            fx = (
                (self.right_circle[0] - self.left_circle[0])
                * DIST_TO_CENTER
                / (right_circle_world[0] - left_circle_world[0])
            )
            return fx

        if self.behind_circle is None or self.front_circle is None:
            return None

        central = [
            int((self.behind_circle[0] + self.front_circle[0]) / 2),
            int((self.behind_circle[1] + self.front_circle[1]) / 2),
        ]
        if self.right_circle is None:
            fx = (
                (central[0] - self.left_circle[0])
                * DIST_TO_CENTER
                / (central_world[0] - left_circle_world[0])
            )
            return fx

        if self.left_circle is None:
            fx = (
                (self.right_circle[0] - central[0])
                * DIST_TO_CENTER
                / (right_circle_world[0] - central_world[0])
            )
            return fx

        return fx
