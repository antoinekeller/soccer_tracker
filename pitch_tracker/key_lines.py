from pitch_tracker.common import draw_line


class KeyLines:
    def __init__(self):
        self.front_line = None
        self.back_line = None
        self.main_line = None
        self.left_goal_line = None
        self.right_goal_line = None

    def draw(self, img):
        img = draw_line(img, self.front_line, "red")
        img = draw_line(img, self.back_line, "red")
        img = draw_line(img, self.main_line, "green")
        img = draw_line(img, self.left_goal_line, "blue")
        img = draw_line(img, self.right_goal_line, "blue")

        return img
