# Soccer players & ball tracker with camera pose estimation

![plot](./solved.png)

Goal of this project is to generate a top-view image of a soccer game with the players and the ball on it, potentially to generate statistics of the game.

This requires several computer vision algorithms. Here are described the main steps:
- first detect the main lines and corners on the soccer pitch like the central circle, the lateral lines, the goal lines etc.
- with those key points, find the optimal camera pose and focal length to match the observation with a real soccer pitch with known dimensions.
- On the other hand, use YOLOv5 neural network to detect players and balls on the image.
- Then reproject 2D bounding boxes to the world coordinates and watch the top view. 


## Demo

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/tXa_gfmQnmw/0.jpg)](https://www.youtube.com/watch?v=tXa_gfmQnmw&ab_channel=antoinekeller)

## Repo description

```
.
├── images                        # 2 images of sudoku to use with sudoku_cv.py
│   ├── 1.jpeg
│   └── 2.png
├── ocr                           # Optical Character Recognition folder
│   ├── confusion_matrix.png      # used in README.md
│   ├── fonts                     # 10 common fonts for the neural network training
│   │   ├── arial.ttf
│   │   ├── calibri.ttf
│   │   ├── Cambria.ttf
│   │   ├── FranklinGothic.ttf
│   │   ├── futur.ttf
│   │   ├── Garamond.ttf
│   │   ├── Helvetica 400.ttf
│   │   ├── rock.ttf
│   │   ├── times.ttf
│   │   └── verdana.ttf
│   ├── model.h5                  # CNN weights
│   ├── model.json                # CNN description
│   ├── ocr_trainer.py            # The Keras neural network trainer
│   ├── README.md
│   └── test_examples.png         # used in README.md
├── README.md
├── solved.png                    # used in README.md
├── solver.py                     # the core algorithm to solve a sudoku
├── sudoku_cv.py                  # useful script to detect/solve a sudoku on an image
├── sudoku_locator.py             # class that detects and localize the grid
└── sudoku_webcam.py              # the main script that uses the webcam to detect/solve/track the sudoku
```

## Installation

This was tested under Python 3.8.13 in a virutal environment.

Please run:
```
cd camera_pose_estimation_package
pip install -e .
cd ../pitch_tracker_package
pip install -e .
```