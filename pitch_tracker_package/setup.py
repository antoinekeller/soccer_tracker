"""To easily install code samples."""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pitch_tracker",
    packages=find_packages(include=["pitch_tracker"]),
    author="Antoine",
    description="Pitch tracker module",
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
