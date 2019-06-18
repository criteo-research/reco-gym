import pathlib

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="recogym",
    version="0.1.1.0",
    description="Open-AI gym reinforcement learning environment for recommendation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/criteo-research/reco-gym",
    author="Criteo AI Lab",
    license="Apache License",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[
        "numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "simplegeneric", "gym"
    ],
)
