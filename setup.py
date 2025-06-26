"""
Configuration file for setuptools.
"""

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    description = f.read()
if __name__ == "__main__":
    # Installs requried packages
    setup(
        packages=find_packages(include=["feedback_grape", "feedback_grape.*"]),
        long_description=description,
        long_description_content_type="text/markdown",
    )
