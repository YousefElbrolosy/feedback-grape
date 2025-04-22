"""
Configuration file for setuptools.
"""

from setuptools import setup

with open("README.md", "r") as f:
    description = f.read()
if __name__ == "__main__":
    # Installs requried packages
    setup(
        long_description=description,
        long_description_content_type="text/markdown",
    )
