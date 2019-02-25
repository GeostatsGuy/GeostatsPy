import setuptools
from geostatspy import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geostatspy",
    version=__version__,
    author="Michael Pyrcz",
    author_email="mpyrcz@austin.utexas.edu",
    description="Geostatistical methods from GSLIB: Geostatistical Library translated and reimplemented in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeostatsGuy/GeostatsPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
