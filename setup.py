"""Setup script for package."""
import re
from setuptools import setup, find_packages

VERSION = re.search(r'^VERSION\s*=\s*"(.*)"', open("rankmetrics/version.py").read(), re.M).group(1)
with open("README.rst", "rb") as f:
    LONG_DESCRIPTION = f.read().decode("utf-8")

setup(
    name="rankmetrics",
    version=VERSION,
    description="Calculate various metrics relevant for learning to rank algorithms.",
    long_description=LONG_DESCRIPTION,
    author="Soren Lind Kristiansen",
    author_email="soren@gutsandglory.dk",
    url="https://github.com/sorenlind/rankmetrics/",
    keywords="recommender system learning to rank python",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "sklearn"],
    extras_require={
        'test': ['pytest', 'tox', 'pandas'],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pandas'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6'
    ])
