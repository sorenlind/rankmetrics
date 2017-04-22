"""Setup script for package."""
import re
from distutils.core import Command
from setuptools import setup, find_packages


class PyTest(Command):
    """Setup class for pytest."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run the tests."""
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, "runtests.py"])
        raise SystemExit(errno)


VERSION = re.search(r'^VERSION\s*=\s*"(.*)"', open("rankmetrics/version.py").read(), re.M).group(1)

with open("./README.md", "rb") as f:
    LONG_DESCRIPTION = f.read().decode("utf-8")

setup(
    name="rankmetrics",
    version=VERSION,
    description="Calculate various metrics relevant for learning to rank algorithms.",
    long_description=LONG_DESCRIPTION,
    author="Soren Lind Kristiansen",
    author_email="soren@gutsandglory.dk",
    keywords="recommender system learning to rank python",
    platforms=["Any"],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["numpy", "scipy", "sklearn"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.5',
    ],
    cmdclass={
        "test": PyTest
    }, )
