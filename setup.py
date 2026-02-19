"""
pip-installable setup for cuDeep.

Builds the CUDA extension via CMake, then installs the Python package.

Usage:
    pip install -e .
    python setup.py bdist_wheel
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        source_dir = Path(__file__).parent.resolve()

        build_dir = source_dir / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCUDEEP_BUILD_PYTHON=ON",
            "-DCUDEEP_BUILD_TESTS=OFF",
            "-DCUDEEP_BUILD_BENCHMARKS=OFF",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--config", "Release", "-j"]

        subprocess.check_call(
            ["cmake", str(source_dir)] + cmake_args,
            cwd=build_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir,
        )


setup(
    name="cuDeep",
    version="0.1.0",
    author="Kevin",
    description="Ultra-high performance deep learning library implemented in CUDA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/cuDeep",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[CMakeExtension("_cudeep_core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black",
            "sphinx",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
