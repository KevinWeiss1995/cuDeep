"""
pip-installable setup for cuDeep.

Builds the CUDA extension via CMake, then installs the Python package.

Usage:
    pip install -e .
    python setup.py bdist_wheel
"""

import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
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
            "-DCMAKE_BUILD_TYPE=Release",
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
    ext_modules=[CMakeExtension("_cudeep_core")],
    cmdclass={"build_ext": CMakeBuild},
)
