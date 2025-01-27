from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([
        "cython_modules/glv_functions.pyx",
        "cython_modules/null_model_functions.pyx",
        "cython_modules/similarity_correlation_functions.pyx",
    ]),
    include_dirs=[numpy.get_include()]
)