# encoding: utf-8

from multiprocessing import cpu_count

# TEST_VERSION (True | False) default = True
# Determine the version number to use when uploading the package to PyPI.
# The python wheel package will be build with the current
# version number define with the variable __VERSION__ or __TVERSION__ for testing.
# A test version will be uploaded to https://test.pypi.org/
# while the production version will be uploaded to https://upload.pypi.org/legacy/
# To upload the wheel package use TWINE (pip install twine).
# e.g twine upload --verbose --repository testpypi dist/*.whl
TEST_VERSION : bool = False

# THREAD_NUMBER (1 - n) default THREAD_NUMBER = 1
# Only for the CPU shaders.
# Represent the maximum concurrent threads that will be used by the CPU shaders library.
# This is the default value. The final value will be taken from the cpu_count (see below).
THREAD_NUMBER : int = 1

# Variable OPENMP (True | False)
# Set the multiprocessing capability for the CPU shaders.
# When the variable is set to False, the shader will work on a single thread.
# If set to True, the cpu shader lib will work with the maximum allowed thread number set
# by the cpu_count.
OPENMP : bool = True

# OPENMP_PROC (
# Enables recognition of OpenMP* features and tells the parallelizer to generate
# multi-threaded code based on OpenMP* directives.
# This option is meant for advanced users who prefer to use OpenMP* as it is implemented
# by the LLVM community. You can get most of that functionality by using this option
# and option -fopenmp-simd.
# Option -fopenmp is a deprecated option that will be removed in a future release.
# For most users, we recommend that you instead use option qopenmp, Qopenmp.
OPENMP_PROC : str = "-fopenmp"  # "-lgomp"

# Latest version to upload to the TEST environment (PyPI)
# Log into the project and check the latest version. The upload will fail
# if the version is incorrect
__TVERSION__ : str  = "1.0.30"

# Latest version to upload to PygameShader project PyPI (production)
__VERSION__ : str = "1.0.10"

# LANGUAGE (c | c++) default c++
# Language to use for the runtime compilation. All the CYTHON module will
# be build with this language.
LANGUAGE : str = "c++"

if TEST_VERSION:
    __VERSION__ : str = __TVERSION__

# Set automatically the maximum allowed threads
try:
    THREAD_NUMBER : int = cpu_count()
except:
    # ignore issue
    # THREAD_NUMBER is already set
    ...

