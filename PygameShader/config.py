# encoding: utf-8

from multiprocessing import cpu_count

# TEST_VERSION (True | False) [default=True]
# Determines whether the package being uploaded to PyPI is a test or production version.
# When TEST_VERSION is set to True, the package will be uploaded to https://test.pypi.org/,
# which is the testing environment for PyPI.
# When TEST_VERSION is False, the package will be uploaded to the production environment at https://upload.pypi.org/legacy/.
# To upload the wheel package, use TWINE (install via 'pip install twine').
# Example: twine upload --verbose --repository testpypi dist/*.whl
TEST_VERSION: bool = False

# THREAD_NUMBER (1 - n) [default=1]
# Defines the maximum number of concurrent threads to be used by the CPU shaders library.
# The final value will be automatically adjusted based on the CPU core count.
# If THREAD_NUMBER is set to 1, the shader will run on a single thread by default.
THREAD_NUMBER: int = 1

# OPENMP (True | False)
# Enables or disables multi-threading for the CPU shaders library.
# When set to False, the shaders will run on a single thread.
# When set to True, the library will utilize all available threads according to the cpu_count.
OPENMP: bool = True

# OPENMP_PROC
# Specifies the OpenMP option to enable parallel processing in the shaders.
# This is meant for advanced users who prefer to use OpenMP with LLVM community implementations.
# For most users, the '-fopenmp-simd' option should suffice, but advanced users can opt for '-fopenmp'.
# Note that '-fopenmp' is deprecated and will be removed in future releases.
OPENMP_PROC: str = "-fopenmp"  # "-lgomp"

# __TVERSION__: str [Latest test version for PyPI]
# The latest version of the package to upload to the test PyPI environment.
# This value should be updated manually before testing, and the upload will fail if the version is incorrect.
__TVERSION__: str = "1.0.32"

# __VERSION__: str [Latest production version for PyPI]
# The current version of the package to upload to the production PyPI environment.
__VERSION__: str = "1.0.11"

# LANGUAGE (c | c++) [default="c++"]
# Defines the programming language used for runtime compilation of the CYTHON modules.
# By default, the code will be compiled using C++.
LANGUAGE: str = "c++"

# Adjust version based on the TEST_VERSION flag
if TEST_VERSION:
    __VERSION__: str = __TVERSION__

# Automatically set THREAD_NUMBER based on the number of available CPU cores
try:
    THREAD_NUMBER: int = cpu_count()
except:
    # If there's an issue fetching the CPU core count, THREAD_NUMBER is kept as set.
    pass
