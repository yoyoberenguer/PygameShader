

echo off
echo Deleting cpp and pyd files in Pygameshader directory prior installation
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_6%
call %python3_6% setup.py build_ext --inplace
call %python3_6% setup.py bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_7%
call %python3_7% setup.py build_ext --inplace
call %python3_7% setup.py bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_8%
call %python3_8% setup.py build_ext --inplace
call %python3_8% setup.py bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_9%
call %python3_9% setup.py build_ext --inplace
call %python3_9% setup.py bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_10%
call %python3_10% setup.py build_ext --inplace
call %python3_10% setup.py bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_11%
call %python3_11% setup.py build_ext --inplace
call %python3_11% setup.py sdist bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_12%
call %python3_12% setup.py build_ext --inplace
call %python3_12% setup.py sdist bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

echo Building PygameShader with python ver %python3_13%
call %python3_13% setup.py build_ext --inplace
call %python3_13% setup.py sdist bdist_wheel
echo Cleaning up last build
cd PygameShader
del *.cpp
del *.pyd
cd ..

REM @echo off
REM setlocal EnableDelayedExpansion

REM :: Versions to use for building
REM set versions=3_6 3.7 3.8 3.9 3.10 3.11 3.12 3.13
REM set variable_system=python3_6 python3_7 python3_8 python3_9 python3_10

REM echo.
REM echo *********************************************
REM echo + Building PygameShader for Python versions:
REM echo %versions%
REM echo *********************************************
REM echo.

REM echo + Checking installed python versions
REM for %%v in (%variable_system%) do (
REM     echo %v%
REM     if EXIST "%%v" (echo exist) else (echo does not)
REM     echo %python3_6% >nul 2>nul
REM     if errorlevel 1 (
REM         echo python%%v is not found on your system
REM     )
REM )

REM echo.
REM echo + Updating pygame and cython libraries
REM for %%v in (%versions%) do (
REM     where pip%%v >nul 2>nul
REM     if errorlevel 1 (
REM         echo pip%%v is not found on your system
REM     ) else (
REM         echo Using pip%%v to install packages for Python %%v
REM         call pip%%v install --upgrade pygame cython
REM     )
REM )

REM echo.
REM echo + Cleaning previous build artifacts from PygameShader...

REM if exist PygameShader (
REM     echo PygameShader directory exists
REM ) else (
REM     echo PygameShader directory does not exist
REM )

REM cd PygameShader || (echo PygameShader directory not found! & exit /b 1)
REM del /Q *.cpp *.pyd *.so >nul 2>nul
REM cd ..

REM echo.
REM echo + Build wheels

REM for %%v in (%versions%) do (
REM     set "python_bin=python%%v"
REM     echo Building wheel for Python %%v using !python_bin!...

REM     if "%%v"=="3.13" (
REM         call !python_bin! setup.py build_ext --inplace
REM         if !errorlevel! == 0 (
REM             call !python_bin! setup.py sdist bdist_wheel
REM         )
REM     ) else (
REM         call !python_bin! setup.py build_ext --inplace
REM         if !errorlevel! == 0 (
REM             call !python_bin! setup.py bdist_wheel
REM         )
REM     )

REM     cd PygameShader
REM     del /Q *.cpp *.so >nul 2>nul
REM     cd ..
REM )
