
echo off
echo Deleting cpp from WHL packages

set version=1.0.10
echo %version%

cd dist

wheel unpack PygameShader-%version%-cp36-cp36m-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp37-cp37m-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q


wheel unpack PygameShader-%version%-cp38-cp38-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp39-cp39-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp310-cp310-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp311-cp311-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

set version=1.0.30
echo %version%

wheel unpack PygameShader-%version%-cp36-cp36m-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp37-cp37m-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q


wheel unpack PygameShader-%version%-cp38-cp38-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp39-cp39-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp310-cp310-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

wheel unpack PygameShader-%version%-cp311-cp311-win_amd64.whl
del PygameShader-%version%\PygameShader\*.cpp
wheel pack PygameShader-%version%
rmdir PygameShader-%version% /S /Q

cd ..
echo finished 