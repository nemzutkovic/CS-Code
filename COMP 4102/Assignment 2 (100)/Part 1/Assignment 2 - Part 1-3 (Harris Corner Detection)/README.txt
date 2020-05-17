Name: Nem Zutkovic
Student #: 101085982
Date: March 18, 2020

OS Used: Windows 10
OpenCV Version Used: 3.2

1. Open the HarrisCornerDetection.sln file in Visual Studio.
2. Change the build configuration to `Release` and have the architecture set to `x64`.
3A. Click `Local Windows Debugger` and 6 windows will appear showing `img0.jpg` in these various states (screenshot attached):
- Minimum Image Output
- Maximum Image Output
- R Image Output
- Original Image
- Final Output Image
3B. Important: You can adjust the `EIGENVALUE` in `HarrisCornerDetection.cpp` to adjust the amount of corners that are detected.

**Please note: 
- I do not know if this code will run on a macOS/Linux environment or on other architectures othan than x64.
- You will also have to alter the project properties to include/link your OpenCV library, because our paths are different.
