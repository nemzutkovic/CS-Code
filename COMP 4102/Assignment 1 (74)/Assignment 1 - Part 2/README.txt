Name: Nem Zutkovic
Student #: 101085982
Date: February 11, 2020

OS Used: Windows 10
OpenCV Version Used: 3.2

1. Open the EdgeDetection.sln file in Visual Studio.
2. Change the build configuration to `Release` and have the architecture set to `x64`.
3. Click `Local Windows Debugger` and 6 windows will appear showing `img0.jpg` in these various states:
- Original Image
- Greyscale Image
- Gaussian Blur Image
- Gradient Magnitude Image (Without Non-Maximum Suppression)
- Gradient Magnitude Image (With Non-Maximum Suppression)
- Gradient Orientation

**Please note: 
- I do not know if this code will run on a macOS/Linux environment or on other architectures othan than x64.
- You will also have to alter the project properties to include/link your OpenCV library, because our paths are different.
