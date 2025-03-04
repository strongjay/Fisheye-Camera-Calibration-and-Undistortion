import cv2
import numpy as np
from fisheye_calibrate import FisheyeCalibrate
from fisheye_undistort import FisheyeUndistort

calibrator = FisheyeCalibrate(checkerboard_size=(6, 8), images_dir=r'/home/work/AWorkSpace/Calibrate_tools/Fisheye_intrinsic/Fisheye-Camera-Calibration-and-Undistortion/imgs3', image_extension='jpg')  # TODO use your image_dir and checkerboard size

# Calculate camera parameters
K, D = calibrator.calculate_parameters()
print('K = ', K)
print('D = ', D)
image = cv2.imread(r'./imgs3/Fisheye1_3.jpg')  # TODO use your image path

# The next method is a simple way to see how well the calibrator did the calibration. 
# However, it's important to note that this method is much slower and less efficient when compared to using the undistort class.
undistorted_image = calibrator.undistort(image, balance=1)  # slow method.

calibration_image_size = calibrator.DIM  # or set it directly
input_size = image.shape[:2][::-1]  # (width, height) of images you're going to undistort
balance = 0
device = 'cpu'  
fisheye_undistorter = FisheyeUndistort(K, D, calibration_image_size, input_size, balance, device, (516 ,389),(1032 ,778))

# Undistort an image
undistorted_image = fisheye_undistorter.undistort(image)

# normal
map3, map4 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (1032,778), cv2.CV_16SC2)
normal_undistorted_image = cv2.remap(image, map3, map4, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow('normal_Undistorted Image', normal_undistorted_image)
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
