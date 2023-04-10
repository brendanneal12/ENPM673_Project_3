#Brendan Neal
#Directory ID: bneal12

#Project 3 Question 1 Code

##-----------------------------Importing Libraries---------------------------##
import cv2 as cv
import numpy as np


##-------------------------DEfining my File Names Array----------------------##
Cali_Images = ["Img1.jpg", "Img2.jpg", "Img3.jpg", "Img4.jpg", "Img5.jpg", "Img6.jpg", "Img7.jpg", "Img8.jpg", "Img9.jpg", "Img10.jpg", "Img11.jpg", "Img12.jpg", "Img13.jpg"]
Cali_Image_Idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12]





##----------------------------Camera Calibration Pipeline---------------------##
''' This pipeline was adapted from OpenCV's official camera calibration example.
It can be found here: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html'''

termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) #Set Termination Criteria
object_points = np.zeros((9*6, 3), np.float32) #Set object points array the size of your calibration target. (9x6)
object_points[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) #Create a meshgrid for the object points

object_points_array = [] #Initialize object points array.
image_points_array = [] #Initialize image points array.

for image in Cali_Images: #For each calibration image
    Original_Image = cv.imread(image) #Read the Image
    Original_Image_Copy = cv.imread(image) #Create a copy to draw on.
    Gray_Image = cv.cvtColor(Original_Image, cv.COLOR_BGR2GRAY) #Convert the color to grayscale.

    Success, FoundCorners = cv.findChessboardCorners(Gray_Image, (9,6), None) #Find the checkerboard corners.


    if Success: #If you have successfully found the corners:
        object_points_array.append(object_points) #Append those corresponding object points to the array.
        better_corners = cv.cornerSubPix(Gray_Image, FoundCorners, (11,11), (-1,-1), termination_criteria) #Refine the found corners in the image.
        image_points_array.append(better_corners) #Append them in the image points array.

        cv.drawChessboardCorners(Original_Image_Copy, (9,6), better_corners, Success) #Draw the chessboard corners.

        ##-----Display Chessboard Corners----##
        cv.namedWindow("Found Corners", cv.WINDOW_NORMAL)
        cv.resizeWindow("Found Corners", 500, 400)
        cv.imshow("Found Corners", Original_Image_Copy); cv.waitKey(1500) 

cv.destroyAllWindows()

#Calibrate the Camera based on the corresponding object points and image points.
success, K, distortion, R, T = cv.calibrateCamera(object_points_array, image_points_array, Gray_Image.shape[::-1], None, None)
print("The Camera Intrinsic Matrix for this Camera is: \n", K)


##------------------------------Reprojection Error Pipeline------------------------------##
'''This pipeline was adapted from OpenCV's official camera calibration example.
It can be found here: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html'''

average_reprojection_error_temp = 0 #Start a running average for all images.
plotting_reproj_points = [] #Initialize an array for plotting

for i in range(len(object_points_array)): #For Each image
    reproj_points, _ = cv.projectPoints(object_points_array[i], R[i], T[i], K, distortion) #Reproject the Points
    plotting_reproj_points.append(reproj_points) #Append the reprojected points to the plotting array.

    reproj_error = cv.norm(image_points_array[i], reproj_points, cv.NORM_L2)/len(reproj_points) #Get the average reprojection error for a single image.

    print("The Reprojection Error for Image",i+1,"is", reproj_error)
    average_reprojection_error_temp += reproj_error





average_reprojection_error = average_reprojection_error_temp/len(object_points_array) #Calculate the reprojection error average for all 13 images.

print("The average reprojection error for all images is", average_reprojection_error)


##----------------------------Plotting the Reprojected Points------------------------------##
for image, i in zip(Cali_Images, Cali_Image_Idxs): #For each image
    Original_Image_Copy = cv.imread(image) #Read the Image
    points = plotting_reproj_points[i] #Grab the reprojected points associated to that image
    for point in points:
        cv.circle(Original_Image_Copy, tuple(point[0]), 8, (0,0,255), -1) #Plot the points
        #Display the Image.
        cv.namedWindow("Reprojected Points", cv.WINDOW_NORMAL)
        cv.resizeWindow("Reprojected Points", 500, 400)
        cv.imshow("Reprojected Points", Original_Image_Copy); cv.waitKey(10) 


 







