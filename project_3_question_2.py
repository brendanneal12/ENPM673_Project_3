#Brendan Neal
#Directory ID: bneal12

#Project 3 Question 1 Code

##-----------------------------Importing Libraries---------------------------##
import cv2 as cv
import numpy as np


##-------------------------DEfining my File Names Array----------------------##
Cali_Images = ["Img1.jpg", "Img2.jpg", "Img3.jpg", "Img4.jpg", "Img5.jpg", "Img6.jpg", "Img7.jpg", "Img8.jpg", "Img9.jpg", "Img10.jpg", "Img11.jpg", "Img12.jpg", "Img13.jpg"]



termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
object_points = np.zeros((9*6, 3), np.float32)
object_points[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

object_points_array = []
image_points_array = []

for image in Cali_Images:
    Original_Image = cv.imread(image)
    Original_Image_Copy = cv.imread(image)
    Gray_Image = cv.cvtColor(Original_Image, cv.COLOR_BGR2GRAY)

    Success, FoundCorners = cv.findChessboardCorners(Gray_Image, (9,6), None)


    if Success:
        object_points_array.append(object_points)
        better_corners = cv.cornerSubPix(Gray_Image, FoundCorners, (11,11), (-1,-1), termination_criteria)
        image_points_array.append(better_corners)

        cv.drawChessboardCorners(Original_Image_Copy, (9,6), better_corners, Success)
        cv.namedWindow("Found Corners", cv.WINDOW_NORMAL)
        cv.resizeWindow("Found Corners", 500, 400)
        cv.imshow("Found Corners", Original_Image_Copy); cv.waitKey(1) #Display Gray Scale Image

cv.destroyAllWindows()

success, K, distortion, R, T = cv.calibrateCamera(object_points_array, image_points_array, Gray_Image.shape[::-1], None, None)
#print(R[1])
print("The Camera Intrinsic Matrix for this Camera is: \n", K)
#print("The Projection Matrix for Image Number",)
#ProjMatrix = K*[R,T]

 







