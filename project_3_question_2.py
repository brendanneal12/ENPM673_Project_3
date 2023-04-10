#Brendan Neal
#Directory ID: bneal12

#Project 3 Question 1 Code

##-----------------------------Importing Libraries---------------------------##
import cv2 as cv
import numpy as np


##-------------------------DEfining my File Names Array----------------------##
Cali_Images = ["Img1.jpg", "Img2.jpg", "Img3.jpg", "Img4.jpg", "Img5.jpg", "Img6.jpg", "Img7.jpg", "Img8.jpg", "Img9.jpg", "Img10.jpg", "Img11.jpg", "Img12.jpg", "Img13.jpg"]
Cali_Image_Idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12]


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
        cv.imshow("Found Corners", Original_Image_Copy); cv.waitKey(1500) 

cv.destroyAllWindows()

success, K, distortion, R, T = cv.calibrateCamera(object_points_array, image_points_array, Gray_Image.shape[::-1], None, None)
print("The Camera Intrinsic Matrix for this Camera is: \n", K)

average_reprojection_error_temp = 0
plotting_reproj_points = []
for i in range(len(object_points_array)):
    reproj_points, _ = cv.projectPoints(object_points_array[i], R[i], T[i], K, distortion)
    plotting_reproj_points.append(reproj_points)

    reproj_error = cv.norm(image_points_array[i], reproj_points, cv.NORM_L2)/len(reproj_points)

    print("The Reprojection Error for Image",i+1,"is", reproj_error)
    average_reprojection_error_temp += reproj_error





average_reprojection_error = average_reprojection_error_temp/len(object_points_array)

print("The average reprojection error for all images is", average_reprojection_error)


for image, i in zip(Cali_Images, Cali_Image_Idxs):
    Original_Image_Copy = cv.imread(image)
    points = plotting_reproj_points[i]
    for point in points:
        cv.circle(Original_Image_Copy, tuple(point[0]), 8, (0,0,255), -1)
        cv.namedWindow("Found Corners with Reprojected Points", cv.WINDOW_NORMAL)
        cv.resizeWindow("Found Corners with Reprojected Points", 500, 400)
        cv.imshow("Found Corners with Reprojected Points", Original_Image_Copy); cv.waitKey(10) 


 







