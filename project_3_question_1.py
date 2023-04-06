#Brendan Neal
#Directory ID: bneal12

#Project 3 Question 1 Code

##-----------------------------Importing Libraries---------------------------##
import numpy as np

##-------------------Setting Up Arrays from Project Document-----------------##

image_points_x = np.array([757, 758, 758, 759, 1190, 329, 1204, 340])
image_points_y = np.array([213, 415, 686, 966, 172, 1041, 850, 159])
world_points_x = np.array([0, 0, 0, 0, 7, 0, 7, 0])
world_points_y = np.array([0, 3, 7, 11, 1, 11, 9, 1])
world_points_z = np.array([0, 0, 0, 0, 0, 7, 0, 7])

##-------------------Mathematical Camera Calibration Script------------------##
#Rename Cali Images