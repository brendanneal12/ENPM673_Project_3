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

##-------------------------Solving for The P Matrix--------------------------##
#w = -1
#v = y
#u = x

P_Calc_Matrix =np.array([[0, 0, 0, 0, -1*world_points_x[0], -1*world_points_y[0], -1*world_points_z[0], -1*1, image_points_y[0]*world_points_x[0], image_points_y[0]*world_points_y[0], image_points_y[0]*world_points_z[0], image_points_y[0]*1],
                         [-1*world_points_x[0], -1*world_points_y[0], -1*world_points_z[0], -1*1, 0, 0, 0, 0, -image_points_x[0]*world_points_x[0], -image_points_x[0]*world_points_y[0], -image_points_x[0]*world_points_z[0], -image_points_x[0]*1],
                         [-image_points_y[0]*world_points_x[0], -image_points_y[0]*world_points_y[0], -image_points_y[0]*world_points_z[0], -image_points_y[0]*1, image_points_x[0]*world_points_x[0], image_points_x[0]*world_points_y[0], image_points_x[0]*world_points_z[0], image_points_x[0]*1, 0, 0, 0,0],
                         
                         [0, 0, 0, 0, -1*world_points_x[1], -1*world_points_y[1], -1*world_points_z[1], -1*1, image_points_y[1]*world_points_x[1], image_points_y[1]*world_points_y[1], image_points_y[1]*world_points_z[1], image_points_y[1]*1],
                         [-1*world_points_x[1], -1*world_points_y[1], -1*world_points_z[1], -1*1, 0, 0, 0, 0, -image_points_x[1]*world_points_x[1], -image_points_x[1]*world_points_y[1], -image_points_x[1]*world_points_z[1], -image_points_x[1]*1],
                         [-image_points_y[1]*world_points_x[1], -image_points_y[1]*world_points_y[1], -image_points_y[1]*world_points_z[1], -image_points_y[1]*1, image_points_x[1]*world_points_x[1], image_points_x[1]*world_points_y[1], image_points_x[1]*world_points_z[1], image_points_x[1]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[2], -1*world_points_y[2], -1*world_points_z[2], -1*1, image_points_y[2]*world_points_x[2], image_points_y[2]*world_points_y[2], image_points_y[2]*world_points_z[2], image_points_y[2]*1],
                         [-1*world_points_x[2], -1*world_points_y[2], -1*world_points_z[2], -1*1, 0, 0, 0, 0, -image_points_x[2]*world_points_x[2], -image_points_x[2]*world_points_y[2], -image_points_x[2]*world_points_z[2], -image_points_x[2]*1],
                         [-image_points_y[2]*world_points_x[2], -image_points_y[2]*world_points_y[2], -image_points_y[2]*world_points_z[2], -image_points_y[2]*1, image_points_x[2]*world_points_x[2], image_points_x[2]*world_points_y[2], image_points_x[2]*world_points_z[2], image_points_x[2]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[3], -1*world_points_y[3], -1*world_points_z[3], -1*1, image_points_y[3]*world_points_x[3], image_points_y[3]*world_points_y[3], image_points_y[3]*world_points_z[3], image_points_y[3]*1],
                         [-1*world_points_x[3], -1*world_points_y[3], -1*world_points_z[3], -1*1, 0, 0, 0, 0, -image_points_x[3]*world_points_x[3], -image_points_x[3]*world_points_y[3], -image_points_x[3]*world_points_z[3], -image_points_x[3]*1],
                         [-image_points_y[3]*world_points_x[3], -image_points_y[3]*world_points_y[3], -image_points_y[3]*world_points_z[3], -image_points_y[3]*1, image_points_x[3]*world_points_x[3], image_points_x[3]*world_points_y[3], image_points_x[3]*world_points_z[3], image_points_x[3]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[4], -1*world_points_y[4], -1*world_points_z[4], -1*1, image_points_y[4]*world_points_x[4], image_points_y[4]*world_points_y[4], image_points_y[4]*world_points_z[4], image_points_y[4]*1],
                         [-1*world_points_x[4], -1*world_points_y[4], -1*world_points_z[4], -1*1, 0, 0, 0, 0, -image_points_x[4]*world_points_x[4], -image_points_x[4]*world_points_y[4], -image_points_x[4]*world_points_z[4], -image_points_x[4]*1],
                         [-image_points_y[4]*world_points_x[4], -image_points_y[4]*world_points_y[4], -image_points_y[4]*world_points_z[4], -image_points_y[4]*1, image_points_x[4]*world_points_x[4], image_points_x[4]*world_points_y[4], image_points_x[4]*world_points_z[4], image_points_x[4]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[5], -1*world_points_y[5], -1*world_points_z[5], -1*1, image_points_y[5]*world_points_x[5], image_points_y[5]*world_points_y[5], image_points_y[5]*world_points_z[5], image_points_y[5]*1],
                         [-1*world_points_x[5], -1*world_points_y[5], -1*world_points_z[5], -1*1, 0, 0, 0, 0, -image_points_x[5]*world_points_x[5], -image_points_x[5]*world_points_y[5], -image_points_x[5]*world_points_z[5], -image_points_x[5]*1],
                         [-image_points_y[5]*world_points_x[5], -image_points_y[5]*world_points_y[5], -image_points_y[5]*world_points_z[5], -image_points_y[5]*1, image_points_x[5]*world_points_x[5], image_points_x[5]*world_points_y[5], image_points_x[5]*world_points_z[5], image_points_x[5]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[6], -1*world_points_y[6], -1*world_points_z[6], -1*1, image_points_y[6]*world_points_x[6], image_points_y[6]*world_points_y[6], image_points_y[6]*world_points_z[6], image_points_y[6]*1],
                         [-1*world_points_x[6], -1*world_points_y[6], -1*world_points_z[6], -1*1, 0, 0, 0, 0, -image_points_x[6]*world_points_x[6], -image_points_x[6]*world_points_y[6], -image_points_x[6]*world_points_z[6], -image_points_x[6]*1],
                         [-image_points_y[6]*world_points_x[6], -image_points_y[6]*world_points_y[6], -image_points_y[6]*world_points_z[6], -image_points_y[6]*1, image_points_x[6]*world_points_x[6], image_points_x[6]*world_points_y[6], image_points_x[6]*world_points_z[6], image_points_x[6]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[7], -1*world_points_y[7], -1*world_points_z[7], -1*1, image_points_y[7]*world_points_x[7], image_points_y[7]*world_points_y[7], image_points_y[7]*world_points_z[7], image_points_y[7]*1],
                         [-1*world_points_x[7], -1*world_points_y[7], -1*world_points_z[7], -1*1, 0, 0, 0, 0, -image_points_x[7]*world_points_x[7], -image_points_x[7]*world_points_y[7], -image_points_x[7]*world_points_z[7], -image_points_x[7]*1],
                         [-image_points_y[7]*world_points_x[7], -image_points_y[7]*world_points_y[7], -image_points_y[7]*world_points_z[7], -image_points_y[7]*1, image_points_x[7]*world_points_x[7], image_points_x[7]*world_points_y[7], image_points_x[7]*world_points_z[7], image_points_x[7]*1, 0, 0, 0,0]
                         ])

