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
#w = 1
#v = y
#u = x

P_Calc_Matrix =np.array([[0, 0, 0, 0, -1*world_points_x[0], -1*world_points_y[0], -1*world_points_z[0], -1*1, image_points_y[0]*world_points_x[0], image_points_y[0]*world_points_y[0], image_points_y[0]*world_points_z[0], image_points_y[0]*1],
                         [1*world_points_x[0], 1*world_points_y[0], 1*world_points_z[0], 1*1, 0, 0, 0, 0, -image_points_x[0]*world_points_x[0], -image_points_x[0]*world_points_y[0], -image_points_x[0]*world_points_z[0], -image_points_x[0]*1],
                         [-image_points_y[0]*world_points_x[0], -image_points_y[0]*world_points_y[0], -image_points_y[0]*world_points_z[0], -image_points_y[0]*1, image_points_x[0]*world_points_x[0], image_points_x[0]*world_points_y[0], image_points_x[0]*world_points_z[0], image_points_x[0]*1, 0, 0, 0,0],
                         
                         [0, 0, 0, 0, -1*world_points_x[1], -1*world_points_y[1], -1*world_points_z[1], -1*1, image_points_y[1]*world_points_x[1], image_points_y[1]*world_points_y[1], image_points_y[1]*world_points_z[1], image_points_y[1]*1],
                         [1*world_points_x[1], 1*world_points_y[1], 1*world_points_z[1], 1*1, 0, 0, 0, 0, -image_points_x[1]*world_points_x[1], -image_points_x[1]*world_points_y[1], -image_points_x[1]*world_points_z[1], -image_points_x[1]*1],
                         [-image_points_y[1]*world_points_x[1], -image_points_y[1]*world_points_y[1], -image_points_y[1]*world_points_z[1], -image_points_y[1]*1, image_points_x[1]*world_points_x[1], image_points_x[1]*world_points_y[1], image_points_x[1]*world_points_z[1], image_points_x[1]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[2], -1*world_points_y[2], -1*world_points_z[2], -1*1, image_points_y[2]*world_points_x[2], image_points_y[2]*world_points_y[2], image_points_y[2]*world_points_z[2], image_points_y[2]*1],
                         [1*world_points_x[2], 1*world_points_y[2], 1*world_points_z[2], 1*1, 0, 0, 0, 0, -image_points_x[2]*world_points_x[2], -image_points_x[2]*world_points_y[2], -image_points_x[2]*world_points_z[2], -image_points_x[2]*1],
                         [-image_points_y[2]*world_points_x[2], -image_points_y[2]*world_points_y[2], -image_points_y[2]*world_points_z[2], -image_points_y[2]*1, image_points_x[2]*world_points_x[2], image_points_x[2]*world_points_y[2], image_points_x[2]*world_points_z[2], image_points_x[2]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[3], -1*world_points_y[3], -1*world_points_z[3], -1*1, image_points_y[3]*world_points_x[3], image_points_y[3]*world_points_y[3], image_points_y[3]*world_points_z[3], image_points_y[3]*1],
                         [1*world_points_x[3], 1*world_points_y[3], 1*world_points_z[3], 1*1, 0, 0, 0, 0, -image_points_x[3]*world_points_x[3], -image_points_x[3]*world_points_y[3], -image_points_x[3]*world_points_z[3], -image_points_x[3]*1],
                         [-image_points_y[3]*world_points_x[3], -image_points_y[3]*world_points_y[3], -image_points_y[3]*world_points_z[3], -image_points_y[3]*1, image_points_x[3]*world_points_x[3], image_points_x[3]*world_points_y[3], image_points_x[3]*world_points_z[3], image_points_x[3]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[4], -1*world_points_y[4], -1*world_points_z[4], -1*1, image_points_y[4]*world_points_x[4], image_points_y[4]*world_points_y[4], image_points_y[4]*world_points_z[4], image_points_y[4]*1],
                         [1*world_points_x[4], 1*world_points_y[4], 1*world_points_z[4], 1*1, 0, 0, 0, 0, -image_points_x[4]*world_points_x[4], -image_points_x[4]*world_points_y[4], -image_points_x[4]*world_points_z[4], -image_points_x[4]*1],
                         [-image_points_y[4]*world_points_x[4], -image_points_y[4]*world_points_y[4], -image_points_y[4]*world_points_z[4], -image_points_y[4]*1, image_points_x[4]*world_points_x[4], image_points_x[4]*world_points_y[4], image_points_x[4]*world_points_z[4], image_points_x[4]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[5], -1*world_points_y[5], -1*world_points_z[5], -1*1, image_points_y[5]*world_points_x[5], image_points_y[5]*world_points_y[5], image_points_y[5]*world_points_z[5], image_points_y[5]*1],
                         [1*world_points_x[5], 1*world_points_y[5], 1*world_points_z[5], 1*1, 0, 0, 0, 0, -image_points_x[5]*world_points_x[5], -image_points_x[5]*world_points_y[5], -image_points_x[5]*world_points_z[5], -image_points_x[5]*1],
                         [-image_points_y[5]*world_points_x[5], -image_points_y[5]*world_points_y[5], -image_points_y[5]*world_points_z[5], -image_points_y[5]*1, image_points_x[5]*world_points_x[5], image_points_x[5]*world_points_y[5], image_points_x[5]*world_points_z[5], image_points_x[5]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[6], -1*world_points_y[6], -1*world_points_z[6], -1*1, image_points_y[6]*world_points_x[6], image_points_y[6]*world_points_y[6], image_points_y[6]*world_points_z[6], image_points_y[6]*1],
                         [1*world_points_x[6], 1*world_points_y[6], 1*world_points_z[6], 1*1, 0, 0, 0, 0, -image_points_x[6]*world_points_x[6], -image_points_x[6]*world_points_y[6], -image_points_x[6]*world_points_z[6], -image_points_x[6]*1],
                         [-image_points_y[6]*world_points_x[6], -image_points_y[6]*world_points_y[6], -image_points_y[6]*world_points_z[6], -image_points_y[6]*1, image_points_x[6]*world_points_x[6], image_points_x[6]*world_points_y[6], image_points_x[6]*world_points_z[6], image_points_x[6]*1, 0, 0, 0,0],

                         [0, 0, 0, 0, -1*world_points_x[7], -1*world_points_y[7], -1*world_points_z[7], -1*1, image_points_y[7]*world_points_x[7], image_points_y[7]*world_points_y[7], image_points_y[7]*world_points_z[7], image_points_y[7]*1],
                         [1*world_points_x[7], 1*world_points_y[7], 1*world_points_z[7], 1*1, 0, 0, 0, 0, -image_points_x[7]*world_points_x[7], -image_points_x[7]*world_points_y[7], -image_points_x[7]*world_points_z[7], -image_points_x[7]*1],
                         [-image_points_y[7]*world_points_x[7], -image_points_y[7]*world_points_y[7], -image_points_y[7]*world_points_z[7], -image_points_y[7]*1, image_points_x[7]*world_points_x[7], image_points_x[7]*world_points_y[7], image_points_x[7]*world_points_z[7], image_points_x[7]*1, 0, 0, 0,0]
                         ])


_, _, v = np.linalg.svd(P_Calc_Matrix)
P_Est = np.reshape(v[:,-1], (3,4))

print("The estimated Projection Matrix is: \n", P_Est)


##-----------------------Calculating C Matrix----------------------------##

_, _, v_c = np.linalg.svd(P_Est)
C_Est = np.reshape(v_c[:,-1], (4,1))
C_Est = C_Est/C_Est.item(3)

print("\n The estimated C Matrix is: \n", C_Est)

##-------------------------Calculating M Matrix--------------------------##
IC = np.array([[1, 0, 0, -C_Est.item(0)], [0, 1, 0, -C_Est.item(1)], [0, 0, 1, -C_Est.item(2)]])

print("\n IC is: \n", IC)

M = P_Est[:,:3]

print("\nThe Estimated M Matrix is:\n", M)


##-------------------------Calculating R Matrix--------------------------##
''' A1 A2 and A3 are the Rows of the M Matrix'''
A1 = M[0][:]
U1 = A1
E1 = U1/np.linalg.norm(U1)
print("\nE1 is:\n", E1)


A2 = M[1][:]
U2 = A2 - np.dot(U1,A2)
E2 = U2/np.linalg.norm(U2)
print("\nE2 is:\n", E2)


A3 = M[2][:]
U3 = A3 - np.dot(U1,A3) - np.dot(U2, A3)
E3 = U3/np.linalg.norm(U3)
print("\nE3 is:\n", E3)


R = np.array([E1, E2, E3]).T
print("\nThe Estimated R Matrix is:\n", R)


##------------------------Calculating K Matrix--------------------------##

K = M*np.linalg.inv(R)
print("\nThe Estimated Intrinsic Matrix K is:\n", K)





