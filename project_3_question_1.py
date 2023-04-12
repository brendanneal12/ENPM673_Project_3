#Brendan Neal
#Directory ID: bneal12

#Project 3 Question 1 Code

##-----------------------------Importing Libraries---------------------------##
import numpy as np


##-------------------Setting Up Arrays from Project Document-----------------##

u = np.array([757, 758, 758, 759, 1190, 329, 1204, 340]) #Image X Points
v = np.array([213, 415, 686, 966, 172, 1041, 850, 159]) #Image Y Points
X = np.array([0, 0, 0, 0, 7, 0, 7, 0]) #World X Points
Y = np.array([0, 3, 7, 11, 1, 11, 9, 1]) #World Y Points
Z = np.array([0, 0, 0, 0, 0, 7, 0, 7]) #World Z Points

point_index_array = np.array([0,1,2,3,4,5,6,7])

##-------------------------Solving for The P Matrix--------------------------##
#uprime = x, 
#vprime = y
#wprime = 1
#X = [worldx, worldy, worldz, 1]

''' This method was adapted from Arunava's Office Hours'''
P_Calc_Matrix =np.array([[X[0], Y[0], Z[0], 1, 0, 0, 0, 0, -u[0]*X[0], -u[0]*Y[0], -u[0]*Z[0], -u[0]],
                         [0, 0, 0, 0, X[0], Y[0], Z[0], 1, -v[0]*X[0], -v[0]*Y[0], -v[0]*Z[0], -v[0]],

                         [X[1], Y[1], Z[1], 1, 0, 0, 0, 0, -u[1]*X[1], -u[1]*Y[1], -u[1]*Z[1], -u[1]],
                         [0, 0, 0, 0, X[1], Y[1], Z[1], 1, -v[1]*X[1], -v[1]*Y[1], -v[1]*Z[1], -v[1]],

                         [X[2], Y[2], Z[2], 1, 0, 0, 0, 0, -u[2]*X[2], -u[2]*Y[2], -u[2]*Z[2], -u[2]],
                         [0, 0, 0, 0, X[2], Y[2], Z[2], 1, -v[2]*X[2], -v[2]*Y[2], -v[2]*Z[2], -v[2]],

                         [X[3], Y[3], Z[3], 1, 0, 0, 0, 0, -u[3]*X[3], -u[3]*Y[3], -u[3]*Z[3], -u[3]],
                         [0, 0, 0, 0, X[3], Y[3], Z[3], 1, -v[3]*X[3], -v[3]*Y[3], -v[3]*Z[3], -v[3]],

                         [X[4], Y[4], Z[4], 1, 0, 0, 0, 0, -u[4]*X[4], -u[4]*Y[4], -u[4]*Z[4], -u[4]],
                         [0, 0, 0, 0, X[4], Y[4], Z[4], 1, -v[4]*X[4], -v[4]*Y[4], -v[4]*Z[4], -v[4]],

                         [X[5], Y[5], Z[5], 1, 0, 0, 0, 0, -u[5]*X[5], -u[5]*Y[5], -u[5]*Z[5], -u[5]],
                         [0, 0, 0, 0, X[5], Y[5], Z[5], 1, -v[5]*X[5], -v[5]*Y[5], -v[5]*Z[5], -v[5]],

                         [X[6], Y[6], Z[6], 1, 0, 0, 0, 0, -u[6]*X[6], -u[6]*Y[6], -u[6]*Z[6], -u[6]],
                         [0, 0, 0, 0, X[6], Y[6], Z[6], 1, -v[6]*X[6], -v[6]*Y[6], -v[6]*Z[6], -v[6]],

                         [X[7], Y[7], Z[7], 1, 0, 0, 0, 0, -u[7]*X[7], -u[7]*Y[7], -u[7]*Z[7], -u[7]],
                         [0, 0, 0, 0, X[7], Y[7], Z[7], 1, -v[7]*X[7], -v[7]*Y[7], -v[7]*Z[7], -v[7]]
                         ])



## Solve for P using SVD
_, _, v_p = np.linalg.svd(P_Calc_Matrix)
P_Est = np.reshape(v_p[:,-1], (3,4))
P_Est = P_Est/P_Est.item(11)

print("The estimated Projection Matrix is: \n", P_Est)


##-----------------------Calculating C Matrix----------------------------##


#Solve for C using SVD
_, _, v_c = np.linalg.svd(P_Est)
C_Est = np.reshape(v_c[:,-1], (4,1))
C_Est = C_Est/C_Est.item(3)

print("\n The estimated C Matrix is: \n", C_Est)



##-------------------------Calculating M Matrix--------------------------##

#M matrix is the left 3x3 of P
M = P_Est[:,:3]

print("\nThe Estimated M Matrix is:\n", M)


##-------------------------Calculating R Matrix--------------------------##

#Gram-Schmidt Process Performed Manually
A1 = M[0]
A2 = M[1]
A3 = M[2]

U1 = A1
E1 = U1 / np.linalg.norm(U1)

print("\n E1 is: \n", E1)

U2 = A2 - (np.dot(U1, A2)/np.dot(U1, U1)) * U1
E2 = U2 / np.linalg.norm(U2)

print("\n E2 is: \n", E2)

U3 = A3 - (np.dot(U1, A3)/np.dot(U1, U1)) * U1 - (np.dot(U2, A3)/np.dot(U2, U2)) * U2
E3 = U3/ np.linalg.norm(U3)

print("\n E3 is: \n", E3)

R = np.array([E1, E2, E3])

print("\nThe Estimated Rotation Matrix R is:\n", R)



##------------------------Calculating K Matrix--------------------------##

K = (M @ np.linalg.inv(R)).T
K = K/K.item(8)
print("\nThe Estimated Intrinsic Matrix K is:\n", K)

##--------------Calculating Reprojection Error for Each Point-----------##

all_reproj_errors = []
for i in point_index_array:
    projected_point = np.dot(P_Est, np.array([[X[i]],[Y[i]], [Z[i]], [1]]))
    projected_point = projected_point[:2]/projected_point[2]
    print(projected_point)



    point_diff = projected_point - np.array([[u[i]] , [v[i]]])

    reprojection_error = np.sqrt(np.sum(point_diff)**2)

    print("\n The reprojection error for world points", X[i], Y[i], Z[i], "and image points", u[i], v[i], "is:", reprojection_error)

    all_reproj_errors.append(reprojection_error)

avg_reproj_error = np.mean(all_reproj_errors)

print("\n The mean reprojection error for all the points is:", avg_reproj_error)




