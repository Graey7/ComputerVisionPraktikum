import cv2
import numpy as np

def energy_function(img):
    f_x = cv2.Sobel(img, -1, dx=1, dy=0)
    f_y = cv2.Sobel(img, -1, dx=0, dy=1)
    energy = abs(f_x) + abs(f_y)
    return energy

def find_seam(energy):
    backtrack = np.empty(energy.shape, dtype=np.int64)
    
    M = np.empty(energy.shape)
    M[0] = energy[0]

    # iterate through row
    for i in range(1, energy.shape[0]):
        # iterate through columns
        for j in range(energy.shape[1]):
    
            # prevent index error -1
            if j == 0:
                M[i][j] = energy[i][j] + min(M[i-1][j], M[i-1][j+1])
                backtrack[i-1][j] = np.argmin(M[i-1, j:j+2]) + j
            # prevent index error 
            elif j == energy.shape[1]-1:
                M[i][j] = energy[i][j] + min(M[i-1][j-1], M[i-1][j])
                backtrack[i-1][j] = np.argmin(M[i-1, j-1:j+1]) + j - 1
            else:
                M[i][j] = energy[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])
                backtrack[i-1][j] = np.argmin(M[i-1, j-1:j+2]) + j - 1

    # backtrack for seam 
    seam = np.empty(energy.shape[0], dtype=np.int64)
    j = np.argmin(M[-1])
    seam[0] = j
    for i in range(1,energy.shape[0]):    
        j = backtrack[energy.shape[0]-i-1,j]
        seam[i] = j

    return seam[::-1]


img = cv2.imread('img.jpg',)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#test = np.array([[1,2,3,0],[4,1,2,1],[3,0,1,6],[2,0,1,4]])
#test = np.array([[0,1,3],[2,1,0],[4,1,2],[1,7,0]])



#cv2.imshow('test window', energy_function(img))
# add wait key. window waits until user presses a key
#cv2.waitKey(0)
# and finally destroy/close all open windows
#cv2.destroyAllWindows()

print(find_seam(energy_function(img)))