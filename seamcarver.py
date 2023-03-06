from datetime import datetime
import numpy as np
import cv2
from PIL import Image

def calculate_energy(img):
    # Calculate the energy of each pixel using the Sobel operator
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_x = cv2.Sobel(img, -1, dx=1, dy=0)
    f_y = cv2.Sobel(img, -1, dx=0, dy=1)
    energy = abs(f_x) + abs(f_y)
    return energy

def calculate_seam(energy):
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

def remove_seam(image, seam):
    # Create a copy of the image with one less column
    new_image = np.zeros((image.shape[0], image.shape[1]-1, image.shape[2]))
    
    # Iterate through the rows of the image
    for j in range(image.shape[2]):
        for i in range(image.shape[0]):
            # Remove the pixel at the seam index
            if(seam[i] ==  image.shape[1]):
                new_image[i, :, j] = image[i, :(seam[i]), j]
            elif(seam[i] == 0):
                new_image[i, :, j] = image[i, (seam[i] + 1):, j]
            else:
                new_image[i, :seam[i], j] = image[i, :(seam[i]), j]
                new_image[i, seam[i]:, j] = image[i, (seam[i]+1):, j]

    
    return new_image

def seam_carving(image, num_seams):
    # Convert the image to a NumPy array
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Calculate the energy of each pixel
    energy = calculate_energy(image)
    
    # Remove the specified number of seams from the image
    for i in range(num_seams):
        # Calculate the minimum energy seam
        init = datetime.now()
        seam = calculate_seam(energy)
        time = (datetime.now() - init)
        print("Calculating seam took: " + str(time) + " seconds")
        # Remove the seam from the image
        image = remove_seam(image, seam)
        # Recalculate the energy of the image
        energy = calculate_energy(image)
    
    # Convert the modified image back to a PIL image and return it
    im = Image.fromarray((image).astype(np.uint8))
    return im

def main(file, carve_num_seams):
    image = cv2.imread(file)
    modified_image = seam_carving(image, carve_num_seams)
    modified_image.save(file + "_carved.png")
    modified_image.show()
    
# Load the image and display it
#image = cv2.imread('test.png')

# Run the seam carving algorithm and display the modified image
#modified_image = seam_carving(image, 100)
#modified_image.show()
#modified_image.save("Seam_Carved_image.png")