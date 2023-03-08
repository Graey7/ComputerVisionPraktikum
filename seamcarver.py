from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import os

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

def calculate_seam2(energy):
    # initialize backtrack and M matrices
    backtrack = np.zeros_like(energy, dtype=np.int64)
    M = np.zeros_like(energy)

    # set first row of M to be the same as the energy matrix
    M[0] = energy[0]

    # iterate through the rows of M and backtrack matrices
    for i in range(1, energy.shape[0]):
        # calculate the cumulative energy using the minimum of the previous row's cumulative energy
        # values and add the current energy value at (i,j)
        M[i, 0] = energy[i, 0] + np.min(M[i-1, 0:2])
        backtrack[i-1, 0] = np.argmin(M[i-1, 0:2])

        M[i, 1:-1] = energy[i, 1:-1] + np.minimum.reduce([M[i-1, :-2], M[i-1, 1:-1], M[i-1, 2:]])
        backtrack[i-1, 1:-1] = np.argmin(np.vstack([M[i-1, :-2], M[i-1, 1:-1], M[i-1, 2:]]), axis=0)

        M[i, -1] = energy[i, -1] + np.min(M[i-1, -2:])
        backtrack[i-1, -1] = np.argmin(M[i-1, -2:]) + energy.shape[1] - 2

    # backtrack to find the seam
    seam = np.zeros(energy.shape[0], dtype=np.int64)
    seam[-1] = np.argmin(M[-1])

    for i in range(energy.shape[0]-2, -1, -1):
        seam[i] = backtrack[i, seam[i+1]]

    return seam

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
    init = datetime.now()
    for i in range(num_seams):
        # Calculate the minimum energy seam
        #TODO calculateseam2???
        seam = calculate_seam(energy)

        # Remove the seam from the image
        image = remove_seam(image, seam)
        # Recalculate the energy of the image
        energy = calculate_energy(image)
    time = (datetime.now() - init)
    print("Calculating seam took: " + str(time) + " seconds")
    # Convert the modified image back to a PIL image and return it
    im = Image.fromarray((image).astype(np.uint8))
    return im

def main(file, carve_num_seams):
    image = cv2.imread(file)
    modified_image = seam_carving(image, carve_num_seams)
    modified_image.save(file + "_carved.png")
    modified_image.show()
    
#carve everything
def carve_all(percent):
	# make new directory
    newpath = 'carved_' + str(percent) #r'C:\Program Files\arbitrary' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    		
    directory = os.fsencode('uncarved_dataset')
    
    # for all images
    for file in os.listdir(directory):
        image_name = os.fsdecode(file)
        full_filename = 'uncarved_dataset/' + image_name
        number_seams = calculate_number_of_seams(percent, full_filename)
        #carve image
        image = cv2.imread(full_filename)
        mod_image = seam_carving(image, number_seams)
        #mod_image = resize_image(mod_image)
        #save new image
        #TODO saving in the right directory
        mod_image.save(newpath + '/carved_'+ str(percent) + '_' + image_name)


def calculate_number_of_seams(percent, image):
    im = Image.open(image)
    width, height = im.size
    result = (percent/100) * width
    result = round(result)
    return result

def testrun():
	# Load the image and display it
	image = cv2.imread('test.png')
	# Run the seam carving algorithm and display the modified image
	modified_image = seam_carving(image, 100)
	modified_image.show()
	modified_image.save("Seam_Carved_image.png")

def carve_all_10times():
	#carve all pictures with 3% 6% 9% 12% 15% 18% 21% 30% 40% 50%
	carve_all(3)
	carve_all(6)
	carve_all(9)
	carve_all(12)
	carve_all(15)
	carve_all(18)
	carve_all(21)
	carve_all(30)
	carve_all(40)
	carve_all(50)


#testrun()


carve_all_10times()

