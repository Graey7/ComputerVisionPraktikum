import numpy as np
from scipy.ndimage.filters import sobel
from PIL import Image

def calculate_energy(image):
    # Calculate the gradient magnitude using the Sobel operator
    gradient_magnitude = np.sqrt(sobel(image, axis=0)**2 + sobel(image, axis=1)**2)
    return gradient_magnitude

def calculate_seam(image, energy):
    # Create an empty seam list
    seam = []
    
    # Iterate through the rows of the image
    for i in range(image.shape[0]):
        # If this is the first row, add the indices of the lowest energy pixels to the seam
        if i == 0:
            seam.append(np.argmin(energy[i]))
        else:
            # Find the lowest energy pixel in the previous row
            prev_index = seam[i-1]
            # Check the pixels to the left, center, and right of the previous pixel
            # and choose the one with the lowest energy
            choices = [prev_index-1, prev_index, prev_index+1]
            energies = [energy[i, j] for j in choices]
            min_index = np.argmin(energies)
            print(min_index)
            seam.append(choices[min_index])

    
    return seam

def remove_seam(image, seam):
    # Create a copy of the image with an extra column of pixels
    new_image = np.zeros((image.shape[0], image.shape[1]+1, 3), dtype=np.uint8)
    new_image[:, 1:] = image
    
    # Iterate through the rows of the image
    for i in range(image.shape[0]):
        # Remove the pixel at the seam index
        new_image[i, :seam[i]] = image[i, :seam[i]]
        new_image[i, seam[i]+1:] = image[i, seam[i]:]
    
    return new_image

def seam_carving(image, num_seams):
    # Convert the image to a NumPy array
    image = np.array(image)
    
    # Calculate the energy of each pixel
    energy = calculate_energy(image)
    
    # Remove the specified number of seams from the image
    for i in range(num_seams):
        # Calculate the minimum energy seam
        seam = calculate_seam(image, energy)
        # Remove the seam from the image
        image = remove_seam(image, seam)
        # Recalculate the energy of the image
        energy = calculate_energy(image)
    
    # Convert the modified image back to a PIL image and return it
    return Image.fromarray(image)

# Load the image and display it
image = Image.open('image.jpg')
image.show()

# Run the seam carving algorithm and display the modified image
modified_image = seam_carving(image, 10)
modified_image.show()