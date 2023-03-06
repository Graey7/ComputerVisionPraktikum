import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms


def compress_img(image_name, quality=100, to_jpg=True):
    # load the image to memory
    img = Image.open(image_name)
    # print the original image shape
    print("Image shape:", img.size)
    # get the original image size in bytes
    image_size = os.path.getsize(image_name)

    xA=img.size[0]
    yA=img.size[1]
    
    # start compressing
    # use y-axis as min bound = 224 pixels
    if(xA > yA):
        # calc to get square (224x224)
        print("y kleiner")
        xRatio = int(yA/224)
        newX = int(xA/xRatio)
        left = int((newX-224)/2)
        right = 224+left

        # compress to y bound
        img = img.resize((newX, 224), Image.Resampling.LANCZOS)
        # Cropped image of above dimension
        img = img.crop((left, 0, right, 224)) 
        
    # use x-axix as min bound 
    if(xA <= yA):
        yRatio = int(xA/224)
        newY = int(yA/yRatio)
        top = int((newY-224)/2)
        bottom = 224+top

        # compress to x bound
        img = img.resize((224, newY), Image.Resampling.LANCZOS)
        # Cropped image of above dimension
        img = img.crop((0, top, 224, bottom))
        
    print("New Image shape:", img.size)
    filename, ext = os.path.splitext(image_name)
    # make new filename appending _compressed to the original file name
    if to_jpg:
        # change the extension to JPEG
        new_filename = f"{filename}_compressed.jpg"
    else:
        # retain the same extension of the original image
        new_filename = f"{filename}_compressed{ext}"
    return img
    #try:
        # save the image with the corresponding quality and optimize set to True

        #img.save(new_filename, quality=quality, optimize=True)
    #except OSError:
        # convert the image to RGB mode first
        #img = img.convert("RGB")
        # save the image with the corresponding quality and optimize set to True
        #img.save(new_filename, quality=quality, optimize=True)
    #print("[+] New file saved:", new_filename)

    


img_mod = compress_img("test_img.jpg", 100, True)

# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])
  
# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = transform(img_mod)
  
# print the converted Torch tensor shape
print(img_tensor.shape)
