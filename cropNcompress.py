import numpy as np
import os
import h5py
import torch
from PIL import Image
import torchvision.transforms as transforms


def compress_img(image_name):
    img = Image.open(f"./input/{image_name}")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    #filename, ext = os.path.splitext(image_name)
    #new_filename = f"{filename}_compressed.jpg"
    #img.save(f"./output/{new_filename}")
    return img

def compress_img_old(image_name):
    # load the image to memory
    img = Image.open(f"./input/{image_name}")
    # print the original image shape
    print("Old Image shape:", img.size)
    # get the original image size
    image_size = os.path.getsize(f"./input/{image_name}")
    xA=img.size[0]
    yA=img.size[1]
    
    # start compressing
    # use y-axis as min bound = 224 pixels
    if(xA > yA):
        # calc to get square (224x224)
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
    # retain the same extension of the original image
    new_filename = f"{filename}_compressed.jpg"
    #img.save(f"./output/{new_filename}")
    return img


# loop through folder
folder = "input"
directory = os.fsencode(folder)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img_mod = compress_img(filename)

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
  
    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(img_mod)

    # print the converted Torch tensor shape
    print(img_tensor.shape,"\n")

    filename, ext = os.path.splitext(filename)
    torch.save(img_tensor, f'./output/{filename}.pt')
