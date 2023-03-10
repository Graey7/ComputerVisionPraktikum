import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms


def compress_img(image_name):
    img = Image.open(f"./input/{image_name}")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    #filename, ext = os.path.splitext(image_name)
    new_filename = f"{filename}_compressed.jpg"
    img.save(f"./output/{new_filename}")
    return img

# loop through folder
folder = "input"
directory = os.fsencode(folder)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img_mod = compress_img(filename)

    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
  
    # Convert the PIL image to Torch tensor
    img_tensor = transform(img_mod)

    # print the converted Torch tensor shape
    #print(img_tensor.shape,"\n")

    filename, ext = os.path.splitext(filename)
    torch.save(img_tensor, f'./output/{filename}.pt')
