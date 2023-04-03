import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from xception import xception

import seamcarver as sc

# NVIDIA GPU?
nvidia = torch.cuda.is_available()

#Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Create the main window
root = ctk.CTk()

# Set the window size
root.geometry("515x190")
#root.geometry("525x400")
root.resizable(False, False)

root.iconbitmap("@/home/jerrit/Programs/icon.xbm")

root.title('Seam Carving')

#Frame for the seam carver---------------------------------------------------------------
scframe = ctk.CTkFrame(root, 
                       width = 280, 
                       height = 400, 
                       border_width = 2)
scframe.grid()
scframe.grid_propagate(False)

inputlabel = ctk.CTkLabel(scframe, text = "Enter the path to the image:")
inputlabel.pack(padx = 2, pady = 2)

inputdialog = ctk.CTkLabel(scframe, text = "Filepath: ", 
                           width = 250, 
                           height = 10,
                           anchor = "w",
                           compound = "left")
inputdialog.pack(padx = 2, pady = 2)

temp = ctk.CTkLabel(scframe, text = "Empty")

def open_file():
    filetypes = (
        ("PNG files", "*.png"), 
        ("All files", "*.*")
        )
    filename = fd.askopenfilename(
        title = "Open a PNG file",
        initialdir = "/",
        filetypes = filetypes
    )
    if(len(filename)>27):
        inputdialog.configure(text = "Filepath: " + filename[:28] + "[...]")
    else:
        inputdialog.configure(text = "Filepath: " + filename)
    temp.configure(text = filename)
    return filename


open_button = ctk.CTkButton(
    scframe,
    text = "Open Image",
    command = open_file
)
open_button.pack(expand=True)

carvelabel = ctk.CTkLabel(scframe, text = "Input number of seams to be carved:")
carvelabel.pack(padx = 2, pady = 2)


def carve():
    global entry
    string = entry.get()
    print("Carving...")
    if(temp.cget("text") == "Empty"):
        showinfo(title = "Error", message = "Please select an image to carve.")
    else:
        sc.main(temp.cget("text"), int(string))

entry = ctk.CTkEntry(scframe)
entry.focus_set()
entry.pack(padx = 2, pady = 2)


carvebutton = ctk.CTkButton(scframe, text = "Carve", command = carve)
carvebutton.pack(padx = 2, pady = 2)

#Frame for the model---------------------------------------------------------------
modelframe = ctk.CTkFrame(root, 
                       width = 280, 
                       height = 400, 
                       border_width = 2)
modelframe.grid(column = 1, row = 0, padx = 5)
modelframe.grid_propagate(False)

modelinputlabel = ctk.CTkLabel(modelframe, text = "Enter the path to the image:")
modelinputlabel.pack(padx = 2, pady = 2)

modelinputdialog = ctk.CTkLabel(modelframe, text = "Filepath: ", 
                           width = 250, 
                           height = 10,
                           anchor = "w",
                           compound = "left")
modelinputdialog.pack(padx = 2, pady = 2)

modeltemp = ctk.CTkLabel(modelframe, text = "Empty")



#textbox = ctk.CTkTextbox(root, width=260)
#textbox.grid(row=400, column=0)

#textbox.configure(state="disabled")  # configure textbox to be read-only
#textbox.bind()

def open_model():
    modelfilename = fd.askopenfilename(
        title = "Open a Model",
        initialdir = "/"
    )
    if(len(modelfilename)>27):
        modelinputdialog.configure(text = "Model path: " + modelfilename[:28] + "[...]")
    else:
        modelinputdialog.configure(text = "Image path: " + modelfilename)
    modeltemp.configure(text = modelfilename)
    return modelfilename


open_model_button = ctk.CTkButton(
    modelframe,
    text = "Open model",
    command = open_model
)
open_model_button.pack(expand=True)

modellabel = ctk.CTkLabel(modelframe, text = "Enter the path to the image:")
modellabel.pack(padx = 2, pady = 2)

modelimageinputdialog = ctk.CTkLabel(modelframe, text = "Filepath: ", 
                           width = 250, 
                           height = 10,
                           anchor = "w",
                           compound = "left")
modelimageinputdialog.pack(padx = 2, pady = 2)

modelimagetemp = ctk.CTkLabel(modelframe, text = "Empty")

def open_modelimage():
    filetypes = (
        ("Image files", "*.png"), 
        ("All files", "*.*")
        )
    modelimagefilename = fd.askopenfilename(
        title = "Open an Image",
        initialdir = "/",
        filetypes = filetypes
    )
    if(len(modelimagefilename)>27):
        modelimageinputdialog.configure(text = "Modelpath: " + modelimagefilename[:28] + "[...]")
    else:
        modelimageinputdialog.configure(text = "Modelpath: " + modelimagefilename)
    modelimagetemp.configure(text = modelimagefilename)
    return modelimagefilename

open_modelimage_button = ctk.CTkButton(
    modelframe,
    text = "Open image",
    command = open_modelimage
)
open_modelimage_button.pack(expand=True)



loader = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0) 
    image = image[:3]
    if(nvidia):
        return image.cuda()
    return image

def checkimage():
    # Define the classes
    classes = ['not_carved', 'carved']
    model = xception(output='softmax', pretrained=False)
    temp = modeltemp.cget("text")
    if(nvidia):
        model.load_state_dict(torch.load(temp))
        model.eval()
        model = model.to("cuda")
    else:
        model.load_state_dict(torch.load(temp, map_location=torch.device('cpu')))
        model.eval()
        model = model.to("cpu")
        
    image = image_loader(modelimagetemp.cget("text"))
    with torch.no_grad():
        logits = model(image)
    ps = torch.exp(logits)
    _, predTest = torch.max(ps, 1)

    # Get the predicted class
    predicted_class = classes[predTest.item()]

    # Print the predicted class
    print('Predicted class:', predicted_class)
    if predicted_class == 'carved':
        print('Image has been seam-carved')
        tk.messagebox.showinfo("showinfo", "Image has been seam-carved")
    else:
        print('Image has not been seam-carved')
        tk.messagebox.showinfo("showinfo", "Image has not been seam-carved")
    


checkimagebutton = ctk.CTkButton(modelframe, text = "Check Image", command = checkimage)
checkimagebutton.pack(padx = 2, pady = 2)
# Run the main loop
root.mainloop()