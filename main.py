import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk

import seamcarver as sc

#Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Create the main window
root = ctk.CTk()

# Set the window size
root.geometry("515x190")
root.resizable(False, False)

#root.iconbitmap('icon.ico')
#root.iconbitmap(r'ComputerVisionPraktikum/icon.ico')

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

def open_model():
    filetypes = (
        ("Model files", "*.mdl"), 
        ("All files", "*.*")
        )
    modelfilename = fd.askopenfilename(
        title = "Open a Model",
        initialdir = "/",
        filetypes = filetypes
    )
    if(len(modelfilename)>27):
        modelinputdialog.configure(text = "Modelpath: " + modelfilename[:28] + "[...]")
    else:
        modelinputdialog.configure(text = "Modelpath: " + modelfilename)
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

def checkimage():
    print("Test")




checkimagebutton = ctk.CTkButton(modelframe, text = "Check Image", command = checkimage)
checkimagebutton.pack(padx = 2, pady = 2)
# Run the main loop
root.mainloop()
