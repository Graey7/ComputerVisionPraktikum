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
root.geometry("600x400")
root.resizable(False, False)

root.iconbitmap("icon.ico")

root.title('Seam Carving')

scframe = ctk.CTkFrame(root, 
                       width = 280, 
                       height = 180, 
                       border_width = 2)
scframe.grid()

inputlabel = ctk.CTkLabel(scframe, text = "Enter the path to the image:")
inputlabel.pack(padx = 2, pady = 2)

inputdialog = ctk.CTkLabel(scframe, text = "Filepath: ", 
                           width = 250, 
                           height = 10,
                           anchor = "w",
                           compound = "left")
inputdialog.pack(padx = 2, pady = 2)


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



open_button = ctk.CTkButton(
    scframe,
    text = "Open file",
    command = open_file
)
open_button.pack(expand=True)

carvelabel = ctk.CTkLabel(scframe, text = "Input number of seams to be carved:")
carvelabel.pack(padx = 2, pady = 2)


def carve():
    global entry
    string = entry.get()
    print("Carving...")
    sc.main(inputdialog.cget("text"), int(string))

entry = ctk.CTkEntry(scframe)
entry.focus_set()
entry.pack(padx = 2, pady = 2)


carvebutton = ctk.CTkButton(scframe, text = "Carve", command = carve)
carvebutton.pack(padx = 2, pady = 2)


# Run the main loop
root.mainloop()