import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk

import seamcarver as sc

# Create the main window
root = tk.Tk()

# Set the window size
root.geometry("600x400")
root.resizable(False, False)

photo = tk.PhotoImage(file = "icon.png")

root.wm_iconphoto = (photo)

root.title('Seam Carving')

scframe = tk.Frame(root)
scframe.pack()

inputlabel = tk.Label(scframe, text = "Enter the path to the image:")
inputlabel.pack()

inputdialog = tk.Label(scframe, text = "Filepath:")
inputdialog.pack()

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
    inputdialog["text"] = filename


open_button = ttk.Button(
    scframe,
    text = "Open file",
    command = open_file
)
open_button.pack(expand=True)

carvelabel = tk.Label(scframe, text = "Input number of pixels to be carved:")
carvelabel.pack()


def carve():
    global entry
    string = entry.get()
    print("Carving...")
    sc.main(inputdialog.cget("text"), int(string))

entry = tk.Entry(scframe)
entry.focus_set()
entry.pack()


carvebutton = tk.Button(scframe, text = "Carve", command = carve)
carvebutton.pack()


# Run the main loop
root.mainloop()