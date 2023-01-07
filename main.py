import tkinter as tk
import seamcarver

# Create the main window
root = tk.Tk()
root.title('Seam Carving')

# Create an instance of the SeamCarver class
app = seamcarver(root)

# Run the main loop
root.mainloop()