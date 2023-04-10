# Seam Carving GUI

Welcome to the Seam Carving GUI, a graphical user interface for the Seam Carving algorithm.

## Description

The Seam Carving algorithm is a content-aware image resizing technique that reduces the size of an image by selectively removing pixels that are less important. The algorithm identifies and removes pixels along a path of least energy in the image, which can be vertical or horizontal.

The Seam Carving GUI provides an easy-to-use interface for applying the Seam Carving algorithm to images. Users can select an input image and specify the desired output size by the number of seams. The algorithm then applies Seam Carving to the image and displays the resulting resized image. Additionally, the GUI includes a feature to detect whether an image has already been resized using Seam Carving. For this purpose, you use one of the five provided models. This has been trained on a data set to detect seam carving.

## Features

The Seam Carving GUI includes the following features:

- Open and save image files in JPEG, PNG, and other common image formats
- Specify the desired number of carved seams for the resized image
- Save the resized image to a new file
- Detect whether an image has already been resized using Seam Carving
- Choose a trained model for detection

## Getting Started

To use the Seam Carving GUI, follow these steps:

1. Install the required dependencies. The GUI requires Python 3 and the packages from the requirements.txt file.
2. Clone or download the Seam Carving GUI repository to your local machine.
3. Open a terminal or command prompt and navigate to the repository directory.
4. Run the `gui_main.py` script with Python to start the GUI.

## Usage

Once the Seam Carving GUI is open, follow these steps to resize an image:

1. Click the "Open Image" button to select an input image file.
2. Choose the desired number of seams to be carved for the resized image.
3. Click the "Carve" button to apply Seam Carving to the image and display the resized image.

To detect whether an image has already been resized using Seam Carving, follow these steps:

1. Click the "Open Model" button and select the desired trained model file.
2. Open the image you want to check.
3. Click the "Check Image" button.
4. The GUI will display a message indicating whether the image has already been resized using Seam Carving or not.

## License

The Seam Carving GUI is released for Prof. Jiang's Computer Vision lab at WWU.
