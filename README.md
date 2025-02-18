# Seam Carving GUI

Welcome to the Seam Carving GUI, a graphical user interface for the Seam Carving algorithm.

## Description

The Seam Carving algorithm is a content-aware image resizing technique that reduces the size of an image by selectively removing pixels that are less important. The algorithm identifies and removes pixels along a path of least energy in the image.

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

## Code

### resize.py

Provides basic functions for cropping the training images.

* loops through folder, crops images and saves then as an pytorch tensor
* compress_img: save picutre as separated comperessed file in 224x224 format

### seamcarver.py

Implements the Seam Carving functionality.

* calculate_energy: Calculates the energy of each pixel using the Sobel operator
* calculate_seam: Calculates the seam with minimal energy
* calculate_seam2: Test function for alternative calculation methods
* remove_seam: Creates a copy of the image with the calculated seam removed
* seam_carving: Implements the mentioned functions to remove a specific number of seams

### data.py

Provides functions to create suitable a pytorch dataset.

* extract_data: Extracts data from folders and labels it
* make_k_folds: Splits an array in k folds
* DatasetSeamCarved: Pytorch dataset class

### utils.py

Methods for metric plotting.

* make_plots: plots train and validation metrics for each fold
* make_bars: plots test metrics

### xception.py 

https://github.com/tstandley/Xception-PyTorch modified to include softmax.

### train.py

All functions used in the training and testing process of the model.

* make_train_step: Performs one training step and calculates metrics
* make_valid_step: Performs one validation step and calculates metrics
* train: Calls make_train_step, make_valid_step to train the model and logs the metrics
* make_test_step: Performs one test step and calculates metrics
* test: Calls make_test_step to test the model and logs the metrics

### main.py

Control and set training parameters for the training.
Prepare data for training, start training and save the model and plots for the metrics.

## License

The Seam Carving GUI is released for Prof. Jiang's Computer Vision lab at WWU.
