MMR-GAN: README

Overview

MMR-GAN (Multi-phase Material Reconstruction GAN) is a generative adversarial network (GAN) designed to generate and reconstruct 2D and 3D representations of multi-phase materials. The GAN utilizes both 2D and 3D convolutional neural networks to learn the distribution of different phases in materials and then reconstruct them accurately, ensuring that the generated outputs closely mimic the real material structures.

The model's unique aspect is its ability to handle both 2D and 3D data, making it highly versatile for applications in material science where accurate 3D modeling of structures is critical.

Files Description
config.py: This file contains configuration settings using EasyDict, including paths for saving images, learning rates, batch sizes, input shapes, and other hyperparameters required for training and testing the models.

data_loader.py: This script is responsible for loading and preprocessing the 3D data. It handles the reading of material data, organizes it into the correct format, and prepares it for input into the GAN models.

models.py: Contains the implementation of the 2D and 3D generator and discriminator models. The models are built using TensorFlow's Keras API, with specific configurations tailored to handle 2D and 3D convolutions necessary for the material reconstruction tasks.

utils.py: This file provides utility functions that support the main processing tasks, such as generating noise for the GAN input, smoothing images, and calculating certain reconstruction metrics (like l2_rec1_3D).

test.py: Implements the main testing loop, where the trained 2D and 3D models are applied to input data to generate reconstructed material images. It includes functionality for loading model checkpoints, performing inference, and saving the reconstructed outputs.

main.py: The entry point for executing the main functions of the MMR-GAN, primarily invoking the testing procedure. It links the testing script to the appropriate main function and ensures that the necessary components are initialized correctly.

Customizing Configurations
All configuration options such as learning rate, batch size, and input shapes are managed through the config.py file. Adjust the values in this file to suit your specific needs.

Results
The output will include reconstructed 2D and 3D images saved as .txt files, which represent the multi-phase structures in a format ready for further analysis or visualization.

Contributing
If you wish to contribute to this project, please fork the repository and submit a pull request. Make sure to include detailed descriptions of your changes.