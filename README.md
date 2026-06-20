# Human Recognition in Surveillance Settings

> **Bachelor's Final Project**\
> *Author:* Ruben Filipe Nascimento Abadesso\
> *Institution:* Universidade da Beira Interior (UBI)

This repository contains the code and documentation for the project on human recognition in surveillance environments, developed at Universidade da Beira Interior (UBI).

## Overview
Recognizing individuals in video surveillance is a significant challenge due to poor image quality, low resolution, varying lighting, and occlusions. This project focuses on an unsupervised Deep Learning approach to translate low-quality surveillance facial images into high-quality images to facilitate easier recognition. 

## Project Structure
The project explores three distinct deep learning approaches to map low-quality inputs to their high-quality ground truths. The experiments are documented in the following notebooks:

* **`1_Abordagem_U-Net.ipynb`**: Initial baseline approach using a standard U-Net Encoder-Decoder architecture.
* **`2_Abordagem_Pix2pix.ipynb`**: Advanced approach employing a Conditional Generative Adversarial Network (cGAN) based on the Pix2Pix architecture.
* **`3_Abordagem_Pix2pix_headpose.ipynb`**: The final and most effective model, combining the Pix2Pix architecture with mapped head pose data (Yaw, Pitch, Roll) for superior image translation.

## Documentation
For a comprehensive overview of the motivation, synchronized dataset collection, methodology, and visual comparisons of the results, please consult the full report:
* **`Projeto_Final_Licenciatura_Ruben_Abadesso.pdf`**

## Technologies & Hardware
* **Frameworks:** Python, TensorFlow, and Keras.
* **Development Environments:** PyCharm & DataSpell.
* **Training Hardware:** AMD Ryzen 9 5900HX, 16GB RAM, NVIDIA GeForce RTX 3060.
