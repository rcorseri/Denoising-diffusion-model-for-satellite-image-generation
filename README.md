# Denoising diffusion probabilistic model - Application to satellite and handwritten digits image generation

### Author: R. Corseri

## Overview 

This repository contains code for training and sampling denoising diffusion probabilistic model for satellite and handwritten digits image generation. The project aims at (1) testing the architecture, hyparameters of diffusion models to create realistic images based on a limited amount of training images (2) Investigating foundation models that have trained on large unlabeled satellite image datasets.  


## Features

- Implementation of a multihead CNN architecture tailored for 1D MT inversion.
- Integration of physics-based constraints into the CNN model to improve inversion results.
- Input data preprocessing and normalization routines.
- Training and evaluation scripts for the CNN model.
- Utilities for data loading, visualization, and analysis.

## Project Structure

- models/ : Contains the definition of the CNN architecture used for 1D MT inversion. The directory also stores the trained models. 
- src/ : Includes source code files for data preprocessing, model training, evaluation, and utilities.
- data/ : Placeholder directory for storing input data (e.g., apparent resistivity and phase data).
- tests/ : Directory for storing the field data, outputs of the tests and visualization
- report/ : contains the report

## Reproducibility

- To train the models, plot the loss function and visualize the output for an example: run main.py
- To test the trained models on real data: run tests/test_model_on_field_data.py

## Reference

All the codes used in the project have been modified from these two repositories:
- https://github.com/wgcban/ddpm-cd/tree/master#51-download-the-change-detection-datasets
- https://github.com/lucidrains/denoising-diffusion-pytorch

## Dependencies

    Python (>=3.6)
    PyTorch (>=1.9.0)
    NumPy
    Pandas
    Matplotlib

 
