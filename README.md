# Denoising diffusion probabilistic model - Application to satellite and handwritten digits image generation

### Author: R. Corseri

## Overview 

This repository contains code for training and sampling denoising diffusion probabilistic model for satellite and handwritten digits image generation. The project aims at (1) testing the architecture, hyparameters of diffusion models to create realistic images based on a limited amount of training images, and (2) Investigating the power of a foundation model that has been trained on large corpus of unlabeled satellite image datasets.  


## Project Structure

- model/ : Contains the definition of the CNN architecture used for 1D MT inversion. The directory also stores the trained models. 
- core/ : Includes source code files for data preprocessing, model training, evaluation, and utilities.
- datasets/ : Placeholder directory for storing training images.
- experiments/ : Placeholder directory for storing the results of the training and sampling experiments
- report/ : contains the report

## Reproducibility

- To train the models, plot the loss function and visualize the output for an example: run train_diffusion_model.ipynb
- To samople the pre-trained model, download the weights and biases  ( https://www.dropbox.com/scl/fo/eeeclganhghux3g657u6b/AOOeiz4h-Er9RAVD5a_t7GQ?rlkey=nnsfglyg6quuaydzm4ctnn0uu&e=2&dl=0 ) and then run sample_pre_trained_diffusion_model.ipynb

## References

The PyTorch implementation of "sr3-type" diffusion models was adapted from the codes copied from these two repositories:
- https://github.com/wgcban/ddpm-cd/tree/master#51-download-the-change-detection-datasets
- https://github.com/lucidrains/denoising-diffusion-pytorch

## Dependencies

    Python (>=3.6)
    PyTorch (>=1.9.0)
    NumPy
    Pandas
    Matplotlib

 
