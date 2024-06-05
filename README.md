# Denoising diffusion probabilistic model - Application to satellite and handwritten digits image generation

### Author: R. Corseri

## Overview 
This repository contains code for training and sampling denoising diffusion probabilistic models (DDPM)  for generating satellite and handwritten digit images. The project has two main objectives:

- Testing the architecture, noise schedule, and hyperparameters of diffusion models to create realistic images based on a limited amount of training images.
- Investigating the power of a pre-trained "foundation" model that was trained on a large corpus of unlabeled satellite image datasets.

The experiments demonstrate the potential of DDPMs in generating high-quality images and explore their practical applications, including change detection in satellite imagery. Despite computational challenges, the use of pre-trained models highlights the accessibility and effectiveness of advanced generative capabilities for broader scientific and remote sensing applications.

## Project Structure

- config/ : Contains the *.json files that defines diffusion model architecture, beta schedule, path to training data and all relevant training parameters
- datasets/ : Placeholder directory for storing training images.
- experiments/ : Placeholder directory for storing the trained models, the results of the training and sampling experiments
- report/ : contains the report
- model/ : Contains the pytorch implementation of the "sr3-type" diffusion model (See reference repository below). 
- data/ :  Contains utilities and image pre-processing routines

## Reproducibility

- To train the models, plot the loss function and visualize the output for an example: run train_diffusion_model.ipynb
- To samople the pre-trained model, download the weights and biases  ( https://www.dropbox.com/scl/fo/eeeclganhghux3g657u6b/AOOeiz4h-Er9RAVD5a_t7GQ?rlkey=nnsfglyg6quuaydzm4ctnn0uu&e=2&dl=0 ) and then run sample_pre_trained_diffusion_model.ipynb

## References

The PyTorch implementation of "sr3-type" diffusion models was adapted from these two repositories:
- https://github.com/wgcban/ddpm-cd/tree/master#51-download-the-change-detection-datasets
- https://github.com/lucidrains/denoising-diffusion-pytorch

## Dependencies

    Python (>=3.6)
    PyTorch (>=1.9.0)
    NumPy
    Pandas
    Matplotlib

 
