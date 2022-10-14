# Genomic_diffusion
This repository contains the code of our article on quantitative genetics models.
In the gaussian_field_fourier you will find a function make_diffusion_from_matrix(), and a function generate_random_cov().

The function generate_random_cov() generates a random I as detailed in our paper.
The function make_diffusion_from_matrix() uses any interaction I matrix (representing a function of [0,1]^2) and generates a Gaussian process accordingly.
