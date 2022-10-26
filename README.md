# Genomic_diffusion
This repository contains the code of our article on quantitative genetics models.
In the file concatenated_bridges.py you will find a function generate_genomic_diffusion implementing the first method described in our paper.


In the file gaussian_field_fourier.py you will find a function make_diffusion_from_matrix(), and a function generate_random_cov().

The function generate_random_cov() generates a random I as detailed in our paper, with a discretization timestep of 1/1001.
The function make_diffusion_from_matrix() uses any interaction 1001x1001 matrix, representing a function of [0,1]^2, and generates a Gaussian process accordingly.
