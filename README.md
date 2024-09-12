# Dragonfly MORF Mouse Brain Registration

This repository contains the scripts and documentation for the pipeline used to register the 10x images, 30x images, and neuronal reconstructions for over 12 mouse brains to the Allen Common Coordinate Frameowrk (CCF).

## Step 1 (Atlas to 10x Registration)
Using the `argparse` library, we created a command-line interface in order to register the Allen Mouse Brain Atlas to our dataset of 10x mouse brain images. This program can be used by running:

`python3 step1_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.

## Step 2 (30x to 10x Registration)
Using the `argparse` library, we created another command-line interface in order to register our dataset of 30x mouse brain images to our dataset of 10x mouse brain images. This program can be used by running:

`python3 step2_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.

## Step 3 (Neuron Reconstruction Registration)
Using the `argparse` library, we created a command-line interface in order to register our dataset of neuron reconstructions (defined as a set of .swc files) to our dataset of 10x mouse brain images. This program can be used by running:

`python3 step3_atlas_registration.py`, followed by all the relevant positional and optional arguments from any command-line interface.
