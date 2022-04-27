# Deep Learning Project 1

First deep learning project.

## Setup

### Requirements before installation

* CUDA 11.3

### Installation

* Clone the repository
* In a terminal, navigate to the root of the project and run `pip install -r requirements.txt`
* Run the file `config_verification.py` to perform a basic check of your setup
* Create a `data` folder and download the data.
* Adjust `parameters.json` to what suits you.
* Run `test_V1.py` to train the network. 

## Content of the repository

### `src` folder

* `dlc_practical_prologue.py` - Contains helper functions provided
* `config_verification.py` - Contains basic checks of your setup
* `utils.py` - Contains some helper functions.
* `Noise2noise.py` - Contains the Noise2noise torch.nn.Module model. 
* `parameters.json` - Contains the parameters for training and storing the results. 

### `outputs` folder
* `logs` - Contains the training logs as a `torch.tensor`, the title contains details about the training's hyperparameters. 
* `trained_model` - Contains all the trained models with the weights and architecture. 

