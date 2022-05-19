# Producing-realistic-climate-data-with-GANs

Associated repository to the article Producing realistic climate data (10.5281/zenodo.4436274)

## Installation

 - Clone the repository in your working folder
 - Create a virtual environment and install the `requirements.txt` file
   - `python -m venv env_folder`
   - `source env_folder/bin/activate`
   - `pip install -r requirements.txt`
- /!\ The code is in version tensorflow==1.14.0 and keras==2.2.4
- /!\ Cartopy package requires some preinstallation (see [docuementation](https://scitools.org.uk/cartopy/docs/latest/installing.html))
- Download the data ( a sample is available [here](https://www.kaggle.com/datasets/camilleb469/climate-data-3years-simulation)) and place it in the ./data/raw/ folder.
- Finally run `python training.py`

## Notebooks

In the notebooks the code used to create the figures in the article is available. 
These notebooks run on the reduced dataset, consequently the figures and the results will 
be less accurate due to the small sample size. 

However, the saved model of the GAN Generator used for the article can be found in the folder `model`.


## Interpolation example

Radial interpolation :

![Radial interpolation](./src/gifs/interp_radial.gif)

Linear interpolation in latent space :

![Linear interpolation in latent space.](./src/gifs/interp_linear.gif)

Linear interpolation in image space :

![Linear interpolation in image space.](./src/gifs/interp_xlin.gif)

Spherical interpolation :

![Spherical interpolation](./src/gifs/interp_spherical.gif)