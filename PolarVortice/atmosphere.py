# -*- coding: utf-8 -*-
from mayavi import mlab
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw
from tqdm import tqdm, trange
import time
import warnings
from scipy.ndimage.filters import gaussian_filter

def read_input_file(path):
    return parameter_list

def plWave(parameter_list): # create sinewave planetary scale wave
    return wave_modulator:

def elVortic(parameter_list): # 

def meshgrid_image_exporter(waveconfig):
    plWave()
    elVortic()
    return image_mayavi

def time_series_builder(parameter_list, writeFlux = False):
    image = meshgrid_image_exporter()
    imagegray = convert
    flux_value_creator = function(imagegray)
    return fluxarray
    
if name == __main__:
    read_input_file()
    time_series_builder()
    if parameter_list['save_flux'] is True:
        save flux
    if parameter_list['save_animation'] is True:
        save animation
    if parameter_list['save_'] is True: