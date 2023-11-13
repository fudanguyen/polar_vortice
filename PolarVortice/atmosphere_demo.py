#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:46:15 2023

@author: nguyendat
"""

import pickle
from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from PIL import Image, ImageDraw
import warnings
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from sklearn.decomposition import PCA
import datetime
import cv2 

### IMPORTANT: create offscreen rendering to save computation
mlab.options.offscreen = True

### Path management
import os
from os.path import join
folderarray = os.path.abspath('').split('/')
homedir = '/'
for i in range(len(folderarray)):
   homedir = join(homedir, folderarray[i])
plotPath = join(homedir, 'plot/')

########## Construct Mayavi spherical mesh #######################
### Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin

### Construct the spherical mesh grid
phi0 = np.linspace(0, np.pi, 200)
theta0 = np.linspace(0, 2*np.pi, 200)
phi, theta = np.meshgrid(phi0, theta0)

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

xsize, ysize = x.shape[0], x.shape[1]

#################### FUNCTIONS FOR CREATING ATMOSPHERIC FEATURES ##########

############## Create Planetary-scale waves ##############################
### unit: hour, rotational period: 5 hour
def plWave(x, value, t, w=xsize, f=0, amplitude=0.4, base=0.5, RP=5): # planetary scale waves
    # create the sine-wave modulation, such that the sine wave looks the same after
    # every rotational period RP
    # w is the spatial period equals to the pixel size
    sine = amplitude*np.sin(2*np.pi/w * (x + (t/RP)*w) + f*np.pi/180) + base
    # return the original value + sine wave
    return value + sine
############## Create uniform time-variant polar-layer ##############################
def polar_change(value=0, amplitude=0.25, t=0, f=0, RP=60):
    # value: polar cap base flux value
    flux = value + amplitude*np.sin(2*np.pi/RP*t + f*np.pi/180)
    return flux
############## QUALITY OF LIFE ##############
##### Create equi-distant list for vortices center coordinate ######
def equidistant_values(X0, X1, n):
    N = n+1
    if N < 1:
        raise ValueError("N must be greater than or equal to 1")
    if N == 1:
        return [X0]
    step = (X1 - X0) / (N - 1)
    equidistant_list = [X0 + i*step + step/2 for i in range(N)]
    return np.array(equidistant_list[:-1])
####### convert latitude to pixel location ############
def lat(coord,n=ysize):
    return abs(coord-90)/180*n # pixel location
####### convert longitude to pixel location ############
def long(coord,n=xsize):
    return abs(coord)/360*n # pixel location
####### area of region bounded by two latitudes #######
def area_bounded_latitudes(lat1, lat2, radius=1):
    ## lat1 > lat2, lat in degree
    return 2*np.pi * radius * (np.sin(lat2*np.pi/180)-np.sin(lat1*np.pi/180))
#%% Vortices generator
def circle_vortice(recmap, coord, t=0, RP=60, number=5, variable_vortice=False,
                   phaseEnhancementFactor=1):
    # draw circular patch near the polar region
    # coord: [[lat1, lat2], ...]
    # spacing: equidistance between [number] of circular vortices
    # radius such that area_vortice / area_cap = percentage
    
    ### Elliptical vortices prop:
    # Setup each vortice to be individually variable (each with phase offset)
    radiusfrac = 0.3
    a, b= 0.75, 0.25
    
    if variable_vortice:
        phaseValue = [0, -2, 5, -7, 11, 2, 4, -6, -8]
    else:
        phaseValue = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for array in coord:
        lat1, lat2 = array[0], array[1]
        center_coord_long = equidistant_values(0, 360, n=number)
        center_coord_lat = (lat1+lat2)/2
        centerlat = int(lat(center_coord_lat))
        
        # Variable vortices
        amplitude = recmap[centerlat, int(xsize/2)]
        variableflux = []
        for i in range(number):
            fluxchange = polar_change(value=amplitude, amplitude=0.15, 
                                      t=t, f=phaseEnhancementFactor*phaseValue[i], RP=RP)
            variableflux.append(fluxchange)
        
        # Set up elliptical vortices
        r_vortice = np.sqrt(radiusfrac*area_bounded_latitudes(lat1, lat2, radius=1))*(xsize/np.pi)
        ar, br = a*r_vortice, b*r_vortice
        
        for xx in range(xsize):
            for yy in range(ysize):
                if lat(lat2) < yy < lat(lat1):
                    for i, xi in enumerate(long(center_coord_long)):
                        centerlong = int(xi)
                        if (xx-centerlong)**2/ar**2 + (yy-centerlat)**2/br**2 < 1:
                            recmap[xx, yy] = amplitude + variableflux[i]
        return recmap
    
# ## polar_lats = [[65, 90],[-90, -65]]
# mapper = s.copy()
# polar_lats = [[65, 90]]
# vormap = circle_vortice(mapper, polar_lats, variable_vortice=True)
# ## plot
# plt.close()
# fig, ax = plt.subplots(figsize=(4,4), dpi=150)
# ax.imshow(vormap)

#%% RUN THE ATMOSPHERE EVOLUTION 3D MODEL
###############################
warnings.filterwarnings("ignore")
###############################
## Save each frame
save = False
# save = True
###############################
# draw_animation = False
draw_animation = True
#### save animation as gif
# save_animation = False
save_animation = True
# save_still = True # save the first still frame
save_still = False 
###############################
# save_specmap = True
save_specmap = False
###############################
#### save or run command
# run_visualization_again, save_output_array = True, True
run_visualization_again, save_output_array = True, False
# run_visualization_again, save_output_array = False, False
# 
#############################################################################################
### RUN THE VISUALIZATION OR READ PAST RESULTS
#### model config
### Sine Amplitutde
Fband= 1.0
Fambient = 0.25
Fpolar = 0.55
### periods
Pband = 5.0
Ppol = 120.0

##############################################################
######### Modulation config
## Polar color are different from band color, but the question is
## where the long-term modulation comes from band or pole.
## Three cases to solve this problem.

######### Configuration: Comment out to choose a config
#### Simple, all planetary-scale sinusoid models
# modu_config = 'all_sinusoid'

##### Case A: Uniform polar cap
# modu_config = 'polarCap'

##### Case B: Polar cap with heterogenous spatial features (spots)
modu_config = 'polarVortice'

##### Case C: No polar cap, long-term change comes from bands.
# modu_config = 'noPolar'

##############################################################

### Lookup Map Spectra Dict Key
speckey = {'A':0.25, 'B':0.58, 'P':0.75}
### Observation run 

# frame_no = 100
# t0, t1 = 0,15

frame_no = 45
t0, t1 = 0,120
#### Time array construction
option_time_array = 'simple'# simple: generate continuous evenly-spaced-time-array
# option_time_array = 'multivisit'# simple: generate continuous evenly-spaced-time-array'
#### Color mapping
cmap = 'plasma'

if not run_visualization_again:
    modelname = input('modelname:')
    inclin = float(input('inclination:'))
    grayArrayPath = homedir+'/output/[%s][%s]_[inclin=%i]_[grayArray].pkl'%(modu_config, modelname, inclin)
    metadataPath = homedir+'/output/[%s][%s]_[inclin=%i]_[metadata].pkl'%(modu_config, modelname, inclin)
    with open(grayArrayPath, 'rb') as file:
        gray_array = pickle.load(file)
    with open(metadataPath, 'rb') as file:
        metadata = pickle.load(file)
else:    
    #### model name
    # modelname = 'BD_B'
    # config = [[45, 30, Fband, 'B', 10, Pband/2],
    #           [20, 5, Fband, 'B', 60, Pband], 
    #           [89, 65, Fband, 'P', -30, Ppol]]
    # 
    # modelname = 'BD_C'
    # config = [[89, 70, Fband, 'P', -30, Ppol],
    #           [45, 35., Fband, 'B', 10, Pband/2],
    #           [22, 15, Fband, 'B', 150, Pband], 
    #           [-10, -20, Fband, 'B', -26, Pband/2],
    #           [-30, -40, Fband, 'B', 135, Pband],
    #           [-72, -89, Fband, 'P', 15, Ppol]]
    #
    # modelname = 'BD_D'
    # config = [[88, 77, Fband, 'P', -30, Ppol],
    #           [72, 65, Fband, 'P', -70, Ppol],
    #           [53, 46., Fband, 'B', 100, Pband/2],
    #           [35, 27., Fband, 'B', -10, Pband/2],
    #           [22, 15, Fband, 'B', 150, Pband],
    #           [8, 0, Fband, 'B', -50, Pband],
    #           [-10, -18, Fband, 'B', 56, Pband],
    #           [-25, -34, Fband, 'B', 16, Pband],
    #           [-40, -48, Fband, 'B', 85, Pband/2],
    #           [-54, -60, Fband, 'B', -35, Pband/2],
    #           [-68, -75, Fband, 'P', 5, Ppol],
    #           [-77, -88, Fband, 'P', 15, Ppol]]
    
    modelname = 'BD_E'
    config = [[90, 65, Fpolar, 'P', -30, Ppol],
              [45, 35., Fband, 'B', 10, Pband/2],
              [22, 15, Fband, 'B', 150, Pband], 
              [-10, -20, Fband, 'B', -26, Pband/2],
              [-30, -40, Fband, 'B', 135, Pband],
              [-65, -90, Fpolar, 'P', 15, Ppol]]
    
    #### inclination
    # inclin = 0
    inclin = +10
    # inclin = +20
    # inclin = +30
    # inclin = +45
    # inclin = +70
    # inclin = +90
    # inclin = -10
    # inclin = -30
    # inclin = -45
    # inclin = -70
    # inclin = -90
    
    ### Build metadata list
    metadata = {}
    metadata['modu_config'] = modu_config
    metadata['modelname'], metadata['inclin'] = modelname, inclin
    metadata['Fband'], metadata['Fambient'] = Fband, Fambient
    metadata['Pband'], metadata['Ppol'] = Pband, Ppol
    metadata['config_columns'] = ['lat2', 'lat1', 'amp', 'typ', 'phase', 'period']
    metadata['config'] = config
    
    ###### Create an atmosphere mesh 
    ### Edit the property of bands here!
    def atmos_mesh(x, config, t=0, spec=False):
        im = Fambient*np.ones(x.shape)
        sm = 0.25*np.ones(x.shape)
        # lat = theta0*180/pi-90    
        for xx in range(xsize):
            for yy in range(ysize):
                ############# Adding latitude-dependent features ############# 
                
                ### The lat() function converts latitude to pixel coordinate
                ### The order is reverse: input the latitudes from large to small using
                ### the format: lat(LARGE) < yy < lat(SMALL)
                for group in config:
                    lat2, lat1, amp, typ, phase, period = group
                    #### Band-type modulation
                    if typ in ['B', 'b']:
                        if lat(lat2) <= yy <= lat(lat1): 
                            im[xx,yy] = plWave(x=xx, value=amp, f=phase, t=t, base=0, RP=period)
                            if spec: 
                                sm[xx,yy] = speckey[typ]
                    #### Polar-type modulation:            
                    elif typ in ['P', 'p']:
                        if lat(lat2) <= yy <= lat(lat1):
                            if modu_config == 'noPolar':
                                im[xx,yy] = amp
                            else:
                                im[xx,yy] = polar_change(value=amp, t=t, f=phase, RP=period)
                            if spec: 
                                sm[xx,yy] = speckey[typ]
                                
        ############# Adding vortices features ############# 
        if modu_config == 'polarVortice':
            polar_lats = []
            for group in config:
                lat2, lat1, amp, typ, phase, period = group
                if typ in ['P', 'p']:
                    polar_lats.append([lat1, lat2])
            im = circle_vortice(im, polar_lats, t=t, RP=period, 
                                phaseEnhancementFactor=1, variable_vortice=True)
                    
        if not spec: 
            return im
        else:
            return im, sm
    
    ################# GENERATE SPECTRAL LOOKUP MAPS #####################
    ### spectral lookup map; Run once only
    ### SPECTRA WINDOW
    # inclin = 0
    sfig = mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(0, 0, 0), size=(500, 500))
    mlab.view(azimuth=180.0, elevation=90.0-inclin, distance=2.0, roll=90)
    _, s = atmos_mesh(x, config, spec=True)
    sphere_spec = mlab.mesh(x, y, z, scalars=s, colormap='plasma', vmin=0, vmax=1, figure=sfig)
    sphere_spec.actor.property.lighting = False
    sfig.scene.parallel_projection = True
    sfig.scene.reset_zoom()
    scamera = sfig.scene.camera
    scamera.zoom(1.57)
    
    specmap = mlab.screenshot(figure=sfig, mode='rgba', antialiased=True)
    mlab.clf()
    mlab.close('all')
    
    specgray = specmap[:,:,0] * 0.2989 + specmap[:,:,1] * 0.587 + specmap[:,:,2] * 0.114
    ### Condition
    is_amb = ((specgray >= 0.2) & (specgray < 0.3)).astype(int)
    is_band, is_pol = ((specgray >= 0.5) & (specgray < 0.6)).astype(int) , ((specgray >= 0.6) & (specgray < 0.75)).astype(int)
    total_count = is_amb.sum() + is_band.sum() + is_pol.sum()
    
    ####################### CREATE PHOTOMETRY ######################################
    ### Create list of images
    imarray = []
    fluxarray = []
    gray_array = []
    
    ### create time_array in hours unit
    if option_time_array == 'simple':
        # simple evenly-spaced time array
        time_array = np.linspace(t0,t1,frame_no)
        
    if option_time_array == 'multivisit':
        # create multivisit
        time_array = [0]
    
    ### Create matplotlib figure
    plt.close('all')
    fig2, ax2 = plt.subplots()
    ### create 3d figure
    mfig = mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(0, 0, 0), size=(500, 500))
        
    ### Start looping through the timesteps array
    for i in range(frame_no):
        ti = time_array[i]
        ######### Create 3d mesh in Mayavi THEN paste into matplotlib figures ##########
    
        ### configuring the 3d view and camera
        mlab.view(azimuth=180.0, elevation=90.0-inclin, distance=2.0, roll=90)
    
        ### Do the meshplot 
        s = atmos_mesh(x,config,t=ti)
        sphere = mlab.mesh(x, y, z, scalars=s, colormap=cmap, vmin=0, vmax=1)
    
        ### Set up cosmetics
        sphere.actor.property.lighting = False
        mfig.scene.parallel_projection = True
        mfig.scene.reset_zoom()
        camera = mfig.scene.camera
        camera.zoom(1.57)
    
        ### Take screenshot as array object
        imgmap = mlab.screenshot(figure=mfig, mode='rgba', antialiased=True)
        mlab.clf()
    
        ######### Pasting into matplotlib figures ############################
        ### Show mayavi im as Matplotlib object and draw animation
        if draw_animation: 
            imobj = ax2.imshow(imgmap, vmin=0, vmax=1, animated=True, cmap=cmap)
            if i == 0:
                ax2.imshow(imgmap, vmin=0.2, vmax=1, cmap=cmap) # show an initial one first
                # plt.show()
        else:
            imobj = ax2.imshow(imgmap, vmin=0, vmax=1, cmap=cmap)
        imarray.append([imobj])
    
        if save:
            plt.margins(0, 0), plt.draw(), plt.axis('off')
            fig2.savefig(plotPath+'ani_%04d.jpeg'%i, format='jpeg', bbox_inches = 'tight', pad_inches=0)
        
        ### convert map into grayscale
        imgray = imgmap[:,:,0] * 0.2989 + imgmap[:,:,1] * 0.587 + imgmap[:,:,2] * 0.114
        gray_array.append([imgray])
        if ((i+1)/frame_no*100)%10 == 0.0: 
            print('%i%% done'%((i+1)/frame_no*100))
    
    ### plotting
    if draw_animation:
        ani = animation.ArtistAnimation(fig2, imarray)
        if save_animation:
            ### To save the animation using Pillow as a gif
            writer = animation.PillowWriter(fps=60,
                                            metadata=dict(artist='Me'),
                                            bitrate=1800)
            ani.save(plotPath+'[%s][%s]_[i=%i]_[ani_%i-%i_no=%i].gif'%(modu_config, modelname, inclin, t0, t1, frame_no), writer=writer)
        # plt.show()
        plt.close()
        
#####################################################################################
##### OUTPUT HANDLING AND METADATA WRITER
# 1. write output image cube file
# 2. write output FLUX file
# 3. write model metadata: numbers of period bands, periods, location, types, fambient, fband, fpole, 
#########################################
## WRITE PICKLE OUTPUT FOR GRAY_ARRAY AND METADATA
if save_output_array:
    grayArrayPath = homedir+'/output/[%s][%s]_[inclin=%i]_[grayArray].pkl'%(modu_config, modelname, inclin)
    metadataPath = homedir+'/output/[%s][%s]_[inclin=%i]_[metadata].pkl'%(modu_config, modelname, inclin)
    ### Write file
    with open(grayArrayPath, 'wb+') as file:
        pickle.dump(gray_array, file)
    with open(metadataPath, 'wb+') as file:
        pickle.dump(metadata, file)

### Plot first photometry frame
plt.close(), plt.figure(dpi=300), 
plt.imshow(gray_array[0][0], vmin=0.0, vmax=1, cmap='inferno')
if save_still:     
    plt.savefig(plotPath+'[%s][%s]_[i=%i]_[still].png'%(modu_config, modelname, inclin), format='png', dpi=300)
plt.show()

### Plot specmap
plt.close(), plt.figure(dpi=300)
plt.imshow(specmap, vmin=0, vmax=1)
fracA, fracP, fracB = is_amb.sum()/total_count, is_pol.sum()/total_count, is_band.sum()/total_count
plt.title('Spectra Area Coverage: [A]=%.2f, [P]=%.2f, [B]=%.2f'%(fracA, fracP, fracB))
if save_specmap:
    plt.savefig(plotPath+'[%s]%s_[spectraCoverageMap]_i=%i.png'%(modu_config, modelname, inclin), format='png', dpi=300)
plt.show()

#%% PLOT 'FLUX' EVOLUTION
### Find flux by applying gaussian filter sigma=50
# save = True
save = False 

print(metadata)

typfluxratio_calc = True
do_gaussian_filter = False
gaussian_sigma = 0
fluxtyp = []
periodConfigList = [metadata['config'][i][5] for i in range(len(metadata['config']))]

flux = []
for frame in range(frame_no):
    frameim = gray_array[frame][0]
    if do_gaussian_filter:
        frameim=gaussian_filter(frameim, sigma=gaussian_sigma)
    else: frameim = np.copy(frameim)
    
    flux.append(np.sum(frameim))
    if typfluxratio_calc:
        ## mask-out band and pole and calculate their 
        ## respective flux contribution at each frame
        ambim, polim, bandim = frameim * is_amb, frameim * is_pol, frameim * is_band
        fluxtyp.append([polim.sum(), bandim.sum(), ambim.sum()])

flux = np.array(flux)
fluxtyp = np.array(fluxtyp)
polflux, bandflux = 0.5*fluxtyp[:,0]/np.mean(flux), 0.5*fluxtyp[:,1]/np.mean(flux)
ambflux = fluxtyp[:,2]/np.mean(flux)

def bring_to_one(array):
    shift = (array.max() + array.min())/2
    return 1 + array - shift
    # return array

plt.close('all'), plt.figure(figsize=(8,3.5), dpi=300)
normflux = flux / flux.mean()
normflux = polflux + bandflux + ambflux
plt.plot(time_array, bring_to_one(normflux), c='xkcd:purple', label='Total Flux', marker='^', ms=5, ls='-', lw=2.5)
plt.plot(time_array, bring_to_one(bandflux), c='xkcd:red', label='Band Flux: Short-period', ls='--', lw=1.5)
plt.plot(time_array, bring_to_one(polflux), c='xkcd:orange', label='Polar Flux: Long-period', ls='-.', lw=1.5)
plt.axhline(y=1, alpha=0.3, ls='--', c='k', lw=1.5)
# plt.xlim(t0, t1), plt.ylim(0.968, 1.048), plt.xlim(0,12)
# plt.title(r'Model: "%s", i=%i$^\circ$, p$_i$=%s hr'%(metadata['modelname'], metadata['inclin'], periodConfigList))
plt.xlabel('Time (hours)'), plt.ylabel('Normalized Flux')
plt.legend()
if save:
    plt.savefig(plotPath+'[%s]%s_[flux]_i=%i.png'%(metadata['modu_config'], metadata['modelname'], metadata['inclin']), format='png', dpi=300)

#%% SPECTRA MAKER
# save = True
save = False
testPlot = True
# testPlot = False

###### Use 2M3139 sample spectra from Apai 2013 as the base
header = ['wave', 'flux']
unbinfilename = '/Users/nguyendat/Documents/GitHub/polar_vortice/polar_vortice/data/sm3139_spectra_reduced.csv'
sm3139_spec = pd.read_csv(unbinfilename, names=header)
plt.close(), plt.figure(dpi=100), plt.title('2M3139 HST')
plt.plot(sm3139_spec.wave, sm3139_spec.flux, marker='o', ls='--')
spectra = sm3139_spec[::2].copy()
spectra['err'] = spectra.flux*0.04

###### SPECTRA MAKER
### CASE: Gaussian Peak
def polar_spectra_gaussian(lamda):
    mean = 1.6
    std = 2.5e-2
    return 0.22*np.exp(-(lamda - mean) ** 2 / (2 * std ** 2))

def bands_spectra_gaussian(lamda):
    mean = 1.27
    std = 5.5e-2
    return 0.13*np.exp(-(lamda - mean) ** 2 / (2 * std ** 2))

peakFunctionType = 'gaussian'
if peakFunctionType == 'gaussian':
    bandspec = bands_spectra_gaussian(spectra.wave)
    polarspec = polar_spectra_gaussian(spectra.wave) 
    spectraType = 'Gaussian Peak'

if testPlot:
    plt.close(), 
    fig, ax = plt.subplots(figsize=(6,3),dpi=300)
    ax.set_xlabel('Wavelength (um)'), ax.set_ylabel('Normalize Intensity')
    
    ax.set_title('%s Spectra: Synthetic spectra at t=0'%(spectraType))
    bands = spectra.flux + bandspec
    ax.plot(spectra.wave, bands, label='Base + Bands spectra', c='xkcd:red', ls='--', marker='+', ms=6, lw=1)
    ax.plot(spectra.wave, bandspec, ls='--', c='xkcd:red', ms=10, lw=1.5, alpha=0.6)
    
    polar = spectra.flux + polarspec
    ax.plot(spectra.wave, polar, label='Base + Polar spectra', c='xkcd:orange',ls='-', marker='+', ms=6, lw=1)
    ax.plot(spectra.wave, polarspec, ls='-', c='xkcd:orange', ms=10, lw=1.5, alpha=0.6)
    
    basespec = spectra.flux
    ax.errorbar(spectra.wave, spectra.flux, yerr=spectra.err, c='xkcd:purple', label='Base spectra',ls='-', lw=2.0, marker='.', ms=10) 
    # ax.legend(), 
    ax.set_xlim(1.12, 1.72)
    plt.show() 
    # plt.show()
    if save:
        plotOut = plotPath + '/[%s]spectra_at_t0'%peakFunctionType
        fig.savefig(plotOut+'.pdf', dpi=300, format='pdf', bbox_inches = 'tight')
        fig.savefig(plotOut+'.png', dpi=300, format='png', bbox_inches = 'tight')
#%% Spectral cube maker
# save = True
save = False
testPlot = True
# testPlot = False

## Generate a modulated spectra
lam = spectra.wave.values
dataTable = [lam]
spectraCube2 = []
spectraCube, spectraCubeNoTime = [], []
# noiseAddition = 0.01*np.random.randn(t.shape[0])

for i,timestep in enumerate(time_array): 
    modulatedSpectra = np.array(basespec + bandflux[i]*bandspec + polflux[i]*polarspec)
    timetup = timestep * np.ones(len(lam))
    spectraCube.append((timestep, (lam, modulatedSpectra)))
    # print(table0)
    table0 = (timetup, lam, modulatedSpectra)
    for i,lamda in enumerate(lam):
        spectraCube2.append((timetup[i], lamda, modulatedSpectra[i]))
    spectraCubeNoTime.append((lam, modulatedSpectra))
    dataTable.append(modulatedSpectra)
dataTable = np.transpose(dataTable)

if testPlot:
    # Convert to np.array
    spectraCubeNoTime = np.array(spectraCubeNoTime)
    plt.close()
    f, axe = plt.subplots(figsize=(6,4), dpi=120)
    axe.set_title('Spectra at all Timestamp')
    for i in range(len(time_array)):
        specnum = spectraCube[i]
        axe.plot(specnum[1][0], specnum[1][1], ls='-', marker='.', ms=3, c='tab:blue', alpha=0.05)
    axe.set_xlabel('Wavelength (um)'), axe.set_ylabel('Intensity')
    
    if save:
        plotOut = plotPath + '/[%s][%s]_spectra_at_3_timestamp_[%s][i=%i]'%(metadata['modu_config'], 
                                                                            metadata['modelname'],
                                                                            peakFunctionType, 
                                                                            metadata['inclin'])
        fig.savefig(plotOut+'.pdf', dpi=300, format='pdf', bbox_inches = 'tight')
        fig.savefig(plotOut+'.png', dpi=300, format='png', bbox_inches = 'tight')
        
#%% Implement PCA analysis
df = pd.DataFrame(np.transpose(dataTable))
# save = True
save = False

### implement a PCA with n_components
# if n_component > 1: choose n first-most-important component
# if n_component < 1: choose percertage value of component sum that preserves n% data variability
pca = PCA(n_components=0.9999)
pca = PCA(n_components=3)
pca.fit(df)
X_pca = pca.transform(df)
# pca.components_

condplot = [1,1,1,1]
## spectral PCA, time PCA, PCA strength bar plot, wavelength component over time
print("Relative variance in principal components:", pca.explained_variance_ratio_)
#########################################################################
#### SPECTRAL COMPONENT PLOT
if condplot[0]:
    color = {}
    fig1, ax1 = plt.subplots(figsize=(6,4),dpi=120)
    ax1.set_title('Spectral Component: Strength over Wavelength', weight='bold')
    for i, comp in enumerate(pca.components_):
        valuecomp = comp
        g = ax1.plot(lam, np.abs(valuecomp), ls='--', ms=4, lw=1.5, label='Component %i'%(i+1))
        ax1.set_xlabel('Wavelength (um)'), ax1.set_ylabel('Absolute Strength')
        ax1.legend()
        color[i] = g[0].get_color()
    if save:
        plotOut = plotPath + '/[%s][%s]_PCA_SpectralCompPlot_[%s][i=%i]'%(metadata['modu_config'], 
                                                                        metadata['modelname'],
                                                                        peakFunctionType, 
                                                                        metadata['inclin'])
        fig1.savefig(plotOut+'.pdf', dpi=300, format='pdf', bbox_inches = 'tight')
        fig1.savefig(plotOut+'.png', dpi=300, format='png', bbox_inches = 'tight')
    
#########################################################################
#### TEMPORAL COMPONENT PLOT
if condplot[1]:
    fig2, ax2 = plt.subplots(figsize=(9,4),dpi=120)
    ax2.set_title('Temporal Component: Strength over Time', weight='bold')
    for i, comp in enumerate(np.transpose(X_pca)):
        print(comp)
        ax2.plot(time_array, comp[1:], ls='-', lw=1, label='Component %i'%(i+1), c=color[i])
        ax2.set_xlabel('Time (hours)'), ax2.set_ylabel('Components strength'), ax2.legend()
    if save:
        plotOut = plotPath + '/[%s][%s]_PCA_TimeCompPlot_[%s][i=%i]'%(metadata['modu_config'], 
                                                                    metadata['modelname'],
                                                                    peakFunctionType, 
                                                                    metadata['inclin'])
        fig2.savefig(plotOut+'.pdf', dpi=300, format='pdf', bbox_inches = 'tight')
        fig2.savefig(plotOut+'.png', dpi=300, format='png', bbox_inches = 'tight')
    
#########################################################################
#### BAR PLOT: explained_variance
if condplot[2]:
    fig3, ax3 = plt.subplots(figsize=(4,4),dpi=120)
    for i, variance in enumerate(pca.explained_variance_):
        ax3.bar(i,variance, label='Component %i'%(i+1), color=color[i])
    ax3.set_xlabel('PCA Feature'), ax3.set_ylabel('Explained variance')
    ax3.set_title('PCA: Explained Variance', weight='bold')
    if save:
        plotOut = plotPath + '/[%s][%s]_PCA_VarianceBarPlot_[%s][i=%i]'%(metadata['modu_config'], 
                                                                        metadata['modelname'],
                                                                        peakFunctionType, 
                                                                        metadata['inclin'])
        fig3.savefig(plotOut+'.pdf', dpi=300, format='pdf', bbox_inches = 'tight')
        fig3.savefig(plotOut+'.png', dpi=300, format='png', bbox_inches = 'tight')

#########################################################################
#### Flux via time: Component strength over time at 1.3um and 1.6um
if condplot[3]:
    a = np.where(np.logical_and(lam>1.28, lam<1.29))[0]
    b = np.where(np.logical_and(lam>1.59, lam<1.60))[0]
    print(a,b)
    
    df1 = df.iloc[:, a][1:].values
    df2 = df.iloc[:, b][1:].values
    
    plt.close()
    plt.figure(figsize=(10,5), dpi=120)
    plt.title('Component strength over time at 1.3um and 1.6um')
    
    plt.plot(time_array, df1/df1.max(), label='1.3um feature - band component', marker='o', ls='-', ms=3)
    plt.plot(time_array, df2/df2.max(), label='1.6um feature - polar component', marker='o', ls='-', ms=3)
    
    plt.xlabel('Time (hours)'), plt.ylabel('Component strength')
    plt.legend()
#%% TEST1: Vortices generator
def circle_vortice(recmap, coord, t=0, RP=60, number=5, variable_vortice=False,
                   phaseEnhancementFactor=1):
    # draw circular patch near the polar region
    # coord: [[lat1, lat2], ...]
    # spacing: equidistance between [number] of circular vortices
    # radius such that area_vortice / area_cap = percentage
    
    ### Elliptical vortices prop:
    # Setup each vortice to be individually variable (each with phase offset)
    radiusfrac = 0.3
    a, b= 0.75, 0.25
    
    if variable_vortice:
        phaseValue = [0, -2, 5, 5, -7, 11, 2, 4, -6, -8]
    else:
        phaseValue = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for array in coord:
        lat1, lat2 = array[0], array[1]
        center_coord_long = equidistant_values(0, 360, n=number)
        center_coord_lat = (lat1+lat2)/2
        centerlat = int(lat(center_coord_lat))
        
        # Variable vortices
        amplitude = recmap[centerlat, int(xsize/2)]
        variableflux = []
        for i in range(number):
            fluxchange = polar_change(value=amplitude, amplitude=0.35, 
                                      t=t, f=phaseEnhancementFactor*phaseValue[i], RP=RP)
            variableflux.append(fluxchange)
        
        # Set up elliptical vortices
        r_vortice = np.sqrt(radiusfrac*area_bounded_latitudes(lat1, lat2, radius=1))*(xsize/np.pi)
        ar, br = a*r_vortice, b*r_vortice
        
        for xx in range(xsize):
            for yy in range(ysize):
                if lat(lat2) < yy < lat(lat1):
                    for i, xi in enumerate(long(center_coord_long)):
                        centerlong = int(xi)
                        if (xx-centerlong)**2/ar**2 + (yy-centerlat)**2/br**2 < 1:
                            recmap[xx, yy] = amplitude + variableflux[i]
        return recmap
    
# ## polar_lats = [[65, 90],[-90, -65]]
# mapper = s.copy()
# polar_lats = [[65, 90]]
# vormap = circle_vortice(mapper, polar_lats, variable_vortice=True)
# ## plot
# plt.close()
# fig, ax = plt.subplots(figsize=(4,4), dpi=150)
# ax.imshow(vormap)


