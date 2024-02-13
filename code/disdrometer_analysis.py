#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:18:31 2024

@author: idariash
"""

import pydsd as pyd

import numpy as np
import matplotlib.pyplot as plt
from datetime import timezone,  timedelta
import datetime

from pytmatrix import tmatrix

from pytmatrix.tmatrix import Scatterer
from pytmatrix.psd import PSDIntegrator, GammaPSD, BinnedPSD
from pytmatrix import orientation, radar, tmatrix_aux, refractive

import os

import matplotlib as mpl
from pylab import cm
from matplotlib.dates import DateFormatter
import pytz
import matplotlib.colors as mcolors

def plot_time_series(Data_Path):
    
    files = os.listdir(Data_Path)
    files = sorted(files)
    
    Bin_Size = np.empty((0, 32))
    Bin_Counts = np.empty((0, 32))
    Times = []
    
    for file in files:
        filename = os.path.join(Data_Path, file)
        dsd = pyd.read_parsivel(filename)
        dsd.calculate_dsd_parameterization()
        bin_size = dsd.diameter['data']
        bin_counts = dsd.fields['Nd']['data'][0]
        time = datetime.datetime.fromtimestamp(dsd.time['data'][0]) 
        time = time + timedelta(hours=1) # adding an hour to match central time
        
        Bin_Size = np.append(Bin_Size, [bin_size], axis=0)
        Bin_Counts = np.append(Bin_Counts, [bin_counts], axis=0)
        Times.append(time)
        #formatted_dates = [date.strftime("%M:%S") for date in Times]
        
    fig = plt.figure(figsize = [6,4])
    colors = [("white")] + [(cm.jet(i)) for i in range(1, 256)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("new_map", colors, N=256)
    plt.pcolormesh(Times, bin_size, Bin_Counts.T, cmap = cmap,
                         norm=mcolors.LogNorm(vmin=1, vmax=1E4))
    plt.axis("tight")
    plt.ylim(0, 7)
    hh_mm = DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(hh_mm)
    
    plt.xlabel('Time (Central)')
    plt.ylabel('Diameter (mm)')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts (mm^-1 m^-3)')
    
    # Set limits for the color bar
    cbar.set_clim(1E0, 1E4)
    
    return fig

def main():

    Data_Path = '/net/k2/storage/people/idariash/home/CSU/DFW/data/Charlie/20240108'
    '/net/k2/storage/people/idariash/home/CSU/DFW/data/20240108'
    plot_time_series(Data_Path)
    
if __name__ == "__main__":
    main()



    

