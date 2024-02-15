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

import pyart
import netCDF4


def plot_nexrad_ppi_ref(filename, sweep):
    
    radar = pyart.io.read(filename)
    x = sweep
    angles = radar.fixed_angle['data'];
    display = pyart.graph.RadarDisplay(radar)
    
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_invalid('reflectivity')
    #gatefilter.exclude_invalid('differential_phase')
    gatefilter.exclude_outside('reflectivity', 0, 100)
    #gatefilter.exclude_outside('normalized_coherent_power', 0.5, 1)
    gatefilter.exclude_outside('cross_correlation_ratio', 0.9, 1)
    
    angle = str( round(angles[x],2))
    fig = plt.figure(figsize = [10,8])
    
    time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    time_text = time_start.strftime('%Y-%m-%dT%H:%M:%S')
           
    #  Reflectivity               
    display.plot_ppi('reflectivity', sweep = x , # axislabels = ['', 'North South distance from radar (km)'],
                    title = 'Reflectivity', colorbar_label='Ref. (dBZ)',
                    vmin = 0, vmax = 70, mask_outside = False,
                    cmap = pyart.graph.cm.NWSRef, gatefilter = gatefilter) #pyart.graph.cm.NWSRef
    #display.plot_range_rings([25, 50, 75])
    display.plot_grid_lines(ax=None, col='k', ls=':')
    display.set_limits(xlim=[-100,100], ylim=[-100,100])
    
    radar_name = radar.metadata['instrument_name']
    plt.suptitle(radar_name + ' | Elevation: ' + angle + ' Deg.| ' + time_text
                  + ' UTC', fontsize=16)
    
    return fig



def plot_DSD(filename):
    dsd = pyd.read_parsivel(filename)

    fig = plt.figure(figsize = [5,4])
    plt.scatter(dsd.diameter['data'], dsd.fields['Nd']['data'])
    plt.yscale('log')
    plt.ylim(10E-2, 10E5)
    plt.xlim(0, 5)
    
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Concentration (mm^-1 m^-3)')
    
    time = datetime.datetime.fromtimestamp(dsd.time['data'][0])
    time = time + timedelta(hours=1)
    
    #time = str(datetime.datetime.fromtimestamp(dsd.time['data'][0]))
    time = str(time)
    plt.title(f'Measured DSD at {time}')
    return fig

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

    filename = '/net/k2/storage/people/idariash/home/CSU/DFW/radar/KFWS20240108_173658_V06'
    sweep = 0
    plot_nexrad_ppi_ref(filename, sweep)
    
if __name__ == "__main__":
    main()



    

