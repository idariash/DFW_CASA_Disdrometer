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


def find_closest_gate(radar, target_lat, target_lon, sweep = None):
    """Find the closest radar gate to a given latitude, longitude."""
    
    if sweep is None:
        sweep = 0
        
    radar_lat = radar.latitude['data'][0]
    radar_lon = radar.longitude['data'][0]
    radar_alt = radar.altitude['data'][0]
    
    airport_alt = 185

    # Convert latitude, longitude, and altitude to Cartesian coordinates
    target_x, target_y = pyart.core.geographic_to_cartesian_aeqd(
        target_lon, target_lat, radar_lon, radar_lat)
    # Compute distance to each gate
    X, Y, Z = radar.get_gate_x_y_z(sweep = sweep)

    distances = np.sqrt((X - target_x)**2 + (Y - target_y)**2)

    # Find the index of the closest gate
    closest_gate_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    min_distance = np.amin(distances)

    return closest_gate_index


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

def plot_nexrad_ppi_ref_near_DFW_airport(filename, sweep):
    
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
    fig = plt.figure(figsize = [5,4])
    
    time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    time_text = time_start.strftime('%Y-%m-%dT%H:%M:%S')
           
    #  Reflectivity               
    display.plot_ppi('reflectivity', sweep = x , # axislabels = ['', 'North South distance from radar (km)'],
                    title = '', colorbar_label='Reflectivity (dBZ)',
                    vmin = 0, vmax = 70, mask_outside = False,
                    cmap = pyart.graph.cm.NWSRef, gatefilter = gatefilter) #pyart.graph.cm.NWSRef
    #display.plot_range_rings([25, 50, 75])
    display.plot_grid_lines(ax=None, col='k', ls=':')
    display.set_limits(xlim=[20,30], ylim=[30,40])
    
    charlie_lat = 32.91365
    charlie_lon = -97.05752
    charlie_location = [charlie_lat, charlie_lon]
    
    SEHP_lat = 32.87544
    SEHP_lon = -97.03275
    SEHP_location = [SEHP_lat, SEHP_lon]
    
    SWHP_lat = 32.87729
    SWHP_lon = -97.04658
    SWHP_location = [SWHP_lat, SWHP_lon]
    
    
    display.plot_label(label = '', location = charlie_location, symbol='r+', 
                       text_color='k', ax=None)
    display.plot_label(label = '', location = SEHP_location, symbol='r+', 
                       text_color='k', ax=None)
    display.plot_label(label = '', location = SWHP_location, symbol='r+', 
                       text_color='k', ax=None)
    
    radar_name = radar.metadata['instrument_name']
    plt.suptitle(radar_name + ' | Elevation: ' + angle + ' Deg.| ' + time_text
                  + ' UTC', fontsize=12)
    
    return fig


def plot_DSD(filename, Dmax):
    dsd = pyd.read_parsivel(filename)

    fig = plt.figure(figsize = [5,4])
    plt.scatter(dsd.diameter['data'], dsd.fields['Nd']['data'])
    plt.yscale('log')
    plt.ylim(10E-2, 10E5)
    plt.xlim(0, Dmax)
    
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Concentration (mm^-1 m^-3)')
    
    time = datetime.datetime.fromtimestamp(dsd.time['data'][0])
    time = time + timedelta(hours=1)
    
    #time = str(datetime.datetime.fromtimestamp(dsd.time['data'][0]))
    time = str(time)
    plt.title(f'Measured DSD at {time}')
    return fig

def plot_DSD_falling_velocity(filename, Dmax):
    dsd = pyd.read_parsivel(filename)
    falling_velocity = np.ma.masked_array(dsd.velocity['data']
                        , mask= np.ma.getmaskarray(dsd.fields['Nd']['data']))
    fig = plt.figure(figsize = [5,4])
    plt.scatter(dsd.diameter['data'], falling_velocity)
    plt.ylim(-1, 10)
    plt.xlim(0, Dmax)
    
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Velocity (m/s)')
    
    time = datetime.datetime.fromtimestamp(dsd.time['data'][0])
    time = time + timedelta(hours=1)
    
    #time = str(datetime.datetime.fromtimestamp(dsd.time['data'][0]))
    time = str(time)
    plt.title(f'Falling velocity at {time}')
    return fig

def plot_DSD_falling_velocity_many(Data_Path, Dmax):
    files = os.listdir(Data_Path)
    files = sorted(files)
    Velocities = []
    Diameters = []
    for file in files:
        filename = os.path.join(Data_Path, file)
        dsd = pyd.read_parsivel(filename)
    
        falling_velocity = np.ma.masked_array(dsd.velocity['data'],
                            mask= np.ma.getmaskarray(dsd.fields['Nd']['data']))
        diameter = dsd.diameter['data']
        Velocities.append(falling_velocity)
        Diameters.append(diameter)
        
    fig = plt.figure(figsize = [5,4])
    plt.scatter(Diameters, Velocities)
    plt.ylim(-1, 14)
    plt.xlim(0, Dmax)
    
    plt.xlabel('Diameter (mm)')
    plt.ylabel('Velocity (m/s)')
    
    time = datetime.datetime.fromtimestamp(dsd.time['data'][0])
    time = time + timedelta(hours=1)
    
    #time = str(datetime.datetime.fromtimestamp(dsd.time['data'][0]))
    time = str(time)
    plt.title(f'Falling velocity at {time}')
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

def radar_ref_time_series(Data_Path, disdrometer_lat, disdrometer_lon):
    files = os.listdir(Data_Path)
    files = sorted(files)

    Times = []
    Reflectivities = []
    
    for file in files:
        filename = os.path.join(Data_Path, file)
        radar = pyart.io.read(filename)
        disdrometer_location_index = find_closest_gate(radar = radar, 
                    target_lat = disdrometer_lat, target_lon = disdrometer_lon)
        #time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
        time_start = pyart.util.datetime_from_radar(radar)
        time_text = time_start.strftime('%Y-%m-%dT%H:%M:%S')
        time = datetime.datetime.strptime(time_text, '%Y-%m-%dT%H:%M:%S')
        time = time - timedelta(hours=6)
        reflectivity = radar.fields['reflectivity']['data'][disdrometer_location_index]
        Times.append(time)
        Reflectivities.append(reflectivity)
        
    return Times, Reflectivities


def radar_ref_time_series_3_lowest_sweeps(Data_Path, disdrometer_lat, 
                                          disdrometer_lon):
    files = os.listdir(Data_Path)
    files = sorted(files)

    Times = []
    Reflectivities = []
   
    for file in files:
        filename = os.path.join(Data_Path, file)
        radar = pyart.io.read(filename)
        reflectivities = []
        print(file)
        #radar_angles = radar.fixed_angle['data']
        
        for sweep in range(6):
            disdrometer_location_index = find_closest_gate(radar = radar, 
                target_lat = disdrometer_lat, target_lon = disdrometer_lon, 
                sweep = sweep)
            reflectivity = (radar.fields['reflectivity']['data'][disdrometer_location_index])
            reflectivities.append(reflectivity)

        Reflectivities.append(reflectivities)
        time_start = pyart.util.datetime_from_radar(radar)
        time_text = time_start.strftime('%Y-%m-%dT%H:%M:%S')
        time = datetime.datetime.strptime(time_text, '%Y-%m-%dT%H:%M:%S')
        time = time - timedelta(hours=6)
        Times.append(time)
        
        
    return Times, Reflectivities

def get_subarray_around_position(arr, row, col, radius):
    """
    Extracts a sub-array around a specified position in a 2D array.

    Parameters:
        arr (list of lists): The 2D array.
        row (int): The row index of the specified position.
        col (int): The column index of the specified position.
        radius (int): The radius around the position.

    Returns:
        list of lists: The sub-array around the specified position.
    """
    subarray = []
    for i in range(row - radius, row + radius + 1):
        if 0 <= i < len(arr):
            subarray.append(arr[i][max(0, col - radius): min(len(arr[i]), col + radius + 1)])
    return subarray

def radar_ref_time_series_lowest_sweeps(Data_Path, disdrometer_lat, disdrometer_lon):
    
    files = os.listdir(Data_Path)
    files = sorted(files)

    Times = []
    Reflectivities = []
   
    for file in files:
        filename = os.path.join(Data_Path, file)
        radar = pyart.io.read(filename)
        reflectivities = []
        print(file)
        disdrometer_location_index = find_closest_gate(radar = radar, 
                target_lat = disdrometer_lat, target_lon = disdrometer_lon,)
        
        reflectivities = get_subarray_around_position(
                                    arr = radar.fields['reflectivity']['data'], 
                                    row = disdrometer_location_index[0], 
                                    col = disdrometer_location_index[1], 
                                    radius = 1)

        Reflectivities.append(reflectivities)
        time_start = pyart.util.datetime_from_radar(radar)
        time_text = time_start.strftime('%Y-%m-%dT%H:%M:%S')
        time = datetime.datetime.strptime(time_text, '%Y-%m-%dT%H:%M:%S')
        time = time - timedelta(hours=6)
        Times.append(time)
        
        
    return Times, Reflectivities


def disdrometer_ref_time_series(Data_Path):
    
    files = os.listdir(Data_Path)
    files = sorted(files)

    Times = []
    Reflectivities = []
    scatterer = Scatterer(wavelength=tmatrix_aux.wl_S, m=refractive.m_w_0C[tmatrix_aux.wl_S])
    scatterer.psd_integrator = PSDIntegrator()
    scatterer.psd_integrator.D_max = 5
    scatterer.psd_integrator.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
    scatterer.or_pdf = orientation.gaussian_pdf(10.0)
    scatterer.orient = orientation.orient_averaged_fixed
    scatterer.psd_integrator.init_scatter_table(scatterer)
    
    for file in files:
        filename = os.path.join(Data_Path, file)
        dsd = pyd.read_parsivel(filename)
        dsd.calculate_dsd_parameterization()
        bin_size = dsd.diameter['data']
        bin_counts = dsd.fields['Nd']['data'][0]
        time = datetime.datetime.fromtimestamp(dsd.time['data'][0]) 
        time = time + timedelta(hours=1) # adding an hour to match central time
        
        bin_counts  = np.ma.filled(bin_counts, fill_value=0)
        bin_counts = bin_counts[1:]
        
        scatterer.psd = BinnedPSD(bin_size, bin_counts)
        reflectivity = 10*np.log10(radar.refl(scatterer))
        
        Times.append(time)
        Reflectivities.append(reflectivity)
        #formatted_dates = [date.strftime("%M:%S") for date in Times]
        
    return Times, Reflectivities

def datetime_mean(date_list):
    # Convert datetime objects to timestamps (float)
    timestamps = [date.timestamp() for date in date_list]
    
    # Calculate the mean of timestamps
    mean_timestamp = sum(timestamps) / len(timestamps)
    
    # Convert mean timestamp back to datetime object
    mean_datetime = datetime.datetime.fromtimestamp(mean_timestamp)
    
    return mean_datetime
    
def plot_ref_time_series_radar (Data_Path_disdrometer, 
                                Data_Path_radar, disdrometer_lat, 
                                disdrometer_lon):
    
    Times_radar, Reflectivities_radar = radar_ref_time_series(Data_Path_radar, 
                                            disdrometer_lat, disdrometer_lon)
    
    Times_disdrometer, Reflectivities_disdrometer = disdrometer_ref_time_series(
                                            Data_Path_disdrometer)
    plt.figure(figsize = [5,4])
    plt.scatter(Times_radar, Reflectivities_radar, label='KFWS')
    plt.scatter(Times_disdrometer, Reflectivities_disdrometer, label='Charlie')
    hh_mm = DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(hh_mm)
    plt.xlim(Times_disdrometer[0], Times_disdrometer[len(Times_disdrometer) - 1])
    plt.legend()
    
def plot_ref_time_series_disdrother_and_radar (Data_Path_disdrometer, 
                                               Data_Path_radar, disdrometer_lat, 
                                               disdrometer_lon):
    
    Times_radar, Reflectivities_radar = radar_ref_time_series(Data_Path_radar, 
                                            disdrometer_lat, disdrometer_lon)
    
    Times_disdrometer, Reflectivities_disdrometer = disdrometer_ref_time_series(
                                            Data_Path_disdrometer)
    plt.figure(figsize = [5,4])
    plt.scatter(Times_radar, Reflectivities_radar, label='KFWS')
    plt.scatter(Times_disdrometer, Reflectivities_disdrometer, label='Charlie')
    hh_mm = DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(hh_mm)
    plt.xlim(Times_disdrometer[0], Times_disdrometer[len(Times_disdrometer) - 1])
    plt.legend()
    
def plot_ref_time_series_disdrother_and_radar_mean (Data_Path_disdrometer, 
                                                   Data_Path_radar, 
                                                   disdrometer_lat, 
                                                   disdrometer_lon):
    
    Times_radar, Reflectivities_radar = radar_ref_time_series_lowest_sweeps(
                                                        Data_Path_radar, 
                                                        disdrometer_lat, 
                                                        disdrometer_lon)
    
    Times_disdrometer, Reflectivities_disdrometer = disdrometer_ref_time_series(
                                            Data_Path_disdrometer)
    
    Reflectivities_disdrometer_mean = []
    Reflectivities_disdrometer_std = []
    Times_disdrometer_mean = []
    
    for i in range(len(Times_radar) - 1):
        reflectivity_disdrometer = []
        times_disdrometer = []
        for j in range(len(Times_disdrometer)):
            if Times_radar[i] < Times_disdrometer[j] < Times_radar[i + 1]:
                reflectivity_disdrometer.append(Reflectivities_disdrometer[j])
                times_disdrometer.append(Times_disdrometer[j])
        
        if len(times_disdrometer) == 0:
            continue
        
        time_mean = datetime_mean(times_disdrometer)
        reflectivity_mean = np.ma.mean(reflectivity_disdrometer)
        reflectivity_std = np.ma.std(reflectivity_disdrometer)
            
        Reflectivities_disdrometer_mean.append(reflectivity_mean)
        Reflectivities_disdrometer_std.append(reflectivity_std)
        Times_disdrometer_mean.append(time_mean)
        
    Reflectivities_mean = [np.ma.mean(reflectivities) for reflectivities in Reflectivities_radar]
    Reflectivities_std = [np.ma.std(reflectivities) for reflectivities in Reflectivities_radar]
    
    plt.figure(figsize = [5,4])
    plt.errorbar(Times_radar, Reflectivities_mean, yerr=Reflectivities_std, 
                 fmt='o', label='KFWS')
    plt.errorbar(Times_disdrometer_mean, Reflectivities_disdrometer_mean, 
                 yerr=Reflectivities_disdrometer_std, fmt='o', label='Charlie')

    hh_mm = DateFormatter('%M')
    plt.gca().xaxis.set_major_formatter(hh_mm)
    plt.xlim(Times_radar[28], Times_radar[51])
    plt.ylim(0, 50)
    plt.legend()
    plt.xlabel('Time (mm)')
    plt.ylabel('Reflectivity (dBZ)')
    
def plot_ref_time_series_disdroter_and_radar_mean (Times_radar, 
                                                   Reflectivities_radar, 
                                                   Times_disdrometer, 
                                                   Reflectivities_disdrometer, 
                                                   disdrometer_lat, 
                                                   disdrometer_lon):    
    Reflectivities_disdrometer_mean = []
    Reflectivities_disdrometer_std = []
    Times_disdrometer_mean = []
    
    for i in range(len(Times_radar) - 1):
        reflectivity_disdrometer = []
        times_disdrometer = []
        for j in range(len(Times_disdrometer)):
            if Times_radar[i] < Times_disdrometer[j] < Times_radar[i + 1]:
                reflectivity_disdrometer.append(Reflectivities_disdrometer[j])
                times_disdrometer.append(Times_disdrometer[j])
        
        if len(times_disdrometer) == 0:
            continue
        
        time_mean = datetime_mean(times_disdrometer)
        reflectivity_mean = np.ma.mean(reflectivity_disdrometer)
        reflectivity_std = np.ma.std(reflectivity_disdrometer)
            
        Reflectivities_disdrometer_mean.append(reflectivity_mean)
        Reflectivities_disdrometer_std.append(reflectivity_std)
        Times_disdrometer_mean.append(time_mean)
        
    Reflectivities_mean = [np.ma.mean(reflectivities) for reflectivities in Reflectivities_radar]
    Reflectivities_std = [np.ma.std(reflectivities) for reflectivities in Reflectivities_radar]
    
    plt.figure(figsize = [5,4])
    plt.errorbar(Times_radar, Reflectivities_mean, yerr=Reflectivities_std, 
                 fmt='o', label='KFWS')
    plt.errorbar(Times_disdrometer_mean, Reflectivities_disdrometer_mean, 
                 yerr=Reflectivities_disdrometer_std, fmt='o', label='Disdrometer')

    hh_mm = DateFormatter('%M')
    plt.gca().xaxis.set_major_formatter(hh_mm)
    plt.xlim(Times_radar[28], Times_radar[51])
    plt.ylim(0, 50)
    plt.legend()
    plt.xlabel('Time (mm)')
    plt.ylabel('Reflectivity (dBZ)')
    
def scatter_ref_disdroter_and_radar (Times_radar, 
                                    Reflectivities_radar, 
                                    Times_disdrometer, 
                                    Reflectivities_disdrometer, 
                                    disdrometer_lat, 
                                    disdrometer_lon): 
    
    Reflectivities_disdrometer_mean = []
    
    for i in range(len(Times_radar) - 1):
        reflectivity_disdrometer = []
        times_disdrometer = []
        for j in range(len(Times_disdrometer)):
            if Times_radar[i] < Times_disdrometer[j] < Times_radar[i + 1]:
                reflectivity_disdrometer.append(Reflectivities_disdrometer[j])
                times_disdrometer.append(Times_disdrometer[j])
        
        if len(times_disdrometer) == 0:
            continue

        reflectivity_mean = np.ma.mean(reflectivity_disdrometer)
        Reflectivities_disdrometer_mean.append(reflectivity_mean)
        
        
    Reflectivities_mean = [np.ma.mean(reflectivities) for reflectivities in Reflectivities_radar]
    
    disdrometer = np.array(Reflectivities_disdrometer_mean)
    disdrometer = np.ma.masked_invalid(disdrometer)
    radar = Reflectivities_mean[:-1]
    bias = np.ma.mean(radar - disdrometer)
    bias = round(bias, 2)
    correlation_matrix = np.ma.corrcoef(radar, disdrometer)
    correlation_coefficient = correlation_matrix[0, 1]
    correlation_coefficient = round(correlation_coefficient, 2)
    rmse = np.sqrt(np.ma.mean((radar - disdrometer) ** 2))
    rmse = round(rmse, 2)
    x = np.arange(-100, 100, 1)
    
    fig = plt.figure(figsize = [5,4])
    plt.scatter(Reflectivities_mean[:-1],Reflectivities_disdrometer_mean)
    plt.plot(x,x)
    
    plt.ylim(0, 50)
    plt.xlim(0, 50)
    plt.xlabel('Radar reflectivity (dBZ)')
    plt.ylabel('Disdrometer reflectivity (dBZ)')
    plt.title(f"Bias: {bias}   Corr: {correlation_coefficient}   RMSE: {rmse}")
    
    return fig


def main():
    
    
    
    
    Data_Path_radar = '/net/k2/storage/people/idariash/home/CSU/DFW/radar/20240108'
    
    charlie_lat = 32.91365
    charlie_lon = -97.05752
    
    swhp_lat = 32.87729
    swhp_lon = -97.04658
    
    sehp_lat = 32.87544
    sehp_lon = -97.03275
    
    Data_Path_disdrometer = '/net/k2/storage/people/idariash/home/CSU/DFW/data/Charlie/20240108'
    '/net/k2/storage/people/idariash/home/CSU/DFW/data/SWHP/20240108'
    '/net/k2/storage/people/idariash/home/CSU/DFW/data/SEHP/20240108'
    
    Times_radar, Reflectivities_radar = radar_ref_time_series_lowest_sweeps(
                                            Data_Path = Data_Path_radar,
                                            disdrometer_lat = charlie_lat, 
                                            disdrometer_lon = charlie_lon)
    
    Times_disdrometer, Reflectivities_disdrometer = disdrometer_ref_time_series(
                                            Data_Path_disdrometer)
    
    
    plot_ref_time_series_disdroter_and_radar_mean (Times_radar, 
                                                   Reflectivities_radar, 
                                                   Times_disdrometer, 
                                                   Reflectivities_disdrometer, 
                                                   disdrometer_lat = charlie_lat, 
                                                   disdrometer_lon = charlie_lon)
    
    Reflectivities_disdrometer_mean = []
    Reflectivities_disdrometer_std = []
    Times_disdrometer_mean = []
    
    for i in range(len(Times_radar) - 1):
        reflectivity_disdrometer = []
        times_disdrometer = []
        for j in range(len(Times_disdrometer)):
            if Times_radar[i] < Times_disdrometer[j] < Times_radar[i + 1]:
                reflectivity_disdrometer.append(Reflectivities_disdrometer[j])
                times_disdrometer.append(Times_disdrometer[j])
        
        if len(times_disdrometer) == 0:
            continue
        
        time_mean = datetime_mean(times_disdrometer)
        reflectivity_mean = np.ma.mean(reflectivity_disdrometer)
        reflectivity_std = np.ma.std(reflectivity_disdrometer)
            
        
        Reflectivities_disdrometer_mean.append(reflectivity_mean)
        Reflectivities_disdrometer_std.append(reflectivity_std)
        Times_disdrometer_mean.append(time_mean)
        
    Reflectivities_midle = [reflectivities[1][1] for reflectivities in Reflectivities_radar]
    Reflectivities_mean = [np.ma.mean(reflectivities) for reflectivities in Reflectivities_radar]
    Reflectivities_std = [np.ma.std(reflectivities) for reflectivities in Reflectivities_radar]
    plt.figure(figsize = [5,4])
    #plt.scatter(Times_radar, Reflectivities_midle,label='lowest')
    plt.errorbar(Times_radar, Reflectivities_mean, yerr=Reflectivities_std, 
                 fmt='o', label='KFWS')
    plt.errorbar(Times_disdrometer_mean, Reflectivities_disdrometer_mean, 
                 yerr=Reflectivities_disdrometer_std, fmt='o', label='Charlie')
    #plt.scatter(Times_radar, Reflectivities_mean,label='mean')
    #plt.scatter(Times_radar, Reflectivities_std,label='std')
    #plt.scatter(Times_disdrometer, Reflectivities_disdrometer, label='Charlie')
    hh_mm = DateFormatter('%M')
    plt.gca().xaxis.set_major_formatter(hh_mm)
    plt.xlim(Times_radar[28], Times_radar[51])
    plt.ylim(0, 50)
    plt.legend()
    plt.show()
    
    
    plt.figure(figsize = [5,4])
    plt.scatter(Reflectivities_mean[:-1],Reflectivities_disdrometer_mean)
    plt.show()
    
    disdrometer = np.array(Reflectivities_disdrometer_mean)
    disdrometer = np.ma.masked_invalid(disdrometer)
    radar = Reflectivities_mean[:-1]
    a = np.ma.mean(disdrometer - radar)
    
    a = 1
    
    
    #--------------------------
    
    # Times_radar, Reflectivities_radar = radar_ref_time_series_3_lowest_sweeps(
    #                                         Data_Path = Data_Path_radar,
    #                                         disdrometer_lat = charlie_lat, 
    #                                         disdrometer_lon = charlie_lon)
    # Reflectivities_lowest = [reflectivities[0] for reflectivities in Reflectivities_radar]
    # Reflectivities_mean = [np.ma.mean(reflectivities[0:2]) for reflectivities in Reflectivities_radar]
    # Reflectivities_std = [np.ma.std(reflectivities[0:2]) for reflectivities in Reflectivities_radar]
    # plt.figure(figsize = [5,4])
    # plt.scatter(Times_radar, Reflectivities_lowest,label='lowest')
    # plt.scatter(Times_radar, Reflectivities_mean,label='meain')
    # plt.scatter(Times_radar, Reflectivities_std,label='std')
    # hh_mm = DateFormatter('%H:%M')
    # plt.gca().xaxis.set_major_formatter(hh_mm)
    # plt.xlim(Times_radar[0], Times_radar[-1])
    # plt.legend()
    # plt.show()
    # a = 1
    # a = [1, 2 , 3 , 4]
    # a[0:3]
    
    
    # plot_ref_time_series_disdrother_and_radar (Data_Path_disdrometer, 
    #                                            Data_Path_radar, 
    #                                            disdrometer_lat = charlie_lat, 
    #                                            disdrometer_lon = charlie_lon)
    # plt.show()
    
    
    # Times, Reflectivities = radar_ref_time_series(Data_Path, 
    #     disdrometer_lat = charlie_lat, disdrometer_lon = charlie_lon) 
    # plt.figure(figsize = [5,4])
    # plt.scatter(Times, Reflectivities, label='KFWS')
    
    
    # Data_Path = '/net/k2/storage/people/idariash/home/CSU/DFW/data/Charlie/20240108_test'
    # Times_disdrometer, Reflectivities_disdrometer = disdrometer_ref_time_series(Data_Path)
    
    # plt.scatter(Times_disdrometer, Reflectivities_disdrometer, label='Charlie')
    # hh_mm = DateFormatter('%H:%M')
    # plt.gca().xaxis.set_major_formatter(hh_mm)
    # plt.xlim(Times_disdrometer[0], Times_disdrometer[len(Times_disdrometer) - 1])
    # plt.legend()
    # plt.show()
    
    # a =1
    
    # hh_mm = DateFormatter('%H:%M')
    # plt.gca().xaxis.set_major_formatter(hh_mm)
    # plt.xlim(Times[0], Times[len(Times) - 1])
    # plt.show()
    
    # filename = '/net/k2/storage/people/idariash/home/CSU/DFW/radar/20240108/KFWS20240108_173658_V06.ar2v'
    # # '/net/k2/storage/people/idariash/home/CSU/DFW/radar/KFWS20240108_173658_V06'
    # radar = pyart.io.read(filename)
    # time = pyart.util.datetime_from_radar(radar)
    # b = time.to_datetimeindex()
    
    #-------------------------
    # Data_Path = '/net/k2/storage/people/idariash/home/CSU/DFW/data/Charlie/20240108_test'
    # Dmax = 12
    # plot_DSD_falling_velocity_many(Data_Path, Dmax)
    # plt.show()
    # a =1
    #-----------
    # charlie_lat = 32.91365
    # charlie_lon = -97.05752
    # plot_nexrad_ppi_ref_near_DFW_airport(filename, sweep = 0)
    
    # x, y , z, d = find_closest_gate(radar = radar, target_lat = charlie_lat,
    #                                 target_lon = charlie_lon)
    # a  = 1
    
    
if __name__ == "__main__":
    main()



    

