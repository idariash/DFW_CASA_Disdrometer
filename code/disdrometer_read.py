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

filename = '/net/k2/storage/projects/DistroDFW/DFW/UMASS/SEHP/20240108/DIS_20240108_173401.txt'
'/net/k2/storage/projects/DistroDFW/DFW/charlie/2024/01/08/DIS_20240108_173401.txt'
'/net/k2/storage/projects/DistroDFW/DFW/charlie/2024/01/08/DIS_20240108_173401.txt'
'/net/k2/storage/projects/DistroDFW/DFW/charlie/2024/01/08/DIS_20240108_173501.txt'
'/net/k2/storage/projects/DistroDFW/DFW/charlie/2024/01/08/DIS_20240108_173602.txt'
dsd = pyd.read_parsivel(filename)
dsd.calculate_dsd_parameterization()
N0 = dsd.fields['N0']['data'][0]
mu = dsd.fields['mu']['data'][0]
D0 = dsd.fields['D0']['data'][0] 

D =  np.arange(0.2, 3, 0.1)
gamma = 100000000/2*N0*np.power(D,mu)*np.exp(-(3.67 + mu)*D/D0)

dsd.calculate_dsd_parameterization()
fig = plt.figure(figsize = [5,4])
plt.scatter(dsd.diameter['data'], dsd.fields['Nd']['data'])
plt.plot(D, gamma, label='Gamma aproximation')
plt.yscale('log')
plt.ylim(10E-2, 10E5)
plt.xlim(0, 3)

plt.xlabel('Diameter (mm)')
plt.ylabel('Concentration (mm^-1 m^-3)')

time = datetime.datetime.fromtimestamp(dsd.time['data'][0])
time = time + timedelta(hours=1)

#time = str(datetime.datetime.fromtimestamp(dsd.time['data'][0]))
time = str(time)
plt.title(f'Measured DSD at {time}')

scatterer = Scatterer(wavelength=tmatrix_aux.wl_S, m=refractive.m_w_0C[tmatrix_aux.wl_S])
scatterer.psd_integrator = PSDIntegrator()
scatterer.psd_integrator.D_max = 15
scatterer.psd_integrator.geometries = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
scatterer.or_pdf = orientation.gaussian_pdf(10.0)
scatterer.orient = orientation.orient_averaged_fixed
scatterer.psd_integrator.init_scatter_table(scatterer)

bin_size = dsd.diameter['data']
bin_counts = dsd.fields['Nd']['data'][0]
bin_counts  = np.ma.filled(bin_counts, fill_value=0)
bin_counts = bin_counts[1:]

scatterer.psd = BinnedPSD(bin_size, bin_counts)
reflectivity_DSD = 10*np.log10(radar.refl(scatterer))



plt.show()