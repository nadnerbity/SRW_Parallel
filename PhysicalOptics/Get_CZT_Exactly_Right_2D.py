#!/usr/local/python
#

# This file is for testing the I/O for physical optics using RW produced wave
#  fronts.



from __future__ import print_function #Python 2.7 compatibility
import sys

#sys.path.insert(0, '/Users/brendan/Documents/FACET/SRW/SRW-light/env/work/srw_python') # To find the SRW python libraries.
sys.path.insert(0, '/scratch/brendan/SRW/SRW/env/work/srw_python') # To find
# the SRW python libraries.
sys.path.insert(0, '/scratch/brendan/SRW_Parallel/SRW_Parallel')


# set the backend for matplot lib
import matplotlib
# matplotlib.use("TkAgg")


from srwlib import *
from SRW_Split_Wavefront_Mesh import *
from SRW_Utilities import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import FourierOptics as FO
import tracemalloc
from scipy.signal import CZT
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()



def Gaussian(x, mu, sig):
    XX, YY = np.meshgrid(x, x)
    return np.exp(-((XX-mu)**2+(YY-mu)**2)/2/sig**2)

# Generate data to test with
Nx = 2**7
xMin = -10
xMax = 10
X = np.linspace(xMin, xMax, Nx)
ZZ = Gaussian(X, 0, 1.5)

# CZT in 2D
# FFT the data
x_range = X[-1] - X[0]
x_start = X[0]
x_stop = X[-1]

A = np.exp(2j * np.pi * (x_start) / x_range)
W = np.exp(-2j * np.pi * (x_stop - x_start) / x_range / Nx)
temp = CZT(n = X.size, m = Nx, w = W, a = A)
BB = temp(ZZ, axis=0) * (X[1] - X[0])
BB = temp(BB, axis=1) * (X[1] - X[0])
x = np.linspace(-Nx/2, Nx/2, Nx) / (X[1] - X[0]) / Nx

# IFFT the data
M = 2*Nx
x_range = X[-1] - X[0]
x_start = X[0] * 0.5
x_stop = X[-1] * 0.5

A = np.exp(2j * np.pi * (x_start - x_range/2) / x_range +
           (1.0) * 1j * np.pi/ Nx) # This Nx is correct! Do not change it!
W = np.exp(-2j * np.pi * (x_stop - x_start) / x_range / (M-1))
temp = CZT(n = x.size, m = M, w = W, a = A)
BB = temp(BB, axis=0)
BB = temp(BB, axis=1)
X2 = np.linspace(x_start, x_stop, M)


#
to_plot = np.abs(BB[:, Nx//2]) / np.max(np.abs(BB[:, Nx//2]))

plt.close(1)
plt.figure(1)
plt.plot(X2, to_plot, 'bo')
plt.plot(X, ZZ[:, Nx//2], 'rx')
plt.legend(['After FFT+iFFT','Input'])
# plt.xlim([-1.0, 1.0])
# plt.ylim([0.9, 1.05])

plt.close(2)
plt.figure(2)
plt.imshow(abs(BB))

# Compare to the function you wrote based on the above testing.
# X3, CC = FO.FO_TwoD_exact_SM_CZT(X, ZZ, 0.0, 0.0, M, -7.0, 2.5)

# to_plot_1 = np.abs(BB[:, M//2]) / np.max(np.abs(BB[:, M//2]))
# to_plot_2 = np.abs(CC[:, M//2]) / np.max(np.abs(CC[:, M//2]))
#
# plt.close(1)
# plt.figure(1)
# plt.plot(X2, to_plot_1, 'bo')
# plt.plot(X3, to_plot_2, 'rx')
# plt.legend(['Local CZT', 'Function CZT'])

