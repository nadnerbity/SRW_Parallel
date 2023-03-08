#!/usr/local/python
#

#############################################################################
# This file is simulating the second and third bend magnets in the FACET BC11
# chicane.



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
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()

with open('./Robbie_After_SR_Calc.pkl', 'rb') as file:
    wfr_r = pickle.load(file)

with open('F2_BC11_B2_and_B3_Nx_4096.pkl', 'rb') as file:
    wfr_b = pickle.load(file)


plot_two_SRW_intensity(wfr_r, wfr_b, 'Robbie', 'Brendan')

EM_r = convert_srw_linear_fields_to_matrix_fields(wfr_r)
# Convert the real + complex parts into complex fields
Ex_r = EM_r[:,:,0] + 1j*EM_r[:,:,1] # The Ex part

EM_b = convert_srw_linear_fields_to_matrix_fields(wfr_b)
# Convert the real + complex parts into complex fields
Ex_b = EM_b[:,:,0] + 1j*EM_b[:,:,1] # The Ex part
Ex_b = np.fliplr(Ex_b)

x_r = np.linspace(wfr_r.mesh.xStart, wfr_r.mesh.xFin, wfr_r.mesh.ny)
to_plot_r = np.abs(Ex_r[:, Ex_r.shape[1]//4])
to_plot_r = to_plot_r / np.max(to_plot_r)

x_b = np.linspace(wfr_b.mesh.xStart, wfr_b.mesh.xFin, wfr_b.mesh.ny)

to_plot_b = np.abs(Ex_b[:, Ex_b.shape[1]//4])
to_plot_b = to_plot_b / np.max(to_plot_b)

plt.close(213)
plt.figure(213)
plt.plot(x_r, to_plot_r, 'rx')
plt.plot(x_b, to_plot_b, 'bo')
plt.xlabel('Position [m]', fontsize=18)
plt.ylabel('Intensity [arb]', fontsize=18)
plt.tight_layout()


x_r = np.linspace(wfr_r.mesh.xStart, wfr_r.mesh.xFin, wfr_r.mesh.ny)
to_plot_r = np.angle(Ex_r[:, Ex_r.shape[1]//4])
x_b = np.linspace(wfr_b.mesh.xStart, wfr_b.mesh.xFin, wfr_b.mesh.ny)
to_plot_b = np.angle(Ex_b[:, Ex_b.shape[1]//4])

# x_r = np.linspace(wfr_r.mesh.xStart, wfr_r.mesh.xFin, wfr_r.mesh.nx)
# to_plot_r = np.angle(Ex_r[Ex_r.shape[0]//2, :])
# x_b = np.linspace(wfr_b.mesh.xStart, wfr_b.mesh.xFin, wfr_b.mesh.nx)
# to_plot_b = np.angle(Ex_b[Ex_b.shape[1]//2, :])


plt.close(214)
plt.figure(214)
plt.plot(x_r, to_plot_r, 'rx')
plt.plot(x_b, to_plot_b, 'bo')
plt.xlabel('Position [m]', fontsize=18)
plt.ylabel('Angle [arb]', fontsize=18)
plt.tight_layout()
plt.legend(['Robbie', 'Brendan'])
plt.xlim([-0.005, 0.005])
