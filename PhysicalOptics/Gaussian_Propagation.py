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
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()


# Define the laser beam parameters

photon_lam 	    = 800.0e-9 # Photon energy in eV
Zr              = 5.0 # Rayleigh range in meters
# The extra divide by sqrt(2) converts 'waist' to rms size
sig_x           = np.sqrt(photon_lam * Zr / np.pi) / np.sqrt(2)
sig_y           = sig_x

# Define the parameters for the wavefront object that will hold the gaussian
# beam
Nx 			    = 2**11
Ny 			    = Nx
zSrCalc 	    = 0.0 # Distance from sim start to calc SR. [m]
xMiddle		    = 0.0*sig_x # middle of window in X to calc SR [m]
xWidth 		    = 10*sig_x*sqrt(2) # width of x window. [m]
yMiddle 	    = 0.0*sig_y # middle of window in Y to calc SR [m]
yWidth 		    = xWidth # width of y window. [m]

# SR integration flags.
use_termin 		= 1 #1 #Use "terminating terms" (i.e. asymptotic expansions
# at  zStartInteg and zEndInteg) or not (1 or 0 respectively)
# Precision for SR integration
srPrec 		    = 5.0e-6

sampFactNxNyForProp = 1

################################################################################
################### DERIVED SIMULATION PARAMETERS ##############################
################################################################################

#"Beamline" - Container of optical elements (together with their corresponding wavefront propagation parameters / instructions)
photon_e = 4.135e-15 * 299792458.0 / photon_lam # convert wavelength to eV

# Build the Gaussian photon beam
g_Beam              = SRWLGsnBm()
g_Beam.sigX         = sig_x
g_Beam.sigY         = sig_y
g_Beam.z            = zSrCalc
g_Beam.polar        = 1
g_Beam.avgPhotEn    = photon_e

# Build the wavefront class instance to hold the gaussian beam electric fields
wfr1 = SRWLWfr() #For spectrum vs photon energy
wfr1.allocate(1, Nx, Ny) #Numbers of points vs Photon Energy, Horizontal and Vertical Positions
wfr1.mesh.zStart    = zSrCalc #Longitudinal Position [m] from Center of Straight
# Section at which SR has to be calculated
wfr1.mesh.eStart 	= photon_e #Initial Photon Energy [eV]
wfr1.mesh.eFin 		= photon_e #Final Photon Energy [eV]
wfr1.mesh.xStart 	= xMiddle - xWidth/2.0 #Initial Horizontal Position [m]
wfr1.mesh.xFin 		= xMiddle + xWidth/2.0 #Final Horizontal Position [m]
wfr1.mesh.yStart 	= yMiddle - yWidth/2.0 #Initial Vertical Position [m]
wfr1.mesh.yFin 		= yMiddle + yWidth/2.0 #Final Vertical Position [m]


# Generate the gaussian beam
srwl.CalcElecFieldGaussian(wfr1, g_Beam, [0])

I = convert_srw_linear_fields_to_matrix_fields(wfr1)

plot_SRW_intensity(wfr1, fig_num=10)


# Simulation parameters
focal_length = 0.25
prop_distance = 0.25  # in meters

print('Nx is ' + str(Nx))
print('Ny is ' + str(Ny))

# Apply a lens phase using Brendan's FO ----------------------------------------
# Build the physical spaces for the FO

t0 = time.time()
xgv = np.linspace(wfr1.mesh.xStart, wfr1.mesh.xFin, wfr1.mesh.nx)
ygv = np.linspace(wfr1.mesh.yStart, wfr1.mesh.yFin, wfr1.mesh.ny)
[XX, YY] = np.meshgrid(xgv, ygv)
kappa_0 = 2*np.pi / photon_lam

EM = convert_srw_linear_fields_to_matrix_fields(wfr1)
# Convert the real + complex parts into complex fields
Ex = EM[:,:,0] + 1j*EM[:,:,1] # The Ex part

# Apply the lens phase
ZZ1 = FO.FO_lens(XX, YY, Ex, kappa_0, focal_length)
# Propagate using original SM method
ZZ = FO.FO_TwoD_exact_SM(xgv, ygv, ZZ1, prop_distance, photon_lam)
# Propagate using new CZT method
MM = 2*Nx
x_freq, ZZ2 = FO.FO_TwoD_exact_SM_CZT(xgv, ZZ1, 1.0*prop_distance, photon_lam,
                                      M = MM,
                                      x_out_1 = xgv[0],
                                      x_out_2 = xgv[-1])

time_str = "Brendan FO Run time: %.2f seconds." % (time.time() - t0)
print(time_str)


# Apply a lens phase using SRW -------------------------------------------------

t0 = time.time()

wfr2 = deepcopy(wfr1)
propagParLens =  [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
lens = SRWLOptL(focal_length, focal_length)
drift = SRWLOptD(prop_distance)
a_drift = SRWLOptC(
    [lens, drift],
    [propagParLens, propagParLens])
srwl.PropagElecField(wfr2, a_drift)

EM = convert_srw_linear_fields_to_matrix_fields(wfr2)
# Convert the real + complex parts into complex fields
Exx = EM[:,:,0] + 1j*EM[:,:,1] # The Ex part

time_str = "SRW FO Run time: %.2f seconds." % (time.time() - t0)
print(time_str)

# Plot things ------------------------------------------------------------------

to_plot_1 = np.abs(ZZ[:, Ny//2]) / np.max(np.abs(ZZ[:, Ny//2]))
to_plot_2 = np.abs(Exx[:, Ny//2]) / np.max(np.abs(Exx[:, Ny//2]))
to_plot_3 = np.abs(ZZ2[:, MM//2]) / np.max(np.abs(ZZ2[MM//2, :]))

plt.close(4)
plt.figure(4)
plt.plot(xgv, to_plot_1, 'rx')
plt.plot(xgv, to_plot_2, 'k.')
plt.scatter(x_freq, to_plot_3, s=40,
         facecolors='none', edgecolors='b')
plt.legend(['Brendan FO', 'SRW FO', 'Brendan CZT'])
plt.xlabel('Position [m]', fontsize=20)
plt.ylabel('Amplitude [arb]', fontsize=20)
# plt.xlim([-0.0004, 0.0004])
# plt.ylim([0.9, 1.05])

plot_SRW_intensity(wfr2, fig_num=11)

# plt.close(5)
# plt.figure(5)
# plt.plot(np.abs(ZZ[:, Ny//2]), 'rx')
# plt.plot(np.abs(Exx[:, Ny//2]), 'k.')
# plt.plot(to_plot, 'bo')
# plt.legend(['Brendan FO', 'SRW FO', 'Brendan CZT'])
# plt.xlabel('Position [m]', fontsize=20)
# plt.ylabel('Amplitude [arb]', fontsize=20)
# plt.xlim([250, 261])
# plt.ylim([1.38e8, 1.48e8])

# plt.close(45)
# plt.figure(45)
# plt.plot(np.abs(Ex[:, Ny//2]), 'rx')

