#!/usr/local/python
#

# This file is for testing the I/O of the MPI split and simulate code. It
# doesn't have formal tests. It just run the MPI system serially and compares
#  it to a simulation run the normal way. Then it plots the results on two
# plots so you can visually confirm the code moving around the wavefronts is
# (probably) working.



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


# Define the laser beam parameters

photon_lam 	    = 800.0e-9 # Photon energy in eV
sig_x           = 0.8e-3
sig_y           = sig_x

# Define the parameters for the wavefront object that will hold the gaussian
# beam
Nx 			    = 2**13
Ny 			    = Nx
zSrCalc 	    = 0.0 # Distance from sim start to calc SR. [m]
xMiddle		    = 0.0*sig_x # middle of window in X to calc SR [m]
xWidth 		    = 20*sig_x*2**(1/2) # width of x window. [m]
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

wfr2 = deepcopy(wfr1)

focal_length = 0.105  # in meters
prop_distance = 0.105  # in meters
reres = 1.0
#                [0][1][2] [3][4] [5] [6] [7] [8]
propagParApp  =  [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParLens =  [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDrift = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

aper = SRWLOptA(_shape='c', _ap_or_ob='a', _Dx=0.1)
lens = SRWLOptL(focal_length, focal_length)
drift = SRWLOptD(prop_distance)

a_drift = SRWLOptC(
    [aper, lens, drift],
    [propagParApp, propagParLens, propagParDrift])
srwl.PropagElecField(wfr2, a_drift)

print("")
print('propagParApp: ' + str(propagParApp))
print('propagParLens: ' + str(propagParLens))
print('propagParDrift: ' + str(propagParDrift))
print('Input Nx, Ny = ', wfr1.mesh.nx, ", ", wfr1.mesh.ny)
print('Ouput Nx, Ny = ', wfr2.mesh.nx, ", ", wfr2.mesh.ny)
print('Input xWdith = yWidth =  ', wfr1.mesh.xFin - wfr1.mesh.xStart, "," ,
      wfr1.mesh.yFin - wfr1.mesh.yStart,
      ' [m]')
print('Output xWdith = yWidth =  ', wfr2.mesh.xFin - wfr2.mesh.xStart, "," ,
      wfr2.mesh.yFin - wfr2.mesh.yStart,
      ' [m]')

plot_two_SRW_intensity(wfr1, wfr2, title1="Before PO", title2="After PO",
                           fig_num=20)

plt.figure(10)
plt.plot()

# wfr3 = deepcopy(wfr2)
# resize_params = [0, 1, 2, 1, 2, 0.5, 0.5]
# srwl.ResizeElecField(wfr3, 'c', resize_params)
#
# plot_two_SRW_intensity(wfr2, wfr3, title1="Before Resize", title2="After "
#                                                                   "Resize",
#                            fig_num=21)


# # Extract the photon beam intensity to compare the MPI method to the method
# # that does one big mesh. It should be embarassingly parallelizable.
# arI1 = array('f', [0]*wfr1.mesh.nx*wfr1.mesh.ny)
# srwl.CalcIntFromElecField(arI1, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0)
# B = np.reshape(arI1, [wfr1.mesh.nx, wfr1.mesh.ny])
#
# arI1 = array('f', [0]*wfr2.mesh.nx*wfr2.mesh.ny)
# srwl.CalcIntFromElecField(arI1, wfr2, 6, 0, 3, wfr2.mesh.eStart, 0, 0)
# C = np.reshape(arI1, [wfr2.mesh.nx, wfr2.mesh.ny])
#
# plt.figure(1, facecolor='w')
# plt.subplot(1,2,1)
# plt.imshow(B)
# plt.title('One simulation')
#
# plt.subplot(1,2,2)
# plt.imshow(C)
# plt.title('After Physical Optics')
