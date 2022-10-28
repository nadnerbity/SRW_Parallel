#!/usr/local/python
#

#############################################################################
# This file is simulating propagation of Gaussian photon beams
# v 0.01 First Draft
# Based on SRWLIB_Example01.py
#############################################################################



from __future__ import print_function #Python 2.7 compatibility
import sys

sys.path.insert(0, '/home/brendan/SRW/env/work/srw_python')


# set the backend for matplot lib
import matplotlib
matplotlib.use("TkAgg")


from srwlib import *
from Split_SRW_Wavefront_Mesh import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()


# x = Student("Mike", "Olsen")

# Define the laser beam parameters

photon_lam 	    = 800.0e-9 # Photon energy in eV
sig_x           = 100.0e-6
sig_y           = sig_x

# Define the parameters for the wavefront object that will hold the gaussian
# beam


# Wavefront parameters
# Wavefront mesh parameters
Nx 			    = 2**7
Ny 			    = Nx
zSrCalc 	    = 0.0 # Distance from sim start to calc SR. [m]
xMiddle		    = 0.0 # middle of window in X to calc SR [m]
xWidth 		    = 10*sig_x # width of x window. [m]
yMiddle 	    = 0.0 # middle of window in Y to calc SR [m]
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
wfr1 = subSRWLWfr() #For spectrum vs photon energy
wfr1.allocate(1, Nx, Ny) #Numbers of points vs Photon Energy, Horizontal and Vertical Positions
wfr1.mesh.zStart    = zSrCalc #Longitudinal Position [m] from Center of Straight
# Section at which SR has to be calculated
wfr1.mesh.eStart 	= photon_e #Initial Photon Energy [eV]
wfr1.mesh.eFin 		= photon_e #Final Photon Energy [eV]
wfr1.mesh.xStart 	= xMiddle - xWidth/2.0 #Initial Horizontal Position [m]
wfr1.mesh.xFin 		= xMiddle + xWidth/2.0 #Final Horizontal Position [m]
wfr1.mesh.yStart 	= yMiddle - yWidth/2.0 #Initial Vertical Position [m]
wfr1.mesh.yFin 		= yMiddle + yWidth/2.0 #Final Vertical Position [m]

# Propagate the gaussian beam
srwl.CalcElecFieldGaussian(wfr1, g_Beam, [0])

# Extract the photon beam intensity
arI1 = array('f', [0]*wfr1.mesh.nx*wfr1.mesh.ny)
srwl.CalcIntFromElecField(arI1, wfr1, 6, 0, 3, wfr1.mesh.eStart, 0, 0)
B = np.reshape(arI1, [wfr1.mesh.nx, wfr1.mesh.ny])

plt.figure(1, facecolor='w')
plt.imshow(B)