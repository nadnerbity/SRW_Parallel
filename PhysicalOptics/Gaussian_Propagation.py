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
Nx 			    = 2**10
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

EM = convert_srw_linear_fields_to_matrix_fields(wfr1)