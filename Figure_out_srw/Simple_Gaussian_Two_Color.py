#!/usr/local/python
#

"""
Trying to figure out how SRW splits up the arEx and arEy mesh for two
different colors.
"""



from __future__ import print_function #Python 2.7 compatibility
import sys
import os
file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, file_dir)


# set the backend for matplot lib
import matplotlib
matplotlib.use("TkAgg")


from srwlib import *
from SRW_Utilities import *
from SRW_Split_Wavefront_Mesh import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
# Some matplotlib settings.
plt.close('all')
plt.ion()


# Define the laser beam parameters

photon_lam 	    = 800.0e-9 # Photon energy in eV
sig_x           = 100.0e-6
sig_y           = sig_x

# Define the parameters for the wavefront object that will hold the gaussian
# beam
Nx 			    = 2**1
Ny 			    = Nx
zSrCalc 	    = 0.0 # Distance from sim start to calc SR. [m]
xMiddle		    = 0.0*sig_x # middle of window in X to calc SR [m]
xWidth 		    = 10*sig_x # width of x window. [m]
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
wfr1.allocate(1, Nx, Ny) #Numbers of points vs Photon Energy, Horizontal and
# Vertical Positions
wfr1.mesh.zStart    = zSrCalc #Longitudinal Position [m] from Center of Straight
# Section at which SR has to be calculated
wfr1.mesh.eStart 	= photon_e #Initial Photon Energy [eV]
wfr1.mesh.eFin 		= photon_e #Final Photon Energy [eV]
wfr1.mesh.xStart 	= xMiddle - xWidth/2.0 #Initial Horizontal Position [m]
wfr1.mesh.xFin 		= xMiddle + xWidth/2.0 #Final Horizontal Position [m]
wfr1.mesh.yStart 	= yMiddle - yWidth/2.0 #Initial Vertical Position [m]
wfr1.mesh.yFin 		= yMiddle + yWidth/2.0 #Final Vertical Position [m]

# Create a copy of the wavefront and update the photon energy
wfr3 = deepcopy(wfr1)
wfr3.mesh.eStart         = 0.5*photon_e
wfr3.mesh.eFin           = 0.5*photon_e

# Build a wavefront for 2 colors the SRW way
wfr2 = SRWLWfr() #For spectrum vs photon energy
wfr2.allocate(2, Nx, Ny) #Numbers of points vs Photon Energy, Horizontal and
# Vertical Positions
wfr2.mesh.zStart    = zSrCalc #Longitudinal Position [m] from Center of Straight
# Section at which SR has to be calculated
wfr2.mesh.eStart 	= 0.5*photon_e #Initial Photon Energy [eV]
wfr2.mesh.eFin 		= photon_e #Final Photon Energy [eV]
wfr2.mesh.xStart 	= xMiddle - xWidth/2.0 #Initial Horizontal Position [m]
wfr2.mesh.xFin 		= xMiddle + xWidth/2.0 #Final Horizontal Position [m]
wfr2.mesh.yStart 	= yMiddle - yWidth/2.0 #Initial Vertical Position [m]
wfr2.mesh.yFin 		= yMiddle + yWidth/2.0 #Final Vertical Position [m]

wfr4 = deepcopy(wfr2)

# Generate the gaussian beams
srwl.CalcElecFieldGaussian(wfr1, g_Beam, [0])
srwl.CalcElecFieldGaussian(wfr2, g_Beam, [0])
srwl.CalcElecFieldGaussian(wfr3, g_Beam, [0])

# Check that the real parts of wf1 and wfr2 (for the second color are the same)
# np.array_equiv(wfr2.arEx[2::4], wfr1.arEx[::2])


def add_color_to_existing_wfr(wfr, wfr_s, Nc):
    wfr.arEx[2*Nc::2*wfr.mesh.ne] = wfr_s.arEx[0::2]
    wfr.arEx[2*Nc+1::2*wfr.mesh.ne] = wfr_s.arEx[1::2]
    wfr.arEy[2*Nc::2*wfr.mesh.ne] = wfr_s.arEy[0::2]
    wfr.arEy[2*Nc+1::2*wfr.mesh.ne] = wfr_s.arEy[1::2]
    return wfr

add_color_to_existing_wfr(wfr4, wfr3, 0)
add_color_to_existing_wfr(wfr4, wfr1, 1)


plot_SRW_intensity(wfr2, fig_num=3, title="SRW Generated")
plot_SRW_intensity(wfr4, fig_num=4, title="Brendan Generated")


