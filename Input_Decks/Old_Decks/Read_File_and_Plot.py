#!/usr/local/python
#

#############################################################################
# This file is simulating the second and third bend magnets in the FACET BC11
# chicane.



from __future__ import print_function #Python 2.7 compatibility
import sys
import os
#sys.path.insert(0, '/Users/brendan/Documents/FACET/SRW/SRW-light/env/work/srw_python') # To find the SRW python libraries.
# sys.path.insert(0, '/scratch/brendan/SRW/SRW/env/work/srw_python') # To find
# # the SRW python libraries.
# sys.path.insert(0, '/scratch/brendan/SRW_Parallel/SRW_Parallel')

# Add the path to the SRW_Parallel library (one directory up)
file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, file_dir)


# set the backend for matplot lib
# import matplotlib
# matplotlib.use("TkAgg")


from srwlib import *
from SRW_Split_Wavefront_Mesh import *
from SRW_Utilities import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
# import FourierOptics as FO
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()


if __name__ == '__main__':

    # Load the wavefront to a file.
    filename = 'F2_BC11_B2_and_B3_Nx_1024'
    wfr = load_srw_wavefront(filename)
    # wfr.Rx = 1.05*wfr.Rx
    # wfr.Ry = 1.05*wfr.Ry

    # plot_SRW_intensity(wfr)

    t0 = time.time()
    time_str = 'SRW Physical Optics Calculation started at ' + time.ctime() + \
               '. \n'
    print(time_str, end='')

    wfr_out = deepcopy(wfr)
    focal_length = 1.0*0.105 # in meters
    prop_distance = 1.0*0.105 # in meters
    # Robbie working
    paramsAper  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
    paramsLens  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
    paramsDrift = [0, 0, 1., 1, 0, 1., 1., 1., 1., 0, 0, 0]

    a_drift = SRWLOptC(
        [SRWLOptA(_shape = 'c', _ap_or_ob = 'a', _Dx = 0.075),
         SRWLOptL(focal_length, focal_length),
         SRWLOptD(prop_distance)],
        [paramsAper, paramsLens, paramsDrift])
    srwl.PropagElecField(wfr_out, a_drift)

    time_str = "Run time: %.2f seconds." % (time.time() - t0)
    print(time_str)

    print("")
    print('paramsAper:  ' + str(paramsAper))
    print('paramsLens:  ' + str(paramsLens))
    print('paramsDrift: ' + str(paramsDrift))
    print('Input Nx = Ny = ', wfr.mesh.nx, wfr.mesh.ny)
    print('Ouput Nx = Ny = ', wfr_out.mesh.nx, wfr_out.mesh.ny)
    print('Input xWdith ', wfr.mesh.xFin - wfr.mesh.xStart, ' [m]')
    print('Input yWdith ', wfr.mesh.yFin - wfr.mesh.yStart, ' [m]')
    print('Output xWdith ', wfr_out.mesh.xFin - wfr_out.mesh.xStart, ' [m]')
    print('Output yWdith ', wfr_out.mesh.yFin - wfr_out.mesh.yStart, ' [m]')

    plot_SRW_intensity(wfr, 1)
    plot_SRW_intensity(wfr_out, 2)


  #*****Propagation Parameters for the Optical Elements
  #Meaning of the array element below:
  #[ 0]: Auto-Resize (1) or not (0) Before propagation
  #[ 1]: Auto-Resize (1) or not (0) After propagation
  #[ 2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
  #[ 3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
  #[ 4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
  #[ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
  #[ 6]: Horizontal Resolution modification factor at Resizing
  #[ 7]: Vertical Range modification factor at Resizing
  #[ 8]: Vertical Resolution modification factor at Resizing
  #[ 9]: Type of wavefront Shift before Resizing (not yet implemented)
  #[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
  #[11]: New Vertical wavefront Center position after Shift (not yet implemented)
  #[12]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Horizontal Coordinate
  #[13]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Vertical Coordinate
  #[14]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Longitudinal Coordinate
  #[15]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Horizontal Coordinate
  #[16]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Vertical Coordinate
