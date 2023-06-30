#!/usr/local/python
#

#############################################################################
# Doing some quick analysis on SRW generated data.


import sys
import os
# Add the path to the SRW_Parallel library (One directory up)
file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, file_dir)
# Add the path to the SRW_Utilities library (Two directory up)
file_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, file_dir)

# set the backend for matplot lib
import matplotlib
# matplotlib.use("TkAgg")


from srwlib import *
from SRW_Split_Wavefront_Mesh import *
from SRW_Utilities import *

import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io
from scipy.ndimage import rotate
import itertools
from F2_Single_Magnet_Sim import *
from matplotlib.colors import LinearSegmentedColormap
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()



if __name__ == '__main__':

    # Define a colormap
    cmap0 = LinearSegmentedColormap.from_list('', ['white', 'darkblue'])

    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_dir = '/'.join(file_dir.split('/')[:-1])
    sim1name = file_dir + '/F2_BC11_B2_B3.py'
    sim1 = load_sim_from_disc(sim1name)
    sim1F = np.abs(return_SRW_fluence(sim1.wfr))

    sim2name = file_dir + '/F2_BC11_B2_B3_Camera_BW.py'
    sim2 = load_sim_from_disc(sim2name)
    sim2F = np.abs(return_SRW_fluence(sim2.wfr))

    # Nx1 = sim1.wfr.mesh.nx
    # Ny1 = sim1.wfr.mesh.ny
    xMin1 = 1e3 * sim1.wfr.mesh.xStart
    xMax1 = 1e3 * sim1.wfr.mesh.xFin
    yMin1 = 1e3 * sim1.wfr.mesh.yStart
    yMax1 = 1e3 * sim1.wfr.mesh.yFin

    plt.figure(126743, facecolor='w', figsize=(8,6))

    plt.subplot(1,2,1)
    plt.imshow(sim1F, extent=[xMin1, xMax1, yMin1, yMax1], cmap=cmap0)
    # plt.gca().set_aspect((xMax1 - xMin1) / (yMax1 - yMin1))
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.clim([0, np.max(sim1F)])
    plt.title('With 25 nm Filter', fontsize=20)
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    # Nx1 = sim2.wfr.mesh.nx
    # Ny = sim2.wfr.mesh.ny
    xMin2 = 1e3 * sim2.wfr.mesh.xStart
    xMax2 = 1e3 * sim2.wfr.mesh.xFin
    yMin2 = 1e3 * sim2.wfr.mesh.yStart
    yMax2 = 1e3 * sim2.wfr.mesh.yFin

    plt.subplot(1,2,2)
    plt.imshow(sim2F, extent=[xMin2, xMax2, yMin2, yMax2], cmap=cmap0)
    # plt.gca().set_aspect((xMax2 - xMin2) / (yMax2 - yMin2))
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.clim([0, np.max(sim2F)])
    plt.title('No Filter', fontsize=20)
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.tight_layout()

    ############################################################################
    # Plot a line out along the x axis (constant y) to really capture the
    # difference between the two.
    X1 = np.linspace(xMin1, xMax1, sim1.wfr.mesh.nx)
    Y1 = sim1F[sim1.wfr.mesh.ny // 2, :]

    X2 = np.linspace(xMin2, xMax2, sim2.wfr.mesh.nx)
    Y2 = sim2F[sim2.wfr.mesh.ny // 2, :]

    Y1_scale = Y2[sim2.wfr.mesh.ny // 4] / Y1[sim2.wfr.mesh.ny // 4]

    plt.figure(690, facecolor='w')
    plt.plot(X1, Y1_scale*Y1, 'b')
    plt.plot(X2, Y2, 'r')
    plt.xlim([-0.5, 0.5])
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("Intensity [arb]", fontsize=20)
    plt.legend(('with 25 nm filter', 'without filter'))
    plt.tight_layout()

    ############################################################################
    # Compare simulation data to experimental data
    expData = scipy.io.loadmat('ProfMon-PROF_LI11_342-2022-11-21-173241.mat')['data'][0, 0][1]
    rotatedExpData = rotate(expData, angle=94)
    XE = np.linspace(1, rotatedExpData.shape[1], rotatedExpData.shape[
        1])*3.75e-3
    XE = XE - np.mean(XE)
    YE = np.linspace(1, rotatedExpData.shape[0], rotatedExpData.shape[
        1])*3.75e-3
    YE = YE - np.mean(YE)


    plt.figure(987, facecolor='w', figsize=(10, 8))

    plt.subplot(1,2,1)
    plt.imshow(rotatedExpData, extent=[XE[0], XE[-1], YE[0], YE[-1]],
               cmap=cmap0)
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.title('Experiment', fontsize=20)
    plt.clim([0, np.max(rotatedExpData)])
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.subplot(1,2,2)
    plt.imshow(sim2F, extent=[xMin2, xMax2, yMin2, yMax2], cmap=cmap0)
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("y [mm]", fontsize=20)
    plt.clim([0, np.max(sim2F)])
    plt.title('Simulation', fontsize=20)
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.tight_layout()

    ############################################################################
    # Plot a line out along the x axis (constant y) to really capture the
    # difference between experiment and simulation
    Y1 = rotatedExpData[291+10, :]

    X2 = np.linspace(xMin2, xMax2, sim2.wfr.mesh.nx)
    Y2 = sim2F[sim2.wfr.mesh.ny // 2, :]

    Y1_scale = np.sqrt(1/2)*2.0*Y2[sim2.wfr.mesh.ny // 4] / Y1[
        rotatedExpData.shape[1] // 4]

    plt.figure(691, facecolor='w')
    plt.plot(XE, Y1_scale*Y1, 'b')
    plt.plot(X2, Y2, 'r')
    plt.xlim([-1.0, 1.0])
    plt.xlabel("x [mm]", fontsize=20)
    plt.ylabel("Intensity [arb]", fontsize=20)
    plt.legend(('Experiment', 'Simulation'))
    plt.tight_layout()