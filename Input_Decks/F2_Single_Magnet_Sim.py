#!/usr/local/python
#

#############################################################################
# This file file defines a class for a single magnet in the FACET-II dogleg
# and BC11 chicane. They are the same magnets, so I'm building a class to
# handle doing simulations magnet, but magnet so I can combine them as I see
# fit after the fact.
# A basic SRW comes in 4 parts:
# 1) Build a magnetic field container
# 2) Use (1) to calculate a particle trajectory
# 3) Build a wavefront object to deposit fields onto
# 4) Use the particle trajectory to calculate fields and deposit them on the
# wavefront object from (3)

# This file is for building and testing, so it will have a bunch of stuff you
#  may not need in the future.


import sys
import os
# Add the path to the SRW_Parallel library (one directory up)
file_dir = os.path.dirname(os.path.realpath(__file__))
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
from operator import add
# import h5py
# Some matplotlib settings.
plt.close('all')
plt.ion()


class F2_Single_Magnet_Sim:

    # Magnet parameters --------------------------------------------------------
    # The design bend angle, from theory or simulation. In degrees.
    # goal_Bend_Angle = 0.105 * 180 / np.pi

    # Magnetic field strength in Tesla. This should be negative.
    B0 = -0.6

    # Length of the 'hard' edge of the dipoles in meters.
    L_bend = 0.204

    # Length of the field edge in meters.
    L_edge = 0.05

    # Entry and exit drifts, in meters. The simulation can't start in a
    # magnetic field.
    entry_drift = 0.3

    # Distance between the magnet edges if they had zero length edges.
    # Subtract off L_edge because the measurements/Lucretia files don't
    # include edges in distance. The 1.6 is a fudge factor for in simulation
    # length vs the defined length.
    Bend_sep = 0.93 - 1.6 * 2 * L_edge

    goalCtEnd = 1.0 * L_bend + 2.0 * L_edge + 2.0 * entry_drift

    # Initial beam parameters --------------------------------------------------
    # Beam Parameters.
    beam_energy	= 0.330 # Beam energy in GeV
    x_initial 	= 0.0 # Initial x offset in meters
    xp_initial 	= 0.0 # initial xprime in rad
    z_initial 	= 0.0 # In meters
    # Number of points to compute the particle trajectory.
    npTraj = 2 ** 14


    # Wavefront mesh grid parameters -------------------------------------------
    # Wavefront parameters
    B3_phys_edge    = entry_drift + 1.6*3*L_edge + L_bend + Bend_sep # The
    # physical edge of B3 (i.e. where the field has just become flat.)
    zSrCalc 	    = B3_phys_edge + 1.045 # Distance from sim start to
    # calc SR. [m]
    xMiddle		    = 0.0 # middle of window in X to calc SR [m]
    xWidth 		    = 0.08*1.0 # width of x window. [m]
    yMiddle 	    = 0.00 # middle of window in Y to calc SR [m]
    yWidth 		    = xWidth # width of y window. [m]

    # SR integration flags.
    use_termin 		= 1 #1 #Use "terminating terms" (i.e. asymptotic expansions
    # at  zStartInteg and zEndInteg) or not (1 or 0 respectively)
    # Precision for SR integration
    srPrec 		    = 0.05 #0.01

    def __init__(self, Nx=2**10, goal_Bend_Angle = 6.0, meshZ=2.0, ph_lam=0.65e-6):
        """

        :param Nx: Number of grid cells in X and Y for the wavefront
        calculation
        :param meshZ: Distance of the mesh from the FIRST edge of the magnet.
        I'm working under the assumption that the fields and intensity are
        the same going into and coming out of the magnet.
        """
        # Set the number of grid cells of the wavefront mesh
        self.Nx = Nx
        self.Ny = Nx

        # Set the desired bend angle.
        self.goal_Bend_Angle = goal_Bend_Angle

        # Set the wavelength
        self.photon_lam = ph_lam

        # Set the Z location of the mesh with respect to the first magnet edge
        self.zSrCalc = meshZ + self.entry_drift + 1.6 * self.L_edge

        # Set up the magnetic field container and particle trajectory
        self.magFldCnt = self.build_single_magnet()
        self.partTraj  = self.build_particle_trajectory()

        # Tweak the simulation parameters to better reflect inputs
        self.set_mag_strength_to_desired_bend_angle()
        self.set_simulation_length()

        # Setup the wavefront
        # self.wfr = self.build_wavefront_mesh()

    def build_single_magnet(self):
        """
        # This function builds a SRW magnetic field container containing a single
        # magnet.

        :return: magFldCont - SRW magnetic field container for the magnets
        """
        bend1 = SRWLMagFldM()
        bend1.m = 1  # 1 defines a dipole
        # Field strength of the bend in Tesla, since it is a dipole
        bend1.G = self.B0
        bend1.Leff = self.L_bend
        bend1.Ledge = self.L_edge

        z1 = self.entry_drift + self.L_bend / 2.0 + self.L_edge

        # Offsets for all the magnetic fields in the magFldCnt.
        bendy = [bend1]
        xcID = [0.0]
        ycID = [0.0]
        zcID = [z1]

        # Put everything together.  These is the field.
        magFldCnt = SRWLMagFldC(bendy, array('d', xcID), array('d', ycID),
                                array('d', zcID))
        return magFldCnt

    def build_particle_trajectory(self):
        """
        Every SRW simulation needs a particle trajectory. Setup a particle
        trajectory object and calculate the trajectory.
        The particle trajectory is updated later using other methods,
        if necessary.

        :return: A calculated SRW Particle Trajectory
        """

        # Build an SRWLParticle - No need to save this.
        part = SRWLParticle()
        part.x = self.x_initial
        part.y = 0.0
        part.xp = self.xp_initial
        part.yp = 0.0
        part.z = self.z_initial
        part.gamma = self.beam_energy / 0.51099890221e-03
        part.relE0 = 1  # Electron Rest Mass
        part.nq = -1  # Electron Charge

        partTraj = SRWLPrtTrj()
        partTraj.partInitCond = part
        partTraj.allocate(self.npTraj, True)
        partTraj.ctStart = 0.0  # Start Time for the calculation
        partTraj.ctEnd = self.goalCtEnd

        # Calculate the particle trajectory, [1] is 'trajPrecPar' in the
        # documentation
        temp = srwl.CalcPartTraj(partTraj, self.magFldCnt, [1])
        return temp

    def plot_trajectory_and_b_field(self, figNum = 123):
        """
        Plot the particle trajectory to check that everything is working.

        :param figNum: figure number to plot to prevent figure collision
        :return: Nothing
        """
        plt.close(figNum)
        plt.figure(figNum, facecolor='w')

        plt.subplot(3, 1, 1)
        plt.plot(self.partTraj.arZ, self.partTraj.arX)
        plt.xlabel('Z [m]', fontsize=18)
        plt.ylabel('X [m]', fontsize=18)

        plt.subplot(3, 1, 2)
        plt.plot(self.partTraj.arZ, self.partTraj.arXp)
        plt.xlabel('Z [m]', fontsize=18)
        plt.ylabel('Xp [rad]', fontsize=18)

        plt.subplot(3, 1, 3)
        plt.plot(self.partTraj.arZ, self.partTraj.arBy)
        plt.xlabel('Z [m]', fontsize=18)
        plt.ylabel('By [T]', fontsize=18)

        plt.tight_layout()

    def set_simulation_length(self):
        """
        To make the length of simulation symmetric in magnets with large bend
        angle, like the dogleg magnets, you need to know the total bend
        angle. They bend so much that the particle doesn't travel as far in
        Z because it is deflected far enough in Y to make a difference. This
        function runs the simulation for the input ctEnd and the
        updates the simulation parameter partTraj.ctEnd so that the total
        simulation lenght partTraj.arZ[-1] matches goalCtEnd.

        :return: N/A
        """
        self.partTraj.ctEnd = self.partTraj.ctEnd + \
                              (self.goalCtEnd - self.partTraj.arZ[-1])
        self.partTraj = srwl.CalcPartTraj(self.partTraj, self.magFldCnt, [1])

    def set_mag_strength_to_desired_bend_angle(self):
        """
        When the entrance and exit edge fields change length, the total bend
        angle will change. This function sets the field strength such that
        the bend angle is the user input value.

        This function updates the magFldCnt to have the B value which gets
        the desired bend angle.
        :return: N/A
        """

        currBendAngle = self.partTraj.arXp[self.partTraj.np-1] * 180 / np.pi
        deltaTheta = (self.goal_Bend_Angle - currBendAngle) / currBendAngle
        self.B0 = self.B0 * (1 + deltaTheta)
        self.magFldCnt = self.build_single_magnet()
        self.partTraj  = self.build_particle_trajectory()
        self.set_simulation_length()

    def run_SR_calculation(self):
        # ***********Precision Parameters for SR calculation
        # SR calculation method: 0- "manual", 1- "auto-undulator",
        # 2- "auto-wiggler"
        # This is best set to 2 for edge radiation. It is burried in
        # documentation.
        meth = 2
        # longitudinal position to start integration (effective if < zEndInteg)
        zStartInteg = self.partTraj.ctStart
        # longitudinal position to finish integration (effective if> zStartInteg)
        zEndInteg = self.partTraj.ctEnd
        # sampling factor for adjusting nx, ny (effective if > 0)
        sampFactNxNyForProp = 0
        arPrecPar = [meth,
                     self.srPrec,
                     zStartInteg,
                     zEndInteg,
                     self.partTraj.np,
                     self.use_termin,
                     sampFactNxNyForProp]

        # Run the simulation
        t0 = time.time()
        time_str = 'SR Calculation started at ' + time.ctime() + \
                   '.'
        print(time_str)
        srwl.CalcElecFieldSR(self.wfr, 0, self.magFldCnt, arPrecPar)
        time_str = "Run time: %.2f seconds. \n" % (time.time() - t0)
        print(time_str)

    def propagate_wavefront_through_optics(self, fLength=0.105, pLength=0.105):
        """
        Propagate the wavefront through an aperture, lens and some finite
        distance
        :return:  N/A
        """

        # I'm not sure what the purpose of xc/yc is, but it causes the wavefront
        #  to shift when propagating, and I don't think that is physical for
        # my setup.
        self.wfr.xc = 0.0
        self.wfr.yc = 0.0

        t0 = time.time()
        time_str = 'SRW Physical Optics Calculation started at ' + time.ctime() + \
                   '.'
        print(time_str)

        focal_length    = 1.0 * fLength  # in meters
        prop_distance   = 1.0 * pLength  # in meters

        paramsAper  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
        paramsLens  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
        paramsDrift = [0, 0, 1., 1, 0, 1., 1., 1., 1., 0, 0, 0]

        a_drift = SRWLOptC(
            [SRWLOptA(_shape='c', _ap_or_ob='a', _Dx=0.075),
             SRWLOptL(focal_length, focal_length),
             SRWLOptD(prop_distance)],
            [paramsAper, paramsLens, paramsDrift])
        srwl.PropagElecField(self.wfr, a_drift)

        time_str = "Run time: %.2f seconds. \n" % (time.time() - t0)
        print(time_str)

    def propagate_small_distance(self, propDist):
        """
        Used to add small amounts of phase to the wavefronts
        :return:
        """

        paramsDrift = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

        a_drift = SRWLOptC(
            [SRWLOptD(propDist)],
            [paramsDrift])
        srwl.PropagElecField(self.wfr, a_drift)

    def propagate_wavefront_through_window(self, windowToLens=0.215):
        """
        Propagate the wavefront through an aperture, lens and some finite
        distance
        :param windowToLens: The distance from the window to the lens, in meters
        :return:  N/A
        """

        # I'm not sure what the purpose of xc/yc is, but it causes the wavefront
        #  to shift when propagating, and I don't think that is physical for
        # my setup.
        self.wfr.xc = 0.0
        self.wfr.yc = 0.0

        t0 = time.time()
        time_str = 'SRW Physical Optics Calculation started at ' + time.ctime() + \
                   '.'
        print(time_str)

        focal_length    = 1.0 * 0.105  # in meters
        prop_distance   = 1.0 * 0.105  # in meters

        paramsAper  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
        paramsLens  = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
        paramsDrift = [0, 0, 1., 1, 0, 1., 1., 1., 1., 0, 0, 0]

        a_drift = SRWLOptC(
            [SRWLOptA(_shape='c', _ap_or_ob='a', _Dx=0.038),
             SRWLOptD(windowToLens),
             SRWLOptA(_shape='c', _ap_or_ob='a', _Dx=0.075),
             SRWLOptL(focal_length, focal_length),
             SRWLOptD(prop_distance)],
            [paramsAper, paramsDrift, paramsAper, paramsLens, paramsDrift])
        srwl.PropagElecField(self.wfr, a_drift)

        time_str = "Run time: %.2f seconds. \n" % (time.time() - t0)
        print(time_str)

    def __str__(self):
        print("Simulation Mesh Parameters")
        print('Input Nx = Ny = ', self.wfr.mesh.nx, self.wfr.mesh.ny)
        print('Input xWdith ', self.wfr.mesh.xFin - self.wfr.mesh.xStart, ' [m]')
        print('Input yWdith ', self.wfr.mesh.yFin - self.wfr.mesh.yStart, ' [m]')
        return "\n"

    def lineout_in_y(self, nx):
        """
        This function does a lineout along the y-axis for a fixed nx point.
        It returns the lineout and an appropriate vector for the physical
        pixel locations
        :param nx: x pixel to perform the y-lineout in.
        :return y: vector of pixel locations
        :return yLineout: lineout in Y at x pixel nx
        """

        I = convert_Efield_to_intensity(self.wfr)
        yLineout = I[:, nx]
        y = np.linspace(self.wfr.mesh.yStart,
                        self.wfr.mesh.yFin,
                        self.wfr.mesh.ny)
        return y, yLineout

    def resize_wavefront(self, newX=4.850e-3, newY=3.615e-3):
        """
        * Change the physical size of the wavefront mesh.
        * Does not change the number of grid points. This is accomplished by
        the 1/Scale in the srwl call below.
        * Result is 'in place' and stored in self.wfr
        * Defaults are the sizes of an Allied Vision Mako G125B camera.
        :param newX: Desired horizontal size of the mesh, in meters
        :return:
        """
        xScale = newX / (self.wfr.mesh.xFin - self.wfr.mesh.xStart)
        yScale = newY / (self.wfr.mesh.yFin - self.wfr.mesh.yStart)
        srwl.ResizeElecField(self.wfr, 'c', [0,
                                             xScale,
                                             1./xScale,
                                             yScale,
                                             1./yScale,
                                             0.5,
                                             0.5])

    def lineout(self, xOrY=0, N=1):
        """
        Function for producing a lineout of the intensity of a given wavefront.

        :param xOrY: 0 means the lineout is along the x-axis, 1 means y-axis
        :param N: The pixel number (for x or y) to take the lineout along
        :return: vector of positions, the lineout
        """
        Nx = self.wfr.mesh.nx
        Ny = self.wfr.mesh.ny

        # Extract the single particle intensity
        arI1 = array('f', [0] * self.wfr.mesh.nx * self.wfr.mesh.ny)
        srwl.CalcIntFromElecField(arI1, self.wfr, 6, 0, 3, self.wfr.mesh.eStart, 0, 0)
        B = np.reshape(arI1, [Ny, Nx])

        if xOrY == 0:
            pMin = self.wfr.mesh.xStart
            pMax = self.wfr.mesh.xFin
            pN = self.wfr.mesh.nx
            lineout = B[N, :]
        else:
            pMin = self.wfr.mesh.yStart
            pMax = self.wfr.mesh.yFin
            pN = self.wfr.mesh.ny
            lineout = B[:, N]

        return np.linspace(pMin, pMax, pN), lineout

    def add_wavefront(self, wfr_in):
        """
        Method to add a wavefront to the wavefront contained in this class.
        This method assumes you're adding like to like. It does no checking
        of mesh size, mesh density or photon counts.

        :param wfr_in: The wavefront to add to the current wavefront
        :return: nothing, wavefront is added in place
        """
        self.wfr.addE(wfr_in)

    def match_wavefront_mesh_dimensions(self, wfr_in):
        """
        Update the mesh dimensions of the instance mesh to match the mesh
        dimensions of the wfr_in mesh. This changes wfr.mesh.xStart, xFin,
        yStart, yFin

        :param wfr_in: The wavefront to copy the mesh parameters from
        :return: Nothing, modifies instance mesh.
        """
        try:
            self.wfr.mesh
        except:
            print('Wavefront not defined for current simulation')
            return

        self.wfr.mesh.xStart = wfr_in.mesh.xStart
        self.wfr.mesh.xFin   = wfr_in.mesh.xFin
        self.wfr.mesh.yStart = wfr_in.mesh.yStart
        self.wfr.mesh.yFin   = wfr_in.mesh.yFin
        return


class F2_Single_Magnet_Single_Color_Sim(F2_Single_Magnet_Sim):
    def __init__(self, Nx=2**10, goal_Bend_Angle = 6.0, meshZ=2.0,
                 ph_lam=0.60e-6):
        super(F2_Single_Magnet_Single_Color_Sim, self).__init__(Nx,
                                                   goal_Bend_Angle,
                                                   meshZ,
                                                   ph_lam)
        # Setup the wavefront
        self.wfr = self.build_wavefront_mesh()

    def build_wavefront_mesh(self):
        """
        Build a wavefront mesh to hold the generated radiation data. Written
        for single wavelength

        :return:
        """
        # convert wavelength to eV
        photon_e = 4.135e-15 * 299792458.0 / self.photon_lam

        # Set up an electron beam faking a single particle
        # This is what is used to generate the SR.
        elecBeam = SRWLPartBeam()
        elecBeam.Iavg = 0.5  # Average Current [A]
        elecBeam.partStatMom1.x       = self.partTraj.partInitCond.x
        elecBeam.partStatMom1.y       = self.partTraj.partInitCond.y
        elecBeam.partStatMom1.z       = self.partTraj.partInitCond.z
        elecBeam.partStatMom1.xp      = self.partTraj.partInitCond.xp
        elecBeam.partStatMom1.yp      = self.partTraj.partInitCond.yp
        elecBeam.partStatMom1.gamma   = self.beam_energy / 0.51099890221e-03

        # *********** Wavefront data placeholder
        wfr1 = SRWLWfr()  # For spectrum vs photon energy
        # Numbers of points vs Photon Energy, Horizontal and Vertical Positions
        wfr1.allocate(1, self.Nx, self.Ny)
        wfr1.mesh.zStart    = self.zSrCalc  # Longitudinal Position [m] from
        # Center of Straight Section at which SR has to be calculated
        wfr1.mesh.eStart    = photon_e  # Initial Photon Energy [eV]
        wfr1.mesh.eFin      = photon_e  # Final Photon Energy [eV]
        wfr1.mesh.xStart    = self.xMiddle - self.xWidth / 2.0  # Initial Horizontal Position [m]
        wfr1.mesh.xFin      = self.xMiddle + self.xWidth / 2.0  # Final Horizontal Position [m]
        wfr1.mesh.yStart    = self.yMiddle - self.yWidth / 2.0  # Initial Vertical Position [m]
        wfr1.mesh.yFin      = self.yMiddle + self.yWidth / 2.0  # Final Vertical Position [m]
        wfr1.partBeam       = elecBeam

        return wfr1

class F2_Single_Magnet_Multiple_Color_Sim(F2_Single_Magnet_Sim):
    def __init__(self, Nx=2 ** 10, goal_Bend_Angle=6.0, meshZ=2.0,
                 ph_lam=0.60e-6):
        super(F2_Single_Magnet_Multiple_Color_Sim, self).__init__(Nx,
                                                                goal_Bend_Angle,
                                                                meshZ,
                                                                ph_lam)
        # Setup the wavefront
        self.build_wavefront_mesh()

    def build_wavefront_mesh(self, Ne = 2, p=(500.0e-9, 600e-9)):
        """
        Build a wavefront mesh to hold the generated radiation data. Written
        for multiple wavelengths

        :return:
        """
        # convert wavelength to eV
        photon_e_lo = 4.135e-15 * 299792458.0 / p[0]
        photon_e_hi = 4.135e-15 * 299792458.0 / p[1]

        # Set up an electron beam faking a single particle
        # This is what is used to generate the SR.
        elecBeam = SRWLPartBeam()
        elecBeam.Iavg = 0.5  # Average Current [A]
        elecBeam.partStatMom1.x       = self.partTraj.partInitCond.x
        elecBeam.partStatMom1.y       = self.partTraj.partInitCond.y
        elecBeam.partStatMom1.z       = self.partTraj.partInitCond.z
        elecBeam.partStatMom1.xp      = self.partTraj.partInitCond.xp
        elecBeam.partStatMom1.yp      = self.partTraj.partInitCond.yp
        elecBeam.partStatMom1.gamma   = self.beam_energy / 0.51099890221e-03

        # *********** Wavefront data placeholder
        wfr1 = SRWLWfr()  # For spectrum vs photon energy
        # Numbers of points vs Photon Energy, Horizontal and Vertical Positions
        wfr1.allocate(Ne, self.Nx, self.Ny)
        wfr1.mesh.zStart    = self.zSrCalc  # Longitudinal Position [m] from
        # Center of Straight Section at which SR has to be calculated
        wfr1.mesh.eStart    = photon_e_lo  # Initial Photon Energy [eV]
        wfr1.mesh.eFin      = photon_e_hi  # Final Photon Energy [eV]
        wfr1.mesh.xStart    = self.xMiddle - self.xWidth / 2.0  # Initial Horizontal Position [m]
        wfr1.mesh.xFin      = self.xMiddle + self.xWidth / 2.0  # Final Horizontal Position [m]
        wfr1.mesh.yStart    = self.yMiddle - self.yWidth / 2.0  # Initial Vertical Position [m]
        wfr1.mesh.yFin      = self.yMiddle + self.yWidth / 2.0  # Final Vertical Position [m]
        wfr1.partBeam       = elecBeam

        self.wfr = wfr1
        return

    def add_specific_color_wavefront(self, wfr_in, Nc):
        """
        Method to update only one of the colors of this multiple color
        simulation. You may want to do this when you are simulating and
        tracking each color separately to add in QE effects from cameras.
        This method adds the color, it does not replace it.

        Nc is the color you're adding. An SRWLWfr has mesh.ne colors that run
        from mesh.eStart to mesh.eFin. For example:
        ne = 3, eStart = 500 nm, eFin = 600 nm. Then Nc = 0 is 500 nm,
        Nx = 1 is 550 nm, and Nc = 2 is 600 nm.

        :param wfr_in: The wavefront to add to the multiple color wavefront mesh
        :param Nc: The color you're adding.
        :return:
        """

        self.wfr.arEx[2 * Nc :: 2 * self.wfr.mesh.ne] = \
            array('f', list(
                map(add,
                    self.wfr.arEx[2 * Nc:: 2 * self.wfr.mesh.ne],
                    wfr_in.arEx[0::2]))
                  )

        self.wfr.arEx[2 * Nc + 1 :: 2 * self.wfr.mesh.ne] = \
            array('f', list(
                map(add,
                    self.wfr.arEx[2 * Nc + 1 :: 2 * self.wfr.mesh.ne],
                     wfr_in.arEx[1::2]))
                  )
        self.wfr.arEy[2 * Nc :: 2 * self.wfr.mesh.ne] = \
            array('f', list(
                map(add,
                    self.wfr.arEy[2 * Nc:: 2 * self.wfr.mesh.ne],
                     wfr_in.arEy[0::2]))
                  )
        self.wfr.arEy[2 * Nc + 1 :: 2 * self.wfr.mesh.ne] = \
            array('f', list(
                map(add,
                    self.wfr.arEy[2 * Nc + 1:: 2 * self.wfr.mesh.ne],
                     wfr_in.arEy[1::2]))
            )

if __name__ == '__main__':

    # Create the simulation
    B2B3_first_edge_to_camera = 1.795 # in meters
    B2B3_second_edge_to_camera = 1.795 - 0.75  # in meters
    B2B3_first_edge_to_window = 1.795 - 0.215 # in meters
    B2B3_second_edge_to_window = 1.795 - 0.215 - 0.75  # in meters

    # Run the simulation for the first edge.
    B2B3_first_edge = F2_Single_Magnet_Sim(Nx=2**10,
                                  goal_Bend_Angle=-.105 * 180 / np.pi,
                                  meshZ=B2B3_first_edge_to_window,
                                  ph_lam=0.65e-6)
    # Run the SRW calculation
    B2B3_first_edge.run_SR_calculation()


    # Run the simulation for the second edge.
    B2B3_second_edge = F2_Single_Magnet_Sim(Nx=2**10,
                                  goal_Bend_Angle=.105 * 180 / np.pi,
                                  meshZ=B2B3_second_edge_to_window,
                                  ph_lam=0.65e-6)
    # Run the SRW calculation
    B2B3_second_edge.run_SR_calculation()

    B2B3_first_edge.propagate_wavefront_through_window()
    B2B3_second_edge.propagate_wavefront_through_window()

    # Resize the wavefronts after propagation
    B2B3_first_edge.resize_wavefront()
    B2B3_second_edge.resize_wavefront()

    # Combine the two wavefronts
    wfr = deepcopy(B2B3_second_edge.wfr)
    wfr.addE(B2B3_first_edge.wfr)
    plot_SRW_intensity(wfr, title="650 nm, after lens")

    #
    # B2B3_first_edge.__str__()
    # xScale = 4.850e-3 / (B2B3_first_edge.wfr.mesh.xFin -
    #                       B2B3_first_edge.wfr.mesh.xStart)
    # srwl.ResizeElecField(B2B3_first_edge.wfr, 'c', [0,
    #                                      xScale,
    #                                      1/xScale,
    #                                      1.,
    #                                      1.,
    #                                      0.5,
    #                                      0.5])
    # B2B3_first_edge.__str__()


    # plot_SRW_intensity(B2B3_first_edge.wfr, fig_num = 124,
    #                    title='B2B3 First Edge')
    # print('First Edge Parameters:')
    # B2B3_first_edge.__str__()
    #
    # plot_SRW_intensity(B2B3_second_edge.wfr, fig_num = 125,
    #                    title='B2B3 Second Edge')
    # print('Second Edge Parameters:')
    # B2B3_second_edge.__str__()

    # B2B3_first_y, B2B3_first_yLineout = B2B3_first_edge.lineout_in_y(128)
    # B2B3_second_y, B2B3_second_yLineout = B2B3_second_edge.lineout_in_y(3*128)
    # plt.close(12351)
    # plt.figure(12351, facecolor='w')
    # plt.plot(B2B3_first_y*1e3, B2B3_first_yLineout, 'bo')
    # plt.plot(B2B3_second_y*1e3, B2B3_second_yLineout, 'rx')
    # plt.xlabel('Y [mm]', fontsize=18)
    # plt.ylabel('Intensity [arb]', fontsize=18)
    # plt.legend(['B2B3 First Edge', 'B2B3 Second Edge'])



