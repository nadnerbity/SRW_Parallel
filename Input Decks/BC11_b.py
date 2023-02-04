
import sys
sys.path.insert(0, '/scratch/brendan/SRW/SRW/env/work/srw_python') # To find
# the SRW python libraries.
sys.path.insert(0, '/scratch/brendan/SRW_Parallel/SRW_Parallel')

import matplotlib.pyplot as plt
import srwlib as SW
import numpy as np
import copy
import pickle
from scipy.interpolate import interp2d
import time

me = 9.11e-31
c = 3e8
qe = 1.6e-19
h = 6.63e-34


class EdgeRad:

    def __init__(self, field1_params=None, field2_params=None,
                 cent_traj_params=None, beam_params=None,
                 wavefront_params=None, precision_params=None):
        """
        :param field1_params: [B_field, length, edge_length, position]
        :param field2_params: [B_field, length, edge_length, position]
        :param cent_traj_params: [x, y, z, xp, yp, gamma]
        :param beam_params: [sig_x, sig_xp, sig_y, sig_yp, current]
        :param wavefront_params: [shape:[n_e, n_x, n_y],
                           extent:[x_min, x_max, y_min, y_max], z_pos,
                           energy_bound, use_angle]
        :param precision_params: [method, precision, z_start, z_end, n_Traj,
                                  terminate, sampFactNxNyForProp]
        """

        self.field1_params = field1_params
        self.field2_params = field2_params
        self.cent_traj_params = cent_traj_params
        self.beam_params = beam_params
        self.wavefront_params = wavefront_params
        self.precision_params = precision_params

        # Simulation classes
        self.beam = None
        self.magFldCnt = None
        self.wavefront = None
        self.field = None
        self.setup()

    def setup(self):
        """
        Resets the params for a new simulation
        """

        # Define field
        self.field_1 = SW.SRWLMagFldM(self.field1_params[0], 1, 'n',
                                      self.field1_params[1],
                                      self.field1_params[2], 0)
        self.field_2 = SW.SRWLMagFldM(self.field2_params[0], 1, 'n',
                                      self.field2_params[1],
                                      self.field2_params[2], 0)
        self.magFldCnt = SW.SRWLMagFldC(
            [self.field_1, self.field_2],
            [self.field1_params[3][0], self.field2_params[3][0]],
            [self.field1_params[3][1], self.field2_params[3][1]],
            [self.field1_params[3][2], self.field2_params[3][2]])

        # Define particle beam
        self.beam = SW.SRWLPartBeam()
        self.beam.Iavg = self.beam_params[4]
        self.beam.partStatMom1.x = self.cent_traj_params[0]
        self.beam.partStatMom1.y = self.cent_traj_params[1]
        self.beam.partStatMom1.z = self.cent_traj_params[2]
        self.beam.partStatMom1.xp = self.cent_traj_params[3]
        self.beam.partStatMom1.yp = self.cent_traj_params[4]
        self.beam.partStatMom1.gamma = self.cent_traj_params[5]

        # Set 2nd order moments
        self.beam.arStatMom2[0] = (self.beam_params[0]) ** 2  # <(x - x0)^2>
        self.beam.arStatMom2[2] = (self.beam_params[1]) ** 2  # <(x'-x'0)^2>
        self.beam.arStatMom2[3] = (self.beam_params[2]) ** 2  # <(y - y0)^2>
        self.beam.arStatMom2[5] = (self.beam_params[3]) ** 2  # <(y'-y'0)^2>

        # Define wavefront
        self.wavefront = SW.SRWLWfr()
        self.wavefront.allocate(self.wavefront_params[0][0],
                                self.wavefront_params[0][1],
                                self.wavefront_params[0][2])
        self.wavefront.mesh.xStart = self.wavefront_params[1][0]
        self.wavefront.mesh.xFin = self.wavefront_params[1][1]
        self.wavefront.mesh.yStart = self.wavefront_params[1][2]
        self.wavefront.mesh.yFin = self.wavefront_params[1][3]
        self.wavefront.mesh.eStart = self.wavefront_params[2][0]
        self.wavefront.mesh.eFin = self.wavefront_params[2][1]
        self.wavefront.mesh.zStart = self.wavefront_params[3]
        self.wavefront.partBeam = self.beam

    def prop_wavefront_lens(self, foc_len, shift=0, plot=False, down_sample=1):
        propagParLens = [0, 1, 1., 0, 0, 1.0, 1.0, 1.0, 1., 0, 0, 0]
        optLens = SW.SRWLOptL(_Fx=foc_len, _Fy=foc_len, _x=shift)
        optBL = SW.SRWLOptC([optLens], [propagParLens])

        print("Started lens propagation.")
        SW.srwl.PropagElecField(self.wavefront, optBL)
        print("Finished lens propagation.")

        if plot:
            mesh = SW.deepcopy(self.wavefront.mesh)
            intensity = SW.array('f', [0] * mesh.nx * mesh.ny)
            SW.srwl.CalcIntFromElecField(intensity, self.wavefront, 6, 1, 3,
                                         mesh.eStart, 0, 0)
            x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx) * 1000
            y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny) * 1000
            print(x.shape, y.shape)
            print(x[1] - x[0], y[1] - y[0])
            field = np.array(intensity).reshape((mesh.ny, mesh.nx))
            fig, ax = plt.subplots()
            ax.pcolor(x[::down_sample], y[::down_sample],
                      field[::down_sample, ::down_sample], shading='auto')
            return fig, ax

    def prop_wavefront_drift(self, drift, plot=False, down_sample=1):
        propag_par_drift = [1, 1, 1., 1, 0, 1., 1., 1., 1., 0, 0, 0]
        opt_drift = SW.SRWLOptD(drift)
        opt_con = SW.SRWLOptC([opt_drift], [propag_par_drift])

        print("Started drift propagation.")
        SW.srwl.PropagElecField(self.wavefront, opt_con)
        print("Finished drift propagation.")

        if plot:
            mesh = SW.deepcopy(self.wavefront.mesh)
            intensity = SW.array('f', [0] * mesh.nx * mesh.ny)
            SW.srwl.CalcIntFromElecField(intensity, self.wavefront, 6, 1, 3,
                                         mesh.eStart, 0, 0)
            print("here")
            x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx) * 1000
            y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny) * 1000
            print(x.shape, y.shape)
            print(x[1] - x[0], y[1] - y[0])
            field = np.array(intensity).reshape((mesh.ny, mesh.nx))
            fig, ax = plt.subplots()
            ax.pcolor(x[::down_sample], y[::down_sample],
                      field[::down_sample,::down_sample], shading='auto')
            return fig, ax

    def prop_wavefront_aper(self, diameter, shift=0, plot=False, down_sample=1):
        propag_par_aper = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
        opt_aper = SW.SRWLOptA('c', 'a', _Dx=diameter, _Dy=diameter, _x=shift,
                               _y=0)
        opt_con = SW.SRWLOptC([opt_aper], [propag_par_aper])

        print("Started aperture propagation.")
        SW.srwl.PropagElecField(self.wavefront, opt_con)
        print("Finished aperture propagation.")

        if plot:
            mesh = SW.deepcopy(self.wavefront.mesh)
            intensity = SW.array('f', [0] * mesh.nx * mesh.ny)
            SW.srwl.CalcIntFromElecField(intensity, self.wavefront, 6, 1, 3,
                                         mesh.eStart, 0, 0)
            x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx) * 1000
            y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny) * 1000
            print(x.shape, y.shape)
            print(x[1] - x[0], y[1] - y[0])
            field = np.array(intensity).reshape((mesh.ny, mesh.nx))
            fig, ax = plt.subplots()
            ax.pcolor(x[::down_sample], y[::down_sample],
                      field[::down_sample, ::down_sample], shading='auto')
            return fig, ax

    def print_mesh_parameters(self):
        print('Nx = ', self.wavefront.mesh.nx)
        print('Ny = ', self.wavefront.mesh.ny)
        print(
            'Input xWdith = yWidth =  ',
            self.wavefront.mesh.xFin - self.wavefront.mesh.xStart,
            ' [m]')

    def resize_wavefront(self, extent, res):
        """
        Make both wavefronts the same size, must have same z value
        :param extent: size of chip [xmin, xmax, ymin, ymax]
        :param res: pixel resolution [xres, yres]
        """

        range_factor_x = (extent[1] - extent[0]) / (self.wavefront.mesh.xFin
                                                - self.wavefront.mesh.xStart)
        res_factor_x = res[0] / (self.wavefront.mesh.nx * range_factor_x)
        range_factor_y = (extent[3] - extent[2]) / (self.wavefront.mesh.yFin
                                                - self.wavefront.mesh.yStart)
        res_factor_y = res[1] / (self.wavefront.mesh.ny * range_factor_y)
        params = [0, range_factor_x, res_factor_x, range_factor_y, res_factor_y,
                  0.5, 0.5]
        print(params)
        SW.srwl.ResizeElecField(self.wavefront, 'c', params)

    def calc_electric_field(self):
        """
        Simulates the electron trajectory and calculates initial wavefront
        """
        # initiate all required classes
        self.setup()
        # Calculating electric field (longest part)
        print("Started field calculation.")
        SW.srwl.CalcElecFieldSR(self.wavefront, 0, self.magFldCnt,
                                self.precision_params)
        print("Finished field calculation.")

    def calc_intensity(self):
        """
        Calculates the edge radiation stripe at detector.
        :return: np array of edge ration stripe
        """
        mesh = SW.deepcopy(self.wavefront.mesh)
        intensity = SW.array('f', [0] * mesh.nx * mesh.ny)
        SW.srwl.CalcIntFromElecField(intensity, self.wavefront, 6, 0, 3,
                                     mesh.eStart, 0, 0)
        x = np.linspace(mesh.xStart, mesh.xFin, mesh.nx)
        y = np.linspace(mesh.yStart, mesh.yFin, mesh.ny)
        intensity = np.array(intensity).reshape((mesh.ny, mesh.nx))
        return x, y, intensity

    def sim_central_traj(self, n_traj, end):
        """
        Simulates a single, central electron through the magnet
        :param n_traj: number of points along the trajectory
        :param end: end time of trajectory
        :return: x_coords, y_coords, z_coords of particle trajectory
        """
        part = SW.SRWLParticle()
        part.x = self.cent_traj_params[0]
        part.y = self.cent_traj_params[1]
        part.z = self.cent_traj_params[2]
        part.xp = self.cent_traj_params[3]
        part.yp = self.cent_traj_params[4]
        part.gamma = self.cent_traj_params[5]
        part.relE0 = 1
        part.nq = -1

        partTraj = SW.SRWLPrtTrj()
        partTraj.partInitCond = part
        partTraj.allocate(n_traj, True)
        partTraj.ctStart = 0
        partTraj.ctEnd = end
        traj = SW.srwl.CalcPartTraj(partTraj, self.magFldCnt, [1])
        x_coords = np.array(traj.arX)
        y_coords = np.array(traj.arY)
        z_coords = np.array(traj.arZ)
        xp_coords = np.array(traj.arXp)
        yp_coords = np.array(traj.arYp)
        zp_coords = np.array(traj.arZp)
        return np.array([x_coords, y_coords, z_coords,
                         xp_coords, yp_coords, zp_coords])

    def dump_wavefront(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.wavefront, file)

def sim_b2b3():
    gamma = 339.3 / 0.51099890221
    lamb = 550e-9
    L = 0.203274830142196
    dBL = 0.05
    B = 0.516733459336213
    energy = h * c / (lamb * qe)
    pos1 = [0, 0, 0]
    pos2 = [0, 0, 1.0334]
    wf_z = 0.719262584928902 + pos2[2]

    field1 = [B, L, dBL, pos1]
    field2 = [B, L, dBL, pos2]
    cent_traj_params = [-0.09318194668, 0, -1, 0.09280325784982, 0,
                        gamma]
    beam_params = [319.9632*1e-6, 23.3771*1e-6, 260.9830 * 1e-06, 30.0260 *
                   1e-6, 1e4]
    # wavefront_params = [[1, 500, 500], [-0.02, 0.02, -0.02, 0.02],
    #                     [energy, energy], wf_z]
    wavefront_params = [[1, 2**12, 2**12], [-0.02, 0.02, -0.02, 0.02],
                        [energy, energy], wf_z]
    # precision_params = [2, 0.05, -3, 1.5, 1000, 1, 1.0]
    precision_params = [2, 0.05, -3, 1.5, 2*1000, 1, 0]

    rad = EdgeRad(field1, field2, cent_traj_params, beam_params,
                 wavefront_params, precision_params)
    traj = rad.sim_central_traj(1000, 3)
    fig, ax = plt.subplots()
    ax.plot(traj[2, :], traj[0, :])
    plt.show()

    print("\n Initial/Beginning Mesh Parameters")
    rad.print_mesh_parameters()

    t0 = time.time()
    print('Starting SR Calculation at ' + time.ctime() + '. \n')
    rad.calc_electric_field()
    time_str = "Run time: %.2f seconds." % (time.time() - t0)
    print(time_str)


    # with open('./middle_robbie.pkl', 'wb') as file:
    #     pickle.dump(rad, file)

    rad.dump_wavefront('Robbie_After_SR_Calc_4096.pkl')

    print("\n Mesh Parameters After SR Calc")
    rad.print_mesh_parameters()

    t0 = time.time()
    print("Starting Wavefront Propagation")
    rad.prop_wavefront_aper(0.04)
    rad.prop_wavefront_lens(0.105)
    rad.prop_wavefront_drift(0.105)
    time_str = "Run time: %.2f seconds." % (time.time() - t0)
    print(time_str)

    print("\n Mesh Parameters After Wavefront Propagation")
    rad.print_mesh_parameters()

    rad.dump_wavefront('Robbie_After_WF_Prop_4096.pkl')

    pixels = [1000, 1000]
    chip_size = np.array([-1.8075, 1.8075, -2.4225, 2.4225]) / 1000
    rad.resize_wavefront(chip_size, pixels)

    x, y, I = rad.calc_intensity()
    fig, ax = plt.subplots()
    ax.plot(traj[2, :], traj[0, :])
    ax.set_xlabel("z (m)")
    ax.set_ylabel("x (m)")

    fig, ax = plt.subplots()
    pcol = ax.pcolormesh(I)
    fig.colorbar(pcol)
    plt.show()

if __name__ == "__main__":
    sim_b2b3()