# s file contains functions associated with Fourier Optics in python!

import math
import numpy
from scipy import pi as PI
from scipy import interpolate as interp
from scipy.signal import czt, CZT

def nextpow2(L):
	N = math.log(L)/math.log(2)
	NN = math.ceil(N)
	return int(NN)

# xgv and ygv are the vectors used to generate the 2D grids via meshgrid.
def FO_TwoD_FFT(xgv, ygv, func_in):
	delta_x = xgv[1]-xgv[0]
	delta_y = ygv[1]-ygv[0]
	L = max(func_in.shape) # Make the fft symmetric around the largest dimension.
	N = 2**nextpow2(L)
	# s = [N,N]
	Y = numpy.fft.fft2(func_in)
	YY = numpy.fft.fftshift(Y)
	x_max = 1.0/(2.0*delta_x)
	y_max = 1.0/(2.0*delta_y)
	x_freq = numpy.linspace(-N/2,N/2-1,N)*x_max*2/N
	y_freq = numpy.linspace(-N/2,N/2-1,N)*y_max*2/N
	return x_freq, y_freq, YY
	
# This file calculates the diffraction propagation integral by
# performing the transform of the convolution, see Goodman p. 67 and p. 72.
# The SM indicates Spectral Method.  It uses the exact propagation transfer
# function Goodman Eq. 4-21.

# xgv and ygv are define as above.
# ZZ is the laser profile you are attempting to propagate.  Including phase!
# distance is the distance you wish to propagate
# lambda_0 is the wavelength of the propgating radiation.

def FO_TwoD_exact_SM(xgv,ygv,ZZ,distance,lambda_0):
	kappa_0 = 2.0*PI/lambda_0
	[fx,fy,FF] = FO_TwoD_FFT(xgv,ygv,ZZ) # This performs the FFT.
	# Now form a meshgrid to perform the diffraction calculation.
	[XX,YY] = numpy.meshgrid(fx,fy)
	# Perform the diffraction calculation
	TR_1 = numpy.exp(1j*kappa_0*distance*numpy.sqrt(1-(lambda_0*XX)**2-(lambda_0*YY)**2))
	TR_2 = (numpy.sqrt((lambda_0*XX)**2+(lambda_0*YY)**2) < 1.0)
	TR = TR_1*TR_2
	FF = numpy.fft.ifft2(numpy.fft.ifftshift(FF*TR))
	return FF

def FO_OneD_FFT(xgv, ZZ):
	delta_x = xgv[2] - xgv[1]
	L = max(ZZ.shape)
	N = 2**nextpow2(L)
	Y = numpy.fft.fft(ZZ, N)
	Y = numpy.fft.fftshift(Y)
	x_freq = numpy.linspace(-N/2, N/2-1, N) / delta_x / (N-1)
	return x_freq, Y
	
# Thus function propagates the input beam a certain distance, at user supplied step size and
# returns a 2D array of Nx X Nz so you can see the profile change as it propagates.
# z_start is the start point of the propagation.  It allows you to jump from the input laser position.
# z_length is the length to propagate over.
def FO_prop(xgv,ygv,ZZ,z_start,z_length,lambda_0,Nz):
	print('Propagating the beam', z_length, 'meters in', Nz, 'steps')
	[Ny, Nx] = ZZ.shape
	delta_z = z_length / Nz # Compute delta_z
	prop_array = numpy.zeros((Nx,Nz)) # Allocate the array for the data
	# Jump to the first point desired
	FF_1 = FO_TwoD_exact_SM(xgv,ygv,ZZ,z_start,lambda_0)
	prop_array[:,0] = abs(FF_1[Ny/2-1,:])
	# Step through the propagation.
	for i in range(1,Nz):
		FF_1 = FO_TwoD_exact_SM(xgv,ygv,FF_1,delta_z,lambda_0)
		prop_array[:,i] = abs(FF_1[Ny/2-1,:])
		if( i % 10 == 0):
			print('On propagation step', i, 'of', Nz)
	return prop_array
	
# This function adds the phase due to an axilens.
# XX and YY is the physical grid on which to apply the axilens.
# ZZ is the input field distribution.
# kappa_0 is the laser wavenumber.
# f0 is the focal length of the axilens.  The location of the start of the line focus.
# Dz is the length of the axilens line focus.
# R is the size of the flat top beam input.

def FO_axilens(XX,YY,ZZ,kappa_0,f0,Dz,R):
	axilens_phase = numpy.exp(-1j*kappa_0*(R**2)/(2*Dz)*numpy.log( f0+Dz/(R**2)*(XX**2+YY**2) ))
	ZZ = axilens_phase*ZZ
	return ZZ
	
def FO_axicon(XX,YY,ZZ,kappa_0,phi,n):
	axicon_phase = numpy.exp(-1j*kappa_0*numpy.sqrt(XX**2+YY**2)*(n-1)*numpy.tan(phi))
	ZZ = axicon_phase*ZZ
	return ZZ


def FO_lens(XX,YY,ZZ,kappa_0,f0):
	lens_phase = numpy.exp((-1j*kappa_0/2/f0)*(XX**2+YY**2))
	ZZ_1 = lens_phase * ZZ
	return ZZ_1


def FO_OneD_CZT(xgv, ZZ, M=None, x_start=None, x_stop=None):
	'''
	Use the Chirped Z-Transform to be able to scale the output space of a
	Fourier transform so that you aren't dependent on the input grid spacing
	matching the output grid spacing.

	:param xgv: Input vector that defines the physical space to perform the
	CZT on
	:param ZZ: The intensity of signal at each physical point of xgv
	:param M: Number of points in the output of the CZT
	:param x_start: starting x location to map the output to via the CZT
	:param x_stop: ending x location to map the output to via CZT
	:return:
	'''

	if M is None:
		M = xgv.size

	if x_start is None:
		x_start = xgv[0]

	if x_stop is None:
		x_stop = xgv[-1]

	x_range = xgv[-1] - xgv[0]

	A = numpy.exp(2j * numpy.pi * (x_start) / x_range)
	W = numpy.exp(-2j * numpy.pi * (x_stop-x_start) / x_range / M)

	temp = CZT(n = xgv.size, m = M, w = W, a = A)
	y = temp(ZZ)
	# y = czt(ZZ, M, W, A)

	x_freq = numpy.linspace(0, M-1, M) / (M-1) / (xgv[1] - xgv[0]) \
			* (x_stop - x_start) / x_range \
			 + (x_start/x_range)/(xgv[1] - xgv[0]) * M / (M-1)
	return x_freq, y

def FO_TwoD_CZT(xgv, ZZ, M = None, x_start = None, x_stop = None):
	"""
	Perform the 2D CZT on a square input matrix. This function is used to
	shrink the size of the output window of an FFT and/or change the density
	of points in the output window.

	:param xgv: Input vector that defines the physical space of the data
	:param ZZ: 2D matrix (must be square) to CZT
	:param M: New number of points in the output window.
	:param x_start: Start of the output window frequency space
	:param x_stop: End of the output window frequency space
	:return: frequency space vector (analog of xgv), 2D transformed matrix
	"""
	if M is None:
		M = xgv.size

	if x_start is None:
		x_start = xgv[0]

	if x_stop is None:
		x_stop = xgv[-1]

	print('x_start is ', str(x_start))
	print('x_stop is ', str(x_stop))
	x_range = xgv[-1] - xgv[0]

	A = numpy.exp(2j * numpy.pi * (x_start) / x_range)
	W = numpy.exp(-2j * numpy.pi * (x_stop - x_start) / x_range / M)
	temp = CZT(n=xgv.size, m=M, w=W, a=A)

	BB = temp(ZZ, axis=0)
	BB = temp(BB, axis=1)

	x_freq = numpy.linspace(0, M-1, M) / (M-1) / (xgv[1] - xgv[0]) \
			* (x_stop - x_start) / x_range \
			 + (x_start/x_range)/(xgv[1] - xgv[0]) * M / (M-1)

	return x_freq, BB

def rearrange(x, xn, x0, fn, f0):
	return (fn - f0) * (x - x0) / (xn - x0) + f0


def FO_TwoD_exact_SM_CZT(X, ZZ, distance, lambda_0, M=None, x_out_1=None,
						 x_out_2=None):
	Nx = X.size

	if M is None:
		M = Nx

	if x_out_1 is None:
		x_out_1 = X[0]

	if x_out_2 is None:
		x_out_2 = X[-1]

	# Perform the FFT to get into the Fourier Plane
	x_range = X[-1] - X[0]
	x_start = X[0]
	x_stop = X[-1]

	A = numpy.exp(2j * numpy.pi * (x_start) / x_range)
	W = numpy.exp(-2j * numpy.pi * (x_stop - x_start) / x_range / Nx)
	temp = CZT(n=X.size, m=Nx, w=W, a=A)
	BB = temp(ZZ, axis=0) * (X[1] - X[0])
	BB = temp(BB, axis=1) * (X[1] - X[0])
	# fx = numpy.linspace(-Nx / 2, Nx / 2, Nx) / (X[1] - X[0]) / Nx
	fx = numpy.linspace(-Nx / 2, Nx / 2 - 1, Nx) / (X[1] - X[0]) / Nx

	# Now form a meshgrid to perform the diffraction calculation.
	[XX,YY] = numpy.meshgrid(fx, fx)

	# Perform the propagation diffraction calculation
	kappa_0 = 2.0 * numpy.pi / lambda_0
	TR_1 = numpy.exp(1j*kappa_0*distance*numpy.sqrt(1-(lambda_0*XX)**2-(
			lambda_0*YY)**2))
	TR_2 = (numpy.sqrt((lambda_0*XX)**2+(lambda_0*YY)**2) <= 1.0)
	BB = BB*TR_1*TR_2

	# iFFT back to real space, this is where the size change (if any) occurs
	# IFFT the data
	# M = 2 * Nx
	x_range = X[-1] - X[0]
	x_start = x_out_1
	x_stop  = x_out_2

	A = numpy.exp(2j * numpy.pi * (x_start - x_range / 2) / x_range +
			   (1.0) * 1j * numpy.pi / Nx)  # This Nx is correct! Do not
	# change it!
	W = numpy.exp(-2j * numpy.pi * (x_stop - x_start) / x_range / (M - 1))
	temp = CZT(n=Nx, m=M, w=W, a=A)
	BB = temp(BB, axis=0)
	BB = temp(BB, axis=1)
	X2 = numpy.linspace(x_start, x_stop, M)


	return X2, BB