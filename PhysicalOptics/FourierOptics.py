# s file contains functions associated with Fourier Optics in python!

import math
import numpy
from scipy import pi as PI
from scipy import interpolate as interp

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
	s = [N,N]
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
	
# Thus function propagates the input beam a certain distance, at user supplied step size and
# returns a 2D array of Nx X Nz so you can see the profile change as it propagates.
# z_start is the start point of the propagation.  It allows you to jump from the input laser position.
# z_length is the length to propagate over.
def FO_prop(xgv,ygv,ZZ,z_start,z_length,lambda_0,Nz):
	print 'Propagating the beam', z_length, 'meters in', Nz, 'steps'
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
			print 'On propagation step', i, 'of', Nz
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
	axicon_phase = numpy.exp(-1j*kappa_0*numpy.sqrt(XX**2+YY**2)*(n-1)*numpy.tan(phi));
	ZZ = axicon_phase*ZZ;
	return ZZ

# FO_interp is designed so that vorpal can make a request based on x,y,z,t instead
# of using grid points.  It does this by generating a grid!
# xgv and ygv are the grid spacings used to generate the output from the Fourier
# program, which produces FF.
# It returns an interpolation function that takes in (Y,Z) that will allow
# vorpal to get the intensity distribution and the transverse coordinates (Y,Z).
# (Y,Z) are NOT supplied to this function.
# The function 
def FO_interp(xgv,ygv,FF):
	# Make sure the input is real.
	if numpy.iscomplex(FF[1,1]): #If the [1,1] element is complex, the whole thing is.  And vice-versa.
		FF = abs(FF)
	interp_f = interp.interp2d(xgv,ygv,abs(FF))
	return interp_f
	





