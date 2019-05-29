import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt
# import time

epweno = 1e-6; # "Epsilon" value, meant to avoid divison by zero

def machEqn(m,a,g,prat):
	gpm = (g+1)/(g-1)
	powerterm = prat*(2*g/(g+1)*np.power(m,2.0)-1.0/gpm)
	ret = a*gpm*(1.0-np.power(powerterm,(g-1)/(2*g)))-m+1.0/m;
	return ret;

def pressureEqn(p2p1,p4p1,g,a1a4):
	num = (g-1)*a1a4*(p2p1-1);
	den = np.sqrt(2*g*(2*g+(g+1)*(p2p1-1)))
	powterm = np.power(1-num/den,-2*g/(g-1))
	return p2p1*powterm-p4p1;

def SodsExact(rho4,u4,p4,rho1,u1,p1,tend):
	# (4) (expansion) (3) (contact surface) (2) (shock) (1)
	gamma=1.4;
	x0 = 0.5;
	gpm = (gamma+1)/(gamma-1);

	p4p1 = p4/p1; # Left/right pressure ratio
	a4 = np.sqrt(gamma*p4/rho4); # Left speed of sound
	a1 = np.sqrt(gamma*p1/rho1); # Right speed of sound
	a4a1 = a4/a1; # Ratio of the speed of sounds
	p2p1 = sopt.fsolve(pressureEqn,1.0,args=(p4p1,gamma,1.0/a4a1)) # This gives the pressure ratio of the shock
	ushock = a1*np.sqrt((gamma+1)/(2*gamma)*(p2p1-1)+1)
	# Calculate region between shock and contact surface
	p2 = p2p1*p1;
	rho2rho1 = (1+gpm*p2p1)/(gpm+p2p1); rho2 = rho2rho1*rho1;
	u2 = a1/gamma*(p2p1-1)*np.sqrt(((2*gamma)/(gamma+1))/(p2p1+1.0/gpm));
	# Now calculate the expansion wave
	p3p4 = p2p1/p4p1; p3 = p3p4*p4;
	rho3rho4 = np.power(p3p4,1/gamma); rho3 = rho3rho4*rho4;
	u3 = u2; # Velocity is unchanged across the contact surface
	# Now need where each of these regions are based on time
	x1 = x0 - a4*tend; # Location of the left part of the expansion fan
	a3 = np.sqrt(gamma*p3/rho3)
	x2 = x0 + (u3-a3)*tend; # Location of the right part of the expansion fan
	x3 = x0 + u2*tend; # Location of the contact surface
	x4 = x0 + ushock*tend; # Location of the shock wave
	# # Now calculate behavior in expansion region
	xpts = np.linspace(0,1,1000);
	ypts = np.zeros((3,len(xpts)))
	for i in np.arange(0,len(xpts)):
		x = xpts[i]
		if (x <= x1): # In the left region
			ypts[0,i] = rho4;
			ypts[1,i] = rho4*u4;
			ypts[2,i] = p4/(gamma-1)+rho4*np.power(u4,2.0)/2;
		elif (x > x1 and x <= x2): # In the expansion fan
			u = 2/(gamma+1)*(a4+(x-x0)/tend)
			p = p4*np.power(1-(gamma-1)/2*(u/a4),(2*gamma)/(gamma-1))
			rho = rho4*np.power(1-(gamma-1)/2*(u/a4),2/(gamma-1))
			ypts[0,i] = rho;
			ypts[1,i] = rho*u;
			ypts[2,i] = p/(gamma-1)+rho*np.power(u,2.0)/2;
		elif (x > x2 and x <= x3): # Between the expansion and contact surface
			ypts[0,i] = rho3;
			ypts[1,i] = rho3*u3;
			ypts[2,i] = p3/(gamma-1)+rho3*np.power(u3,2.0)/2;
		elif (x > x3 and x <= x4): # Between the contact surface and the shock
			ypts[0,i] = rho2;
			ypts[1,i] = rho2*u2;
			ypts[2,i] = p2/(gamma-1)+rho2*np.power(u2,2.0)/2;
		elif (x > x4): # The right region
			ypts[0,i] = rho1;
			ypts[1,i] = rho1*u1;
			ypts[2,i] = p1/(gamma-1)+rho1*np.power(u1,2.0)/2;
	return [xpts,ypts];

# BCs are
# drho/dx (x = 0 = L) = 0
# rho*U = 0 (because U = 0) at the edges
# E at the boundaries comes from dp/dx = 0 (Perfect reflector)
# Since E = P/(gamma-1)+rho*u^2/2, P = (E-rho*u^2/2)*(gamma-1), dp/dx = (de/dx-drho/dx*u^2/2-rho*u*du/dx)*(gamma-1) = 0, so de/dx = rho*u*du/dx, but u = 0, so de/dx = 0
# s1, s2, should be equal to their respective initial species fractions
def setBoundaryConditions(y,opts=None):
	# Left hand cells will use a forward difference
	# From: https://en.wikipedia.org/wiki/Finite_difference_coefficient#Forward_finite_difference
	#	−137/60	5	−5	10/3	−5/4	1/5
	# dLdx = ((-137.0/60.0*y[:,2]+5.0*y[:,3]-5.0*y[:,4]+10.0/3.0*y[:,5]-5.0/4.0*y[:,6]+1.0/5.0*y[:,7]))/dx = 0
	# Therefore, y[:,2] = -60.0/137.0*(-1.0/5.0*y[:,7]+5.0/4.0*y[:,6]-10.0/3.0*y[:,5]+5.0*y[:,4]-5.0*y[:,3])
	ynew = y
	ynew[:,2] = -60.0/137.0*(-1.0/5.0*y[:,7]+5.0/4.0*y[:,6]-10.0/3.0*y[:,5]+5.0*y[:,4]-5.0*y[:,3])
	ynew[3,2] = y[3,2]
	ynew[1,2] = 0.0;
	ynew[:,1] = ynew[:,2]
	ynew[:,0] = ynew[:,2]
	# Same idea for the backwards differences, except the signs of the coefficients are flipped
	ynew[:,-3] = 60.0/137.0*(1.0/5.0*y[:,-8]-5.0/4.0*y[:,-7]+10.0/3.0*y[:,-6]-5.0*y[:,-5]+5.0*y[:,-4])
	ynew[3,-3] = y[3,-3]
	ynew[1,-3] = 0.0;
	ynew[:,-2] = ynew[:,-3]
	ynew[:,-1] = ynew[:,-3]
	# Note that this just does a "zero-th order" extrapolation in to the ghost cells, and so does not conserve mass when characteristics start hitting the boundaries
	# http://physics.princeton.edu/~fpretori/Burgers/Boundary.html <- useful link for extrapolation
	return ynew;

def advanceTimestepRK3(to,dt,y,RHS,opts=None):
	f1 = RHS(y,to,opts);
	y2 = y-dt*f1;
	y2 = setBoundaryConditions(y2,opts)
	f2 = RHS(y2,to,opts);
	y3 = 0.75*y+0.25*(y2-dt*f2)
	y3 = setBoundaryConditions(y3,opts)
	f3 = RHS(y3,to,opts);
	y4 = (y+2.*(y3-dt*f3))/3.0
	y4 = setBoundaryConditions(y4,opts)
	return y4;

Runiv = 8.314; # Universal gas constant, [J/K-mol]
# From https://encyclopedia.airliquide.com/
mu1 = 0.0289647; mu2 = 0.14606; # Mollecular weights of air and sf6 [kg/mol]
R1 = Runiv/mu1; R2 = Runiv/mu2; # Specific gas constants
cp1 = 1005.9; cp2 = 627.83; # Cp values for air and sf6
cv1 = 717.09; cv2 = 566.95; # Cv values for air and sf6

def getFlux(u):
	ret = np.zeros(u.shape)
	rho = u[0]
	v = u[1]/rho;
	s1 = u[3]/rho;
	s2 = 1-s1;
	gammaeff = (cp1*s1+cp2*s2)/(cv1*s1+cv2*s2); # Calculate an effective gamma
	P = (u[2]-rho*np.power(v,2.0)/2.0)*(gammaeff-1) # Calculate pressure from ch10.pdf, eq 10.2
	ret[0] = u[1]
	ret[1] = rho*np.power(v,2.0)+P;
	ret[2] = (u[2]+P)*v;
	ret[3] = v*u[3];
	return ret;

# This is an attempt to implement the WENO scheme for Euler's equations. We're going to start with the more simple
# Component-wise treatment, which will still have some oscillations but is simpler to do. This is a good place to start.
# Note that this is the finite volume formulation

def WENOFVCompWise(u,t,opts):
	ushp = u.shape
	ret = np.zeros(ushp)
	LF = np.zeros(ushp)
	npts = ushp[1]
	r = 3; # Order of the method = 2r-1
	# The velocity component is what's getting hosed up here (i=1).
	I = np.arange(r-1,npts-r)
	# Start with v = u^-_i+1/2
	vmm = u[:,I-2]
	vm  = u[:,I-1]
	v   = u[:,I]
	vp  = u[:,I+1]
	vpp = u[:,I+2]
	# Calculate vr (v_i+1/2) and vl (v_i-1/2) for each stencil (2.10)
	p0 = (2.*v + 5.*vp - vpp)/6.;
	p1 = (-vm + 5.*v+2.*vp)/6.;
	p2 = (2.*vmm-7.*vm+11.*v)/6.;

	# Now need to form the weights. Start with beta_r (2.63)
	beta0 = 13./12.*np.power(v-2.*vp+vpp,2.0)+1./4.*np.power(3.*v-4.*vp+vpp,2.0)
	beta1 = 13./12.*np.power(vm-2.*v+vp,2.0)+1./4.*np.power(vm-vp,2.0)
	beta2 = 13./12.*np.power(vmm-2.*vm+v,2.0)+1./4.*np.power(vmm-4.*vm+3.*v,2.0)
	# Now given beta, form alpha from d and beta (2.59)
	alpha0 = (3.0/10.0)/np.power(epweno+beta0,2.0); alpha1 = (6.0/10.0)/np.power(epweno+beta1,2.0); alpha2 = (1.0/10.0)/np.power(epweno+beta2,2.0); # alpha
	alphasum = alpha0+alpha1+alpha2;
	# Now calculate omega (2.58)
	# We seem to be off by one index on the calculation of the tilde quantities versus the matlab (vminus)
	omega0 = alpha0/alphasum; omega1 = alpha1/alphasum; omega2 = alpha2/alphasum; # omega
	vplus = omega0*p0+omega1*p1+omega2*p2; # v-_i+1/2 (2.64)

	# Now with v = u^+_i-1/2. Note that we need to shift this forward by 1 so that we're calculating at v_i+1/2 (Because i-1/2 of the next cell is i+1/2 of this cell)
	vmm = u[:,I-1]
	vm  = u[:,I]
	v   = u[:,I+1]
	vp  = u[:,I+2]
	vpp = u[:,I+3]
	# Calculate vr (v_i+1/2) and vl (v_i-1/2) for each stencil (2.10)
	p0 = (11.*v - 7.*vp +2.*vpp)/6.;
	p1 = (2.*vm + 5.*v-vp)/6.;
	p2 = (-vmm+5.*vm+2.*v)/6.;

	# Now need to form the weights. Start with beta_r (2.63)
	beta0 = 13./12.*np.power(v-2.*vp+vpp,2.0)+1./4.*np.power(3.*v-4.*vp+vpp,2.0)
	beta1 = 13./12.*np.power(vm-2.*v+vp,2.0)+1./4.*np.power(vm-vp,2.0)
	beta2 = 13./12.*np.power(vmm-2.*vm+v,2.0)+1./4.*np.power(vmm-4.*vm+3.*v,2.0)
	# Now given beta, form alpha from d and beta (2.59)
	alpha0 = (1.0/10.0)/np.power(epweno+beta0,2.0); alpha1 = (6.0/10.0)/np.power(epweno+beta1,2.0); alpha2 = (3.0/10.0)/np.power(epweno+beta2,2.0); # alpha
	alphasum = alpha0+alpha1+alpha2;
	# Now calculate omega (2.58)
	omega0 = alpha0/alphasum; omega1 = alpha1/alphasum; omega2 = alpha2/alphasum; # omega
	vminus = omega0*p0+omega1*p1+omega2*p2; # v+_i-1/2
	# # Now calculate flux
	LF[:,I] = 0.5*(getFlux(vplus)+getFlux(vminus)-np.abs(opts['lam'])*(vminus-vplus))
	ret[:,I] = (LF[:,I]-LF[:,I-1])/opts['dx']
	ret[:,2] = ret[:,2] - LF[:,2]/opts['dx']
	ret[:,-2] = ret[:,-2]+LF[:,-2]/opts['dx']
	return ret;

def main():
	Nx = 200;
	L = 1; # [m]
	dx = L/Nx; # [m]
	Nx = Nx+4; # Add in ghost cells
	x = np.linspace(0-2*dx,L+2*dx,Nx)

	cfl = 0.55;
	tfinal = 0.2; # [s]
	nplot = 10; # Show a plot every N steps

	uold = np.zeros((4,Nx));
	# Set up conditions
	# We will normalize everything by the first gas properties at STP
	Tatm = 293.0; # [K], approx 70 F
	Patm = 101300.0; # [Pa], Atmospheric Pressure
	Rhoatm = Patm/(Tatm*R1); # Density of the first gas at STP
	for i in range(0,int(Nx/2)+1):
		P = 8.0*Patm;
		c1 = 1.0; c2 = 1.0-c1;
		R = c1*R1+c2*R2;
		G = (c1*cp1+c2*cp2)/(c1*cv1+c2*cv2)
		rho = P/(R*Tatm);
		#rho = 1.0
		uold[0,i] = rho/Rhoatm; # Rho
		uold[1,i] = 0.; # Rho*U
		uold[2,i] = (P/Patm)/(G-1.0) # E
		uold[3,i] = c1*uold[0,i]
	for i in range(int(Nx/2)+1,int(Nx)):
		P = 1.0*Patm;
		c1 = 1.0; c2 = 1.0-c1;
		R = c1*R1+c2*R2;
		G = (c1*cp1+c2*cp2)/(c1*cv1+c2*cv2)
		rho = P/(R*Tatm);
		#rho = 1.0
		uold[0,i] = rho/Rhoatm; # Rho
		uold[1,i] = 0.; # Rho*U
		uold[2,i] = (P/Patm)/(G-1.0) # E
		uold[3,i] = c1*uold[0,i]

	s1 = uold[3,:]/uold[0,:]; # s2 = W[4,:]/W[0,:]
	s2 = 1-s1;
	G = (s1*cp1+s2*cp2)/(s1*cv1+s2*cv2);
	a0 = np.sqrt(G*uold[2,:]*(G-1)/(uold[0,:]))
	lam = np.max(a0);
	dt = cfl*dx/lam;
	Nt = np.ceil(tfinal/dt)
	Ts = np.arange(0,Nt,dtype=int)
	[xex,yex] = SodsExact(uold[0,0],uold[1,0],uold[2,0]*(cp1/cv1-1.0),uold[0,-1],uold[1,-1],uold[2,-1]*(cp1/cv1-1.0),tfinal)

	for n in Ts:
		if (n%nPlot == 0):
			s1 = uold[3,:]/uold[0,:]; # s2 = uold[4,:]/uold[0,:]
			s2 = 1-s1;
			G = (s1*cp1+s2*cp2)/(s1*cv1+s2*cv2);
			U = uold[1,:]/(uold[0,:])
			P = (uold[2,:]-uold[1,:]*U/2)*(G-1)
			c = np.sqrt(G*P/uold[0,:])
			Jp = U + 2*c/(G-1)
			Jm = U - 2*c/(G-1)
			s1 = uold[3,:]/uold[0,:]; #s2 = uold[4,:]/uold[0,:]
			s2 = 1-s1;
			G = (s1*cp1+s2*cp2)/(s1*cv1+s2*cv2);
			U = uold[1,:]/(uold[0,:])
			P = (uold[2,:]-uold[1,:]*U/2)*(G-1)
			plt.figure()
			plt.subplot(4,1,1)
			plt.title("t={:0.4f}".format(n*dt))
			plt.plot(x,uold[0,:],'b-')
			# plt.plot(xex,Rhoex,'r--')
			plt.ylabel('Density')
			plt.xlim([0,L])
			plt.subplot(4,1,2)
			plt.plot(x,U,'b-')
			# plt.plot(xex,Uex,'r--')
			plt.ylabel('Velocity')
			plt.xlim([0,L])
			plt.ylim([-0.3, 0.3])
			plt.subplot(4,1,3)
			plt.plot(x,P,'b-')
			plt.xlim([0,L])
			# plt.plot(xex,Pex,'r--')
			plt.ylabel('Pressure')
			plt.subplot(4,1,4)
			plt.plot(x,s1,'b-');
			plt.plot(x,s2,'r-');
			plt.xlim([0,L])
			plt.ylabel('Species Fraction')
			plt.xlim([0,L])
			plt.show()
		unew = advanceTimestepRK3(n*dt,dt,uold,WENOFVCompWise,{'dx':dx,'lam':lam})
		uold = unew
	print("Done.")

	s1 = uold[3,:]/uold[0,:]; #s2 = uold[4,:]/uold[0,:]
	s2 = 1-s1;
	G = (s1*cp1+s2*cp2)/(s1*cv1+s2*cv2);
	U = uold[1,:]/(uold[0,:])
	P = (uold[2,:]-uold[1,:]*U/2)*(G-1)
	Rhoex = yex[0,:]
	Uex = yex[1,:]/yex[0,:]
	Pex = (yex[2,:]-yex[1,:]*Uex/2)*(1.4-1)
	plt.figure(1)
	plt.subplot(4,1,1)
	plt.title("t={:0.4f}".format(Nt*dt))
	plt.plot(x,uold[0,:],'b-')
	plt.plot(xex,Rhoex,'r--')
	plt.ylabel('Density')
	plt.xlim([0,L])
	plt.subplot(4,1,2)
	plt.plot(x,U,'b-')
	plt.plot(xex,Uex,'r--')
	plt.ylabel('Velocity')
	plt.xlim([0,L])
	plt.subplot(4,1,3)
	plt.plot(x,P,'b-')
	plt.xlim([0,L])
	plt.plot(xex,Pex,'r--')
	plt.ylabel('Pressure')
	plt.subplot(4,1,4)
	plt.plot(x,s1,'b-');
	plt.plot(x,s2,'r-');
	plt.xlim([0,L])
	plt.ylabel('Species Fraction')
	plt.show();


if __name__ == "__main__":
	main();
