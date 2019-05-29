import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import time

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

def main():
	Nx = 200;
	L = 1; # [m]
	dx = L/Nx; # [m]
	x = np.linspace(0,L,Nx)
	gamma = 1.4;

	uold = np.loadtxt("cloutput.dat");
	rho = uold[:,0]
	U = uold[:,1]/rho
	P = (uold[:,2]-uold[:,1]*U/2)*(gamma-1)
	[xex,yex] = SodsExact(1.0,0,1.0,0.125,0,0.1,0.1)
	Rhoex = yex[0,:]
	Uex = yex[1,:]/yex[0,:]
	Pex = (yex[2,:]-yex[1,:]*Uex/2)*(gamma-1)
	plt.subplot(4,1,1)
	plt.plot(x,rho,'bx')
	plt.plot(xex,Rhoex,'r--')
	plt.ylim([np.min(Rhoex),np.max(Rhoex)]);
	#plt.ylim([0,0.5]);
	#plt.xlim([0.55, 0.70])
	plt.ylabel('Density')
	plt.subplot(4,1,2)
	plt.plot(x,U,'bx')
	plt.plot(xex,Uex,'r--')
	plt.ylim([np.min(Uex),np.max(Uex)]);
	#plt.ylim([-0.05,1.0]);
	#plt.xlim([0.55, 0.70])
	plt.ylabel('Velocity')
	plt.subplot(4,1,3)
	plt.plot(x,P,'bx')
	plt.plot(xex,Pex,'r--')
	plt.ylim([np.min(Pex),np.max(Pex)]);
	#plt.xlim([0.55, 0.70])
	#plt.ylim([0,0.5]);
	plt.ylabel('Pressure')
	plt.subplot(4,1,4)
	plt.plot(x,uold[:,3]/uold[:,0],'b-');
	plt.plot(x,uold[:,4]/uold[:,0],'r-');
	#plt.xlim([0.55, 0.70])
	plt.ylabel('Species Fraction')
	plt.show()


if __name__ == "__main__":
	main();
