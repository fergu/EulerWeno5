// Define the number of components (euqations) here. This will update all the iterations in the program so we don't risk missing one.
// #define NComponents 5

// Basically just redefining/abstracting the double vector type.
// That way it is trivial if we need to move to a longer vector later.
typedef double8 doubleVector;
// Current definition:
// Component 0 (s0): Rho
#define rho s0
// Component 1 (s1): RhoU
#define rhoU s1
// Component 2 (s2): E
#define E s2
// Component 3 (s3): rhoY1
#define rhoY1 s3
// Component 4 (s4): rhoY2
#define rhoY2 s4

// typedef struct _cellData {
// 	double rho;		// Mass
// 	double rhoU;	// Momentum
// 	double E;			// Total Internal Energy
// 	double rhoY1; // Species 1
// 	double rhoY2; // Species 2
// } _cellData;

// static _cellData getFlux(_cellData data) {
// 	_cellData retData;
// 	double v = data.rhoU/data.rho;
// 	double p = (data.E-data.rho*pow(v,2)/2)*(1.4-1);
// 	retData.rho = data.rhoU;
// 	retData.rhoU = data.rho*pow(v,2)+p;
// 	retData.E = (data.E+p)*v;
// 	retData.rhoY1 = v*data.rhoY1;
// 	retData.rhoY2 = v*data.rhoY2;
// 	return retData;
// }

static doubleVector getFlux(doubleVector data) {
	doubleVector retData;
	double v = data.rhoU/data.rho;
	double p = (data.E-data.rho*pow(v,2.0)/2.0)*(1.4-1.0);
	retData.rho = data.rhoU;
	retData.rhoU = data.rho*pow(v,2.0)+p;
	retData.E = (data.E+p)*v;
	retData.rhoY1 = v*data.rhoY1;
	retData.rhoY2 = v*data.rhoY2;
	return retData;
}

// static doubleVector getCellData(doubleVector data, int i) {
//   if (i == 0) {
//     return data.rho;
//   } else if (i==1) {
//     return data.rhoU;
//   } else if (i==2) {
//     return data.E;
//   } else if (i==3) {
// 		return data.rhoY1;
// 	} else if (i==4) {
// 		return data.rhoY2;
// 	} else {
// 		return 0;
// 	}
// }
//
// static void assignCellData(doubleVector *data, int i, double val) {
//   if (i == 0) {
//     data->rho = val;
//   } else if (i==1) {
//     data->rhoU = val;
//   } else if (i==2) {
//     data->E = val;
//   } else if (i==3) {
// 		data->rhoY1 = val;
// 	} else if (i==4) {
// 		data->rhoY2 = val;
// 	}
// }

kernel void wenoCalculateFlux(global doubleVector *state, constant int *nCells, constant int *nGhostCells, constant double *lambda, global doubleVector *flux) {
	int gid = get_global_id(0);
	//doubleVector vplus, vminus;
	double epweno = 1e-6; // "Epsilon" term to prevent division by zero

	if (gid < nGhostCells[0] || gid >= nCells[0]-nGhostCells[0]) {
		flux[gid] = 0.0;
		return;
	}

	doubleVector vmm = state[gid-2];
	doubleVector vm = state[gid-1];
	doubleVector v = state[gid];
	doubleVector vp = state[gid+1];
	doubleVector vpp = state[gid+2];
	// Calculate the polynomials
	doubleVector p0 = (2.0*v + 5.0*vp - vpp)/6.0;
	doubleVector p1 = (-vm+5.0*v+2.0*vp)/6.0;
	doubleVector p2 = (2.0*vmm-7.0*vm+11.0*v)/6.0;
	//Now calculate beta (eqn 2.63)
	doubleVector beta0 = 13.0/12.0*pow(v-2.0*vp+vpp,2.0)+0.25*pow(3.0*v-4.0*vp+vpp,2.0);
	doubleVector beta1 = 13.0/12.0*pow(vm-2.0*v+vp,2.0)+0.25*pow(vm-vp,2.0);
	doubleVector beta2 = 13.0/12.0*pow(vmm-2.0*vm+v,2.0)+0.25*pow(vmm-4*vm+3*v,2.0);
	// Now we can calculate alphas (2.59)
	doubleVector alpha0 = 3.0/(10.0*pow(epweno+beta0,2.0)); doubleVector alpha1 = 6.0/(10.0*pow(epweno+beta1,2)); doubleVector alpha2 = 1.0/(10.0*pow(epweno+beta2,2));
	doubleVector alphasum = alpha0 + alpha1 + alpha2;
	// Now we can finally calculate omega`
	doubleVector omega0 = alpha0/alphasum; doubleVector omega1 = alpha1/alphasum; doubleVector omega2 = alpha2/alphasum;
	doubleVector vplus = omega0*p0+omega1*p1+omega2*p2; // v-_i+1/2 (2.64)
	// assignCellData(&vplus,i,vplusval);
	// Now do the same thing all over for u^+_i-1/2. Note that we need to shift indexes forward by one because i-1/2 of this cell is i+1/2 of the previous.
	// We're also just going to reuse variables where we can here to save memory.
	vmm = state[gid-1];
	vm = state[gid];
	v = state[gid+1];
	vp = state[gid+2];
	vpp = state[gid+3];
	// Calculate the polynomials
	p0 = (11.0*v - 7.0*vp + 2.0*vpp)/6.0;
	p1 = (2.0*vm+5.0*v-vp)/6.0;
	p2 = (-vmm+5.0*vm+2.0*v)/6.0;
	//Now calculate beta (eqn 2.63)
	beta0 = 13.0/12.0*pow(v-2.0*vp+vpp,2.0)+0.25*pow(3.0*v-4.0*vp+vpp,2.0);
	beta1 = 13.0/12.0*pow(vm-2.0*v+vp,2.0)+0.25*pow(vm-vp,2.0);
	beta2 = 13.0/12.0*pow(vmm-2.0*vm+v,2.0)+0.25*pow(vmm-4.0*vm+3.0*v,2.0);
	// Now we can calculate alphas (2.59)
	alpha0 = 1.0/(10.0*pow(epweno+beta0,2.0)); alpha1 = 6.0/(10.0*pow(epweno+beta1,2.0)); alpha2 = 3.0/(10.0*pow(epweno+beta2,2.0));
	alphasum = alpha0 + alpha1 + alpha2;
	// Now we can finally calculate omega`
	omega0 = alpha0/alphasum; omega1 = alpha1/alphasum; omega2 = alpha2/alphasum;
	doubleVector vminus = omega0*p0+omega1*p1+omega2*p2;
	// assignCellData(&vminus,i,vminusval);
  // for (int i=0; i<NComponents; i++) {
  //   // This is a catch for our ghost cells
	//
	//
  // }
	doubleVector vplusFlux = getFlux(vplus);
	doubleVector vminusFlux = getFlux(vminus);
	// doubleVector finalFlux;
	flux[gid] = 0.5*(vplusFlux+vminusFlux-fabs(lambda[0])*(vminus-vplus));
	//
	// for (int i=0; i<NComponents; i++) {
	// 	double final;
	// 	if (gid < nGhostCells[0] || gid >= nCells[0]-nGhostCells[0]) {
	// 		final = 0.0;
	// 	} else {
	// 	}
	// 	assignCellData(&finalFlux,i,final);
	// }
	// // Verified that this is calculating correctly vs the Python program.
	// flux[gid] = finalFlux;
}

kernel void wenoCalculateValue(global doubleVector *stateOld, global doubleVector *flux, constant int *nCells, constant int *nGhostCells, constant double *dx, constant double *dt, constant int *rk3stage, global doubleVector *final) {
  int gid = get_global_id(0);
	int rk3 = rk3stage[0];
	doubleVector lastStepData = final[gid];
	doubleVector finalVal = 0.0;
	if (gid < nGhostCells[0] || gid >= nCells[0]-nGhostCells[0]) {
		finalVal = 0.0;
	} else {
		finalVal = (flux[gid]-flux[gid-1])/dx[0];
		if (gid == nGhostCells[0]) {
				finalVal = finalVal - flux[gid]/dx[0];
		} else if (gid==nCells[0]-nGhostCells[0]) { // Will never actually get here
				finalVal = finalVal + flux[gid]/dx[0];
		}
	}
	if (rk3 == 0) {
		finalVal = stateOld[gid]-dt[0]*finalVal;
	} else if (rk3 == 1) {
		finalVal = 0.75*stateOld[gid]+0.25*(lastStepData-dt[0]*finalVal);
	} else if (rk3 == 2) {
		finalVal = (stateOld[gid]+2.0*(lastStepData-dt[0]*finalVal))/3.0;
	}
	// assignCellData(&data,i,finalVal);
  // for (int i = 0; i<NComponents; i++) {
	//
  // }
	final[gid] = finalVal;
}

// // This should find the maximum speed of sound so that we can alter our dt
// kernel void findMaxLambda(global _cellData *state, global double *amax) {
// 	int lid = get_local_id(0);
// 	int gid = get_group_id(0);
// 	local double a0;
// 	a0 = 0;
// 	_cellData d = state[get_global_id(0)];
// 	double rho = getCellData(d,0);
// 	double U = getCellData(d,1)/rho;
// 	double P = (getCellData(d,2)-getCellData(d,1)*U/2)*(1.4-1);
// 	double al = sqrt(1.4*P/rho); // Speed of sound of this cell
// 	a0 = fmax(a0,al); // Take the maximum of our current a0 or the local sound speed.
// 	// Note: We don't care which thread in the group gets here first because we're just looking for a max.
// 	barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads to get here
// 	if (lid == 0) { // Now ask the first thread to write the value back
// 		amax[gid] = a0;
// 	}
// }
