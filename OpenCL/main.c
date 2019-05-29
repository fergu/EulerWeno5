#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

// Basically just redefining/abstracting the double vector type.
// That way it is trivial if we need to move to a longer vector later.
typedef cl_double8 doubleVector;
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

int main(void) {
	printf("Accelerated Euler Equation solver.\n");
	printf("Written by: Kevin Ferguson\n");

	printf("=== Setting up OpenCL ===\n");
	// Get platform and device information
	cl_platform_id platforms[16]; cl_uint num_platforms;

	cl_int ret = clGetPlatformIDs(16, platforms, &num_platforms);

	cl_platform_id plat; cl_device_id dev; cl_uint cores = 0;
	int i = 0; int j = 0;
	// So it seems that GPUs do not uniformly support double precision math, which is a problem for this kind of code. (double = ~7 places, double = ~16 places)
	// So we need to change this device selection to only pick devices which support double precision math.
	// This probably means that the local_group_size will need to account for max_group_size.
	printf("Found %d platforms.\n",num_platforms);
	for (i = 0; i<num_platforms; i++) {
		char platform_name[256];
		cl_device_id devices[16]; cl_uint num_devices;
		ret = clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,256*sizeof(char),&platform_name,NULL); // Get the platform name
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,16,devices,&num_devices); // Query how many devices we have
		printf("- Platform %d (\"%s\") has %d devices\n",i+1,platform_name,num_devices);
		for (j = 0; j<num_devices; j++) {
			cl_uint ncores; char device_name[256]; cl_device_fp_config fp_config;
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256*sizeof(char), &device_name, NULL);
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),&ncores,NULL);
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config),&fp_config,NULL);
			printf("- - Device %d (\"%s\") has %u compute units\n",j+1,device_name,ncores);
			if (fp_config != 0) {
				if (ncores > cores) {
					plat = platforms[i];
					dev = devices[j];
					cores = ncores;
				}
			} else {
				printf("- - - Device does not support double precision float. Skipping...\n");
			}
		}
	}

	char maxname[256]; char platname[256];
	ret = clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(maxname), &maxname, NULL);
	ret = clGetPlatformInfo(plat,CL_PLATFORM_NAME,256*sizeof(char),&platname,NULL); // Get the platform name
	printf("Using Device \"%s\" on \"%s\" with %u compute units\n",maxname,platname,cores);

	printf("Compiling compute kernels\n");
	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("wenovec.cl", "r");
	if (!fp) {
			fprintf(stderr, "Failed to load kernel.\n");
			exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &dev, NULL, NULL, &ret);
	printf("Return code from context creation: %d\n",ret);
	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, dev, 0, &ret);
	cl_program program = clCreateProgramWithSource(context, 1,(const char **)&source_str, (const size_t *)&source_size, &ret);
	printf("Return code from program creation: %d\n",ret);

	// Build the program
	ret = clBuildProgram(program, 1, &dev, NULL, NULL, NULL);
	printf("Return code from program build: %d\n",ret);
	char build_log[32000];
	clGetProgramBuildInfo(program, dev,  CL_PROGRAM_BUILD_LOG, sizeof(build_log), &build_log, NULL);
	printf("=== Build Log ===\n%s\n=== End Build Log ===\n",build_log);

	// Create the OpenCL kernel
	cl_kernel fluxKernel = clCreateKernel(program, "wenoCalculateFlux", &ret);
	printf("Return code from flux kernel creation: %d\n",ret);
	cl_kernel wenoKernel = clCreateKernel(program, "wenoCalculateValue", &ret);
	printf("Return code from weno kernel creation: %d\n",ret);

	printf("=== Finished setting up OpenCL ===\n\n");
	printf("=== Initializing Simulation Data ===\n");
	//For now we will hard-define our quantities. If we decide to make this more robust down the road we may allow for an input file
	double gamma = 1.4; // Gamma of the gas
	double L = 1.0; // Length of the domain

	// Numerical parameters
	const int NPTS = 200;
	double cfl = 0.55;
	double tfinal = 0.1;
	double dx = L/NPTS;

	doubleVector *cells = (doubleVector *)malloc(sizeof(doubleVector)*NPTS);

	for (i=0; i<=floor(NPTS/2); i++) {
		cells[i].rho = 1.0;
		cells[i].rhoU = 0.0;
		cells[i].E = 1.0/(gamma-1.0);
		cells[i].rhoY1 = cells[i].rho*1.0;
		cells[i].rhoY2 = 0.0;
	}
	for (i=floor(NPTS/2)+1; i<NPTS; i++) {
		cells[i].rho = 0.125;
		cells[i].rhoU = 0.0;
		cells[i].E = 0.1/(gamma-1.0);
		cells[i].rhoY1 = 0.0;
		cells[i].rhoY2 = cells[i].rho*1.0;
	}

	// We'll go ahead and calculate lambda (Which is the maximum speed of sound) in serial here since our data sets are relatively small. Can figure out how to do it in parallel later if we want. See: Reductions
	double lambda = 0.0;
	for (i=0; i<NPTS; i++) {
		double a = sqrt(gamma*cells[i].E*(gamma-1.0)/cells[i].rho);
		if (a > lambda) {
			lambda = a;
		}
	}
	double dt0 = cfl*dx/lambda; // Set our initial timestep
	double dt = dt0;
	int Nt = ceil(tfinal/dt);
	printf("Tfinal: %2.15e\n",(Nt-1.0)*dt);
	// Now create our memory buffers
	printf("Creating data buffers\n");
	cl_mem stateNew = clCreateBuffer(context, CL_MEM_READ_WRITE, NPTS*sizeof(doubleVector), NULL, &ret);
	cl_mem stateOld = clCreateBuffer(context, CL_MEM_READ_WRITE, NPTS*sizeof(doubleVector), NULL, &ret);
	cl_mem stateRK3 = clCreateBuffer(context, CL_MEM_READ_WRITE, NPTS*sizeof(doubleVector), NULL, &ret);
	cl_mem stateFlux = clCreateBuffer(context, CL_MEM_READ_WRITE, NPTS*sizeof(doubleVector), NULL, &ret);
	cl_mem lambdaMem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double), NULL, &ret);
	cl_mem dxMem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double), NULL, &ret);
	cl_mem dtMem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double), NULL, &ret);
	cl_mem rk3stageMem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int),NULL, &ret);
	cl_mem ncellsMem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, &ret);
	cl_mem nghostcellsMem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, &ret);

	printf("Writing data buffers to device\n");
	cl_int nCells = NPTS;
	cl_int nGhostCells = 3;
	cl_int rk3stage = 0;
	ret = clEnqueueWriteBuffer(command_queue, stateNew, CL_TRUE, 0, NPTS*sizeof(doubleVector), cells, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, stateOld, CL_TRUE, 0, NPTS*sizeof(doubleVector), cells, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, stateRK3, CL_TRUE, 0, NPTS*sizeof(doubleVector), cells, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, stateFlux, CL_TRUE, 0, NPTS*sizeof(doubleVector), cells, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, lambdaMem, CL_TRUE, 0, sizeof(cl_double), &lambda, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, dxMem, CL_TRUE, 0, sizeof(cl_double), &dx, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, dtMem, CL_TRUE, 0, sizeof(cl_double), &dt, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, ncellsMem, CL_TRUE, 0, sizeof(cl_int), &nCells, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, nghostcellsMem, CL_TRUE, 0, sizeof(cl_int), &nGhostCells, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, rk3stageMem, CL_TRUE, 0, sizeof(cl_int), &rk3stage, 0, NULL, NULL);

	printf("Setting initial kernel arguments\n");
	printf("- Setting Flux Kernel Arguments\n");
	ret = clSetKernelArg(fluxKernel, 0, sizeof(cl_mem), (void *)&stateRK3);
	ret = clSetKernelArg(fluxKernel, 1, sizeof(cl_mem), (void *)&ncellsMem);
	ret = clSetKernelArg(fluxKernel, 2, sizeof(cl_mem), (void *)&nghostcellsMem);
	ret = clSetKernelArg(fluxKernel, 3, sizeof(cl_mem), (void *)&lambdaMem);
	ret = clSetKernelArg(fluxKernel, 4, sizeof(cl_mem), (void *)&stateFlux);
	printf("- Setting WENO Kernel Arguments\n");
	ret = clSetKernelArg(wenoKernel, 0, sizeof(cl_mem), (void *)&stateOld);
	ret = clSetKernelArg(wenoKernel, 1, sizeof(cl_mem), (void *)&stateFlux);
	ret = clSetKernelArg(wenoKernel, 2, sizeof(cl_mem), (void *)&ncellsMem);
	ret = clSetKernelArg(wenoKernel, 3, sizeof(cl_mem), (void *)&nghostcellsMem);
	ret = clSetKernelArg(wenoKernel, 4, sizeof(cl_mem), (void *)&dxMem);
	ret = clSetKernelArg(wenoKernel, 5, sizeof(cl_mem), (void *)&dtMem);
	ret = clSetKernelArg(wenoKernel, 6, sizeof(cl_mem), (void *)&rk3stageMem);
	ret = clSetKernelArg(wenoKernel, 7, sizeof(cl_mem), (void *)&stateRK3);


	printf("Setup complete. Enqueueing kernels\n");
  // Execute the OpenCL kernel
	// FIXME: Use the kernel preferred size to see if we can get better efficiency. Can do this with a kernel info call
  size_t global_item_size = NPTS; // Process the entire lists
	// FIXME: The global_item_size needs to be evenly divisible by local_item_size. How can we set this automatically based on core count and number of points? Maybe a modulo thing.
  size_t local_item_size = 100; // Process in groups of 64

	for (int t=0; t<Nt; t++) {
		//printf("%f\n",(float)t*100.0/(float)Nt);
		for (i = 0; i<3; i++) {
			rk3stage = i;
			ret = clEnqueueWriteBuffer(command_queue, rk3stageMem, CL_TRUE, 0, sizeof(cl_int), &rk3stage, 0, NULL, NULL);
			ret = clEnqueueNDRangeKernel(command_queue, fluxKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
			ret = clEnqueueNDRangeKernel(command_queue, wenoKernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
		}
		ret = clEnqueueCopyBuffer(command_queue, stateRK3, stateOld, 0,0,NPTS*sizeof(doubleVector),0,NULL,NULL);
	}
	printf("Operation Queued. Please wait for final read...\n");
	doubleVector *out = (doubleVector *)malloc(NPTS*sizeof(doubleVector));
	ret = clEnqueueReadBuffer(command_queue, stateOld, CL_TRUE, 0, NPTS*sizeof(doubleVector), out, 0, NULL, NULL);

	printf("Final read completed. Writing results to file output.dat\n");
	FILE *outputFile;
	outputFile = fopen("cloutput.dat","w");
	for (i = 0; i < NPTS; i++) {
		fprintf(outputFile,"%2.15e\t%2.15e\t%2.15e\t%2.15e\t%2.15e\n",out[i].rho,out[i].rhoU,out[i].E,out[i].rhoY1,out[i].rhoY2);
	}
	fclose(outputFile);
	// Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(fluxKernel);
	ret = clReleaseKernel(wenoKernel);
  ret = clReleaseProgram(program);
	ret = clReleaseMemObject(stateNew);
	ret = clReleaseMemObject(stateOld);
	ret = clReleaseMemObject(stateFlux);
	ret = clReleaseMemObject(lambdaMem);
	ret = clReleaseMemObject(dxMem);
	ret = clReleaseMemObject(ncellsMem);
	ret = clReleaseMemObject(nghostcellsMem);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
	// Now free our malloc'd data
	free(cells);
	free(source_str);
	free(out);
  return 0;
}
