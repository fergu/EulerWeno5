# EulerWeno5

These are two (very basic) implementations of the 5th order WENO scheme for the Euler Equations with two components, using Lax-Friedrichs flux splitting. Both implementations have been validated against Sod's IC.

The first is in python/numpy, written entirely in vectorized form. 

The second is using OpenCL, a language which abstracts the parallel programming aspect, allowing this code to be implemented on a GPU. This has been tested to be consistent with the Python results when running on multiple CPU cores, but technical restrictions have limited GPU testing.

Due to the lesser testing of the OpenCL code, it currently only supports a single gas species while the Python code supports two species. The code allows for different mass fractions to be entered, but there is no actual calculation of the different properties. This will eventually be fixed as I have time to work on it.

# Use

Python: Can simply be run using 'python ./eulerweno_LF.py'. Plots will be output as the program runs.

OpenCL: Compilation differs based on platform. Due to time/testing constraints a makefile is not yet available, but in general compilation should be along the lines of:

'gcc main.c -lopencl -o weno.out' (on Windows/Cygwin or Linux)
'gcc main.c -framework OpenCL -o weno.out' (On MacOS)

NOTE: You may want to define CL_SILENCE_DEPRECATION on MacOS to silence the large number of deprecation warnings. This can be done using '-Wno-depricated-declarations'

Note that this assumes that OpenCL libraries are installed and available in a standard location.

Once compiled the program can be run using './weno.out'. It will produce an output file 'cloutput.dat'. The python script 'python ./plot.py' will plot the result versus the solution to Sod's problem.

# Sources

Any text with the formulation of the Euler equations and fluxes. 

Shu, C-W, "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws", NASA/CR-97-206253. 1997.

Assorted documentation for Python/Numpy and OpenCL.