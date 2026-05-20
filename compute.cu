#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

double *d_mass = NULL; //device array for masses

//cuda error checker
void checkCuda(cudaError_t result, const char *message){
	if (result != cudaSuccess) {
		printf("CUDA error: %s: %s\n", message, cudaGetErrorString(result));
		exit(1);
	}
}

//function runs on gpu but launched from cpu
__global__ void computeKernel(vector3 *pos, vector3 *vel, double *devMass){
	int i = blockIdx.x * blockDim.x + threadIdx.x; //gives each thread a unique number
	if (i>=NUMENTITIES) {
		return; //protects againts extra threads
	}

	double accelX = 0.0;
	double accelY = 0.0;
	double accelZ = 0.0;

	for (int j=0; j<NUMENTITIES; j++) {
		if (i != j)
		{
			double dx = pos[i][0] - pos[j][0];
			double dy = pos[i][1] - pos[j][1];
			double dz = pos[i][2] - pos[j][2];
			
			double magnitiude_sq = dx*dx + dy*dy + dz*dz;
			double magnitude = sqrt(magnitiude_sq);
			double accelMag = -1.0 * GRAV_CONSTANT * devMass[j] / magnitiude_sq;
			
			accelX += accelMag * dx / magnitude;
			accelY += accelMag * dy / magnitude;
			accelZ += accelMag * dz / magnitude;
		}

		//update velocity and position
		vel[i][0] += accelX * INTERVAL;
		vel[i][1] += accelY * INTERVAL;
		vel[i][2] += accelZ * INTERVAL;

		pos[i][0] += vel[i][0] * INTERVAL;
		pos[i][1] += vel[i][1] * INTERVAL;	
		pos[i][2] += vel[i][2] * INTERVAL;
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
extern "C" void compute(){
	static int initialized = 0; //remembers value between calls
	if (!initialized) {
		checkCuda(cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES), "allocating d_hPos");
		checkCuda(cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES), "allocating d_hVel");
		checkCuda(cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES), "allocating d_mass");

		checkCuda(cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice), "copying hPos to GPU");
		checkCuda(cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice), "copying hVel to GPU");
		checkCuda(cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice), "copying mass to GPU");
		initialized = 1;
	}

	int threadsPerBlock = 256;
	int blocks = (NUMENTITIES + threadsPerBlock -1) / threadsPerBlock; //calculates how many GPU blocks are needed
	computeKernel<<<blocks, threadsPerBlock>>>(d_hPos, d_hVel, d_mass); //launches the kernel on the GPU

	checkCuda(cudaGetLastError(), "launching computeKernel"); //checks for errors in kernel launch
	checkCuda(cudaDeviceSynchronize(), "running computeKernel"); //waits for GPU to finish

	//makes sure the printed output uses the updated positions and velocities by copying them back to the CPU
	checkCuda(cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost), "copying hPos back to CPU");
	checkCuda(cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost), "copying hVel back to CPU");
}
