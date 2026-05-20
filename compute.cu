#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

double *d_mass = NULL; //device array for masses
vector3 *d_newPos = NULL; //device array for new positions
vector3 *d_newVel = NULL; //device array for new velocities

//cuda error checker
void checkCuda(cudaError_t result, const char *message){
	if (result != cudaSuccess) {
		printf("CUDA error: %s: %s\n", message, cudaGetErrorString(result));
		exit(1);
	}
}

//function runs on gpu but launched from cpu
__global__ void computeKernel(vector3 *oldpos, vector3 *oldvel, double *devMass, vector3 *newpos, vector3 *newvel) {
	
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
			double dx = oldpos[i][0] - oldpos[j][0];
			double dy = oldpos[i][1] - oldpos[j][1];
			double dz = oldpos[i][2] - oldpos[j][2];

			double magnitiude_sq = dx*dx + dy*dy + dz*dz;
			double magnitude = sqrt(magnitiude_sq);
			double accelMag = -1.0 * GRAV_CONSTANT * devMass[j] / magnitiude_sq;
			
			accelX += accelMag * dx / magnitude;
			accelY += accelMag * dy / magnitude;
			accelZ += accelMag * dz / magnitude;
		}

		double vx = oldvel[i][0] + accelX * INTERVAL;
		double vy = oldvel[i][1] + accelY * INTERVAL;
		double vz = oldvel[i][2] + accelZ * INTERVAL;

		//update velocity and position
		newvel[i][0] = vx;
		newvel[i][1] = vy;
		newvel[i][2] = vz;

		newpos[i][0] = oldpos[i][0] + vx * INTERVAL;
		newpos[i][1] = oldpos[i][1] + vy * INTERVAL;
		newpos[i][2] = oldpos[i][2] + vz * INTERVAL;
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

		checkCuda(cudaMalloc((void **)&d_newPos, sizeof(vector3) * NUMENTITIES), "allocating d_newPos");
		checkCuda(cudaMalloc((void **)&d_newVel, sizeof(vector3) * NUMENTITIES), "allocating d_newVel");

		checkCuda(cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice), "copying hPos to GPU");
		checkCuda(cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice), "copying hVel to GPU");
		checkCuda(cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice), "copying mass to GPU");
		initialized = 1;
	}

	int threadsPerBlock = 256;
	int blocks = (NUMENTITIES + threadsPerBlock -1) / threadsPerBlock; //calculates how many GPU blocks are needed
	computeKernel<<<blocks, threadsPerBlock>>>(d_hPos, d_hVel, d_mass, d_newPos, d_newVel); //launches the kernel on the GPU
	checkCuda(cudaDeviceSynchronize(), "running computeKernel"); //waits for GPU to finish

	vector3 *tempPos = d_hPos;
	d_hPos = d_newPos;
	d_newPos = tempPos;

	vector3 *tempVel = d_hVel;
	d_hVel = d_newVel;
	d_newVel = tempVel;

	checkCuda(cudaGetLastError(), "launching computeKernel"); //checks for errors in kernel launch

	//makes sure the printed output uses the updated positions and velocities by copying them back to the CPU
	checkCuda(cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost), "copying hPos back to CPU");
	checkCuda(cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost), "copying hVel back to CPU");
}
