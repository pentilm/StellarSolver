#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include "debug.h"
#include "BarnesHutParticleSystem.h"
#include "Particle_cuda.cuh"

#include <stdio.h>
#include <cuda.h>
// ==========================================================================================
// CUDA ERROR CHECKING CODE
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) getchar();
   }
}

// ==========================================================================================



BarnesHutParticleSystem::BarnesHutParticleSystem(const SimulationParameters p, const int n) 
{
	parameters = p;
	step = 0;
	numParticles = n;
	numNodes = 2*n+12000;

	// allocate host data
	h_left = new float;
	h_right = new float;
	h_bottom = new float;
	h_top = new float;
	h_mass = new float[numNodes];
	h_x = new float[numNodes];
	h_y = new float[numNodes];
	h_vx = new float[numNodes];
	h_vy = new float[numNodes];
	h_ax = new float[numNodes];
	h_ay = new float[numNodes];
	h_child = new int[4*numNodes];
	h_start = new int[numNodes];
	h_sorted = new int[numNodes];
	h_count = new int[numNodes];
	h_output = new float[2*numNodes];

	// allocate device data
	gpuErrchk(cudaMalloc((void**)&d_left, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_right, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_bottom, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_top, sizeof(float)));
	gpuErrchk(cudaMemset(d_left, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_right, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_bottom, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_top, 0, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_mass, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_x, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_y, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_vx, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_vy, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ax, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ay, numNodes*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_index, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_child, 4*numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutex, sizeof(int))); 

	gpuErrchk(cudaMemset(d_start, -1, numNodes*sizeof(int)));
	gpuErrchk(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

	int memSize = sizeof(float) * 2 * numParticles;

	gpuErrchk(cudaMalloc((void**)&d_output, 2*numNodes*sizeof(float)));
}


BarnesHutParticleSystem::BarnesHutParticleSystem(const BarnesHutParticleSystem &system)
{
	parameters = system.parameters;
	step = system.step;
	numParticles = system.numParticles;
	numNodes = system.numNodes;

	// allocate host data
	h_left = new float;
	h_right = new float;
	h_bottom = new float;
	h_top = new float;
	h_mass = new float[numNodes];
	h_x = new float[numNodes];
	h_y = new float[numNodes];
	h_vx = new float[numNodes];
	h_vy = new float[numNodes];
	h_ax = new float[numNodes];
	h_ay = new float[numNodes];
	h_child = new int[4*numNodes];
	h_start = new int[numNodes];
	h_sorted = new int[numNodes];
	h_count = new int[numNodes];
	h_output = new float[2*numNodes];

	// allocate device data
	gpuErrchk(cudaMalloc((void**)&d_left, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_right, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_bottom, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_top, sizeof(float)));
	gpuErrchk(cudaMemset(d_left, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_right, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_bottom, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_top, 0, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_mass, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_x, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_y, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_vx, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_vy, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ax, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ay, numNodes*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_index, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_child, 4*numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutex, sizeof(int))); 

	gpuErrchk(cudaMemset(d_start, -1, numNodes*sizeof(int)));
	gpuErrchk(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

	int memSize = sizeof(float) * 2 * numParticles;

	gpuErrchk(cudaMalloc((void**)&d_output, 2*numNodes*sizeof(float)));
}


BarnesHutParticleSystem& BarnesHutParticleSystem::operator=(const BarnesHutParticleSystem &system)
{
	if(this != &system){
		delete h_left;
		delete h_right;
		delete h_bottom;
		delete h_top;
		delete [] h_mass;
		delete [] h_x;
		delete [] h_y;
		delete [] h_vx;
		delete [] h_vy;
		delete [] h_ax;
		delete [] h_ay;
		delete [] h_child;
		delete [] h_start;
		delete [] h_sorted;
		delete [] h_count;
		delete [] h_output;
		
		gpuErrchk(cudaFree(d_left));
		gpuErrchk(cudaFree(d_right));
		gpuErrchk(cudaFree(d_bottom));
		gpuErrchk(cudaFree(d_top));

		gpuErrchk(cudaFree(d_mass));
		gpuErrchk(cudaFree(d_x));
		gpuErrchk(cudaFree(d_y));
		gpuErrchk(cudaFree(d_vx));
		gpuErrchk(cudaFree(d_vy));
		gpuErrchk(cudaFree(d_ax));
		gpuErrchk(cudaFree(d_ay));

		gpuErrchk(cudaFree(d_index));
		gpuErrchk(cudaFree(d_child));
		gpuErrchk(cudaFree(d_start));
		gpuErrchk(cudaFree(d_sorted));
		gpuErrchk(cudaFree(d_count));

		gpuErrchk(cudaFree(d_mutex));

		gpuErrchk(cudaFree(d_output));

		parameters = system.parameters;
		step = system.step;
		numParticles = system.numParticles;
		numNodes = system.numNodes;

		// allocate host data
		h_left = new float;
		h_right = new float;
		h_bottom = new float;
		h_top = new float;
		h_mass = new float[numNodes];
		h_x = new float[numNodes];
		h_y = new float[numNodes];
		h_vx = new float[numNodes];
		h_vy = new float[numNodes];
		h_ax = new float[numNodes];
		h_ay = new float[numNodes];
		h_child = new int[4*numNodes];
		h_start = new int[numNodes];
		h_sorted = new int[numNodes];
		h_count = new int[numNodes];
		h_output = new float[2*numNodes];

		// allocate device data
		gpuErrchk(cudaMalloc((void**)&d_left, sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_right, sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_bottom, sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_top, sizeof(float)));
		gpuErrchk(cudaMemset(d_left, 0, sizeof(float)));
		gpuErrchk(cudaMemset(d_right, 0, sizeof(float)));
		gpuErrchk(cudaMemset(d_bottom, 0, sizeof(float)));
		gpuErrchk(cudaMemset(d_top, 0, sizeof(float)));

		gpuErrchk(cudaMalloc((void**)&d_mass, numNodes*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_x, numNodes*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_y, numNodes*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_vx, numNodes*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_vy, numNodes*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_ax, numNodes*sizeof(float)));
		gpuErrchk(cudaMalloc((void**)&d_ay, numNodes*sizeof(float)));

		gpuErrchk(cudaMalloc((void**)&d_index, sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_child, 4*numNodes*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&d_mutex, sizeof(int))); 

		gpuErrchk(cudaMemset(d_start, -1, numNodes*sizeof(int)));
		gpuErrchk(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

		int memSize = sizeof(float) * 2 * numParticles;

		gpuErrchk(cudaMalloc((void**)&d_output, 2*numNodes*sizeof(float)));	
	}

	return *this;
}
		

BarnesHutParticleSystem::~BarnesHutParticleSystem()
{
	delete h_left;
	delete h_right;
	delete h_bottom;
	delete h_top;
	delete [] h_mass;
	delete [] h_x;
	delete [] h_y;
	delete [] h_vx;
	delete [] h_vy;
	delete [] h_ax;
	delete [] h_ay;
	delete [] h_child;
	delete [] h_start;
	delete [] h_sorted;
	delete [] h_count;
	delete [] h_output;
	
	gpuErrchk(cudaFree(d_left));
	gpuErrchk(cudaFree(d_right));
	gpuErrchk(cudaFree(d_bottom));
	gpuErrchk(cudaFree(d_top));

	gpuErrchk(cudaFree(d_mass));
	gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_y));
	gpuErrchk(cudaFree(d_vx));
	gpuErrchk(cudaFree(d_vy));
	gpuErrchk(cudaFree(d_ax));
	gpuErrchk(cudaFree(d_ay));

	gpuErrchk(cudaFree(d_index));
	gpuErrchk(cudaFree(d_child));
	gpuErrchk(cudaFree(d_start));
	gpuErrchk(cudaFree(d_sorted));
	gpuErrchk(cudaFree(d_count));

	gpuErrchk(cudaFree(d_mutex));

	gpuErrchk(cudaFree(d_output));

	cudaDeviceSynchronize();
}


int BarnesHutParticleSystem::getNumParticles()
{
	return numParticles;
}


void BarnesHutParticleSystem::update()
{
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	ResetArrays(d_mutex, d_x, d_y, d_mass, d_count, d_start, d_sorted, d_child, d_index, d_left, d_right, d_bottom, d_top, numParticles, numNodes);
	ComputeBoundingBox(d_mutex, d_x, d_y, d_left, d_right, d_bottom, d_top, numParticles);
	BuildQuadTree(d_x, d_y, d_mass, d_count, d_start, d_child, d_index, d_left, d_right, d_bottom, d_top, numParticles, numNodes);
	ComputeCentreOfMass(d_x, d_y, d_mass, d_index, numParticles);
	SortParticles(d_count, d_start, d_sorted, d_child, d_index, numParticles);
	CalculateForces(d_x, d_y, d_vx, d_vy, d_ax, d_ay, d_mass, d_sorted, d_child, d_left, d_right, numParticles, parameters.gravity);
	IntegrateParticles(d_x, d_y, d_vx, d_vy, d_ax, d_ay, numParticles, parameters.timestep, parameters.dampening);
	FillOutputArray(d_x, d_y, d_output, numNodes);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if(parameters.benchmark == true){
		std::cout<<"Timestep: "<<step<<"    "<<"Elapsed time: "<<elapsedTime<<std::endl;
	}

	if(parameters.opengl == true){
		cudaMemcpy(h_output, d_output, 2*numNodes*sizeof(float), cudaMemcpyDeviceToHost);
	}

	if(parameters.debug == true){
		cudaMemcpy(h_left, d_left, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_right, d_right, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_bottom, d_bottom, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_top, d_top, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_child, d_child, 4*numNodes*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_x, d_x, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_y, d_y, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_mass, d_mass, numNodes*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_count, d_count, numNodes*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sorted, d_sorted, numNodes*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_start, d_start, numNodes*sizeof(int), cudaMemcpyDeviceToHost);
		DEBUG_RUN_TESTS(h_x, h_y, h_mass, h_count, h_start, h_sorted, h_child, h_left, h_right, h_bottom, h_top, numParticles, numNodes);
	}

	step++;
}


void BarnesHutParticleSystem::reset()
{
	// set initial mass, position, and velocity of particles
	if(parameters.model == plummer_model){
		plummerModel(h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay, numParticles);
	}
	else if(parameters.model == colliding_disk_model){
		collidingDiskModel(h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay, numParticles);
	}
	else{
		diskModel(h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay, numParticles);
	}


	// copy data to GPU device
	cudaMemcpy(d_mass, h_mass, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vx, h_vx, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, h_vy, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ax, h_ax, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ay, h_ay, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
}


const float* BarnesHutParticleSystem::getOutputBuffer()
{
	return h_output;
}




//****************************************************************************************
// Plummer model for spherical galaxy http://www.artcompsci.org/kali/vol/plummer/volume11.pdf
//
// rho = 3*M_h/4*pi * (a^2 / (r^2 + a^2)^2.5)
//
// M(r) = M_h * (r^3 / (r^2 + a^2)^1.5)
//****************************************************************************************
void BarnesHutParticleSystem::plummerModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n)
{
	float a = 1.0;
	float pi = 3.14159265;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0, 1.0);
	std::uniform_real_distribution<float> distribution2(0, 0.1);
	std::uniform_real_distribution<float> distribution_phi(0.0, 2 * pi);
	std::uniform_real_distribution<float> distribution_theta(-1.0, 1.0);

	// loop through all particles
	for (int i = 0; i < n; i++){
		float phi = distribution_phi(generator);
		float theta = acos(distribution_theta(generator));
		float r = a / sqrt(pow(distribution(generator), -0.666666) - 1);

		// set mass and position of particle
		mass[i] = 1.0;
		x[i] = r*cos(phi);
		y[i] = r*sin(phi);

		// set velocity of particle
		float s = 0.0;
		float t = 0.1;
		while(t > s*s*pow(1.0 - s*s, 3.5)){
			s = distribution(generator);
			t = distribution2(generator);
		}
		float v = 100*s*sqrt(2)*pow(1.0 + r*r, -0.25);
		phi = distribution_phi(generator);
		theta = acos(distribution_theta(generator));
		x_vel[i] = v*cos(phi);
		y_vel[i] = v*sin(phi);

		// set acceleration to zero
		x_acc[i] = 0.0;
		y_acc[i] = 0.0;
	}
}



//****************************************************************************************
// Simple disk galaxy
//
// 
//
// 
//****************************************************************************************
void BarnesHutParticleSystem::diskModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n)
{
	float a = 1.0;
	float pi = 3.14159265;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> uniform(0.0, 2.0);
	std::uniform_real_distribution<float> distribution(1.5, 12.0);
	std::uniform_real_distribution<float> distribution_theta(0.0, 2 * pi);

	// loop through all particles
	for (int i = 0; i < n; i++){
		float theta = distribution_theta(generator);
		float r = distribution(generator);

		// set mass and position of particle
		if(i==0){
			mass[i] = 200000;
			x[i] = 0;
			y[i] = 0;
		}
		else{
			mass[i] = 1.0;
			x[i] = r*cos(theta);
			y[i] = r*sin(theta);
		}


		// set velocity of particle
		float rotation = -1;  // 1: clockwise   -1: counter-clockwise 
		float v = 1.0*sqrt(parameters.gravity*200000.0 / r);
		if(i==0){
			x_vel[0] = 0;
			y_vel[0] = 0;
		}
		else{
			x_vel[i] = rotation*v*sin(theta);
			y_vel[i] = -rotation*v*cos(theta);
		}

		// set acceleration to zero
		x_acc[i] = 0.0;
		y_acc[i] = 0.0;
	}

}



//****************************************************************************************
// Two galaxies colliding disk galaxy
//
// 
//
// 
//****************************************************************************************
void BarnesHutParticleSystem::collidingDiskModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n)
{
	float a = 1.0;
	float pi = 3.14159265;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution1(1.5, 12.0);
	std::uniform_real_distribution<float> distribution2(1, 5.0);
	std::uniform_real_distribution<float> distribution_theta(0.0, 2 * pi);


	// loop through all particles
	for (int i = 0; i < n; i++){
		float theta = distribution_theta(generator);
		float r1 = distribution1(generator);
		float r2 = distribution2(generator);

		// set mass and position of particle
		if(i==0){
			mass[i] = 100000;
			x[i] = 0;
			y[i] = 0;
		}
		else if(i==1){
			mass[i] = 25000;
			x[i] = 20*cos(theta);
			y[i] = 20*sin(theta);
		}
		else if(i<=3*n/4){
			mass[i] = 1.0;
			x[i] = r1*cos(theta);
			y[i] = r1*sin(theta);
		}
		else{
			mass[i] = 1.0;
			x[i] = r2*cos(theta) + x[1];
			y[i] = r2*sin(theta) + y[1];
		}


		// set velocity of particle
		float rotation = 1;  // 1: clockwise   -1: counter-clockwise 
		float v1 = 1.0*sqrt(parameters.gravity*100000.0 / r1);
		float v2 = 1.0*sqrt(parameters.gravity*25000.0 / r2);
		float v = 1.0*sqrt(parameters.gravity*100000.0 / sqrt(800));
		if(i==0){
			x_vel[0] = 0;
			y_vel[0] = 0;
		}
		else if(i==1){
			x_vel[i] = 0.0;//rotation*v*sin(theta);
			y_vel[i] = 0.0;//-rotation*v*cos(theta);
		}
		else if(i<=3*n/4){
			x_vel[i] = rotation*v1*sin(theta);
			y_vel[i] = -rotation*v1*cos(theta);
		}
		else{
			x_vel[i] = rotation*v2*sin(theta);
			y_vel[i] = -rotation*v2*cos(theta);			
		}

		// set acceleration to zero
		x_acc[i] = 0.0;
		y_acc[i] = 0.0;
	}

}
