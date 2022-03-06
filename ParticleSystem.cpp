#include <math.h>
#include <random>
#include "ParticleSystem.h"


//****************************************************************************************
// Plummer model for spherical galaxy
//
// rho = 3*M_h/4*pi * (a^2 / (r^2 + a^2)^2.5)
//
// M(r) = M_h * (r^3 / (r^2 + a^2)^1.5)
//****************************************************************************************
void ParticleSystem::plummerModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n)
{
	float a = 1.0;
	float pi = 3.14159265;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0, 1.0);
	std::uniform_real_distribution<float> distribution_phi(0.0, 2 * pi);
	std::uniform_real_distribution<float> distribution_theta(-1.0, 1.0);

	// loop through all particles
	for (int i = 0; i < n; i++){
		float phi = distribution_phi(generator);
		float theta = acos(distribution_theta(generator));
		float r = a*pow(distribution(generator), 0.3333) / sqrt(1 - pow(distribution(generator), 0.66667));

		// set mass and position of particle
		if(i==0){
			mass[i] = 100000;
			x[i] = 0;
			y[i] = 0;
		}
		else{
			mass[i] = 1.0;
			x[i] = r*cos(phi);
			y[i] = r*sin(phi);
		}

		// set velocity of particle
		float rotation = 1;  // 1: clockwise   -1: counter-clockwise 
		float v = 1.0*sqrt(parameters.gravity*100000.0 / r);
		if(i==0){
			x_vel[0] = 0;
			y_vel[0] = 0;
		}
		else{
			x_vel[i] = rotation*v*sin(phi);
			y_vel[i] = -rotation*v*cos(phi);
		}

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
void ParticleSystem::diskModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n)
{
	float a = 1.0;
	float pi = 3.14159265;
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(1.5, 12.0);
	std::uniform_real_distribution<float> distribution_theta(0.0, 2 * pi);

	// loop through all particles
	for (int i = 0; i < n; i++){
		float theta = distribution_theta(generator);
		float r = distribution(generator);

		// set mass and position of particle
		if(i==0){
			mass[i] = 100000;
			x[i] = 0;
			y[i] = 0;
		}
		else{
			mass[i] = 1.0;
			x[i] = r*cos(theta);
			y[i] = r*sin(theta);
		}


		// set velocity of particle
		float rotation = 1;  // 1: clockwise   -1: counter-clockwise 
		float v = 1.0*sqrt(parameters.gravity*100000.0 / r);
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
void ParticleSystem::collidingDiskModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n)
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
			mass[i] = 1;
			x[i] = 15*cos(theta);
			y[i] = 15*sin(theta);
		}
		else if(i<=n){
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
		float v2 = 1.0*sqrt(parameters.gravity*50000.0 / r2);
		float v = 1.0*sqrt(parameters.gravity*100000.0 / sqrt(450));
		if(i==0){
			x_vel[0] = 0;
			y_vel[0] = 0;
		}
		else if(i==1){
			x_vel[i] = rotation*v*sin(theta);
			y_vel[i] = -rotation*v*cos(theta);
		}
		else if(i<=n){
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