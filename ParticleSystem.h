#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include <iostream>
#include "SimulationParameters.h"

class ParticleSystem
{
	public:
		SimulationParameters parameters;
		
		ParticleSystem(const SimulationParameters p, const int n){}
		virtual ~ParticleSystem(){};

		virtual int getNumParticles() = 0; 
		virtual void update() = 0;
		virtual void reset() = 0;
		virtual float* getOutputBuffer() = 0;

		void plummerModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n);
		void diskModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n);
		void collidingDiskModel(float *mass, float *x, float* y, float *x_vel, float *y_vel, float *x_acc, float *y_acc, int n);
};


#endif