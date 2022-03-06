#ifndef __SIMULATIONPARAMETERS_H__
#define __SIMULATIONPARAMETERS_H__



typedef enum Model
{
	disk_model,
	colliding_disk_model,
	plummer_model
}Model;


typedef struct SimulationParameters
{
	Model model;
	bool opengl;
	bool debug;
	bool benchmark;
	bool fullscreen;
	float iterations;
	float timestep;
	float gravity;
	float dampening;
}SimulationParameters;

#endif
