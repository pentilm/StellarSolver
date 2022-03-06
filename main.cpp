#include <iostream>
#include <string.h>
#include "SimulationParameters.h"
#include "StellarSolverVisualizer.h"

bool checkCmdLineFlag(const int argc, char** argv, const char* string)
{
	bool flag = false;
	if(argc > 1){
		for(int i=1;i<argc;i++){
			if(strcmp(&argv[i][1], string) == 0){
				flag = true;
				break;
			}
		}
	}

	return flag;
}



int getCmdLineParameterInt(const int argc, char** argv, const char* string, const int defaultValue)
{
	int parameter = defaultValue;
	if(argc > 1){
		for(int i=1;i<argc;i++){
			char* str = &argv[i][1];
			char* eqs = strchr(str, '=');

			if(eqs != NULL){
				int index = (int)(eqs-str);
				char substring[50];
				strncpy(substring, str, index);
				substring[index] = '\0';

				if(strcmp(substring, string)==0){
					parameter = atoi(&eqs[1]);
					break;
				}
			}
		}
	}
	return parameter;
}


float getCmdLineParameterFloat(const int argc, char** argv, const char* string, const float defaultValue)
{
	float parameter = defaultValue;
	if(argc > 1){
		for(int i=1;i<argc;i++){
			char* str = &argv[i][1];
			char* eqs = strchr(str, '=');

			if(eqs != NULL){
				int index = (int)(eqs-str);
				char substring[50];
				strncpy(substring, str, index);
				substring[index] = '\0';

				if(strcmp(substring, string)==0){
					parameter = atof(&eqs[1]);
					break;
				}
			}

		}
	}
	return parameter;
}



int main(int argc, char** argv)
{
	int numbodies = 512*256;  // must be a power of 2!!
	SimulationParameters parameters;

	if(checkCmdLineFlag(argc, argv, "plummer") == true){
		parameters.model = plummer_model;
	}
	else if(checkCmdLineFlag(argc, argv, "colliding-disks") == true){
		parameters.model = colliding_disk_model;
	}
	else{
		parameters.model = disk_model;
	}

	parameters.opengl = checkCmdLineFlag(argc, argv, "opengl");
	parameters.debug = checkCmdLineFlag(argc, argv, "debug");
	parameters.benchmark = checkCmdLineFlag(argc, argv, "benchmark");
	parameters.fullscreen = checkCmdLineFlag(argc, argv, "fullscreen");
	parameters.iterations = getCmdLineParameterInt(argc, argv, "iterations", 50);
	parameters.timestep = getCmdLineParameterFloat(argc, argv, "timestep", 0.001);
	parameters.gravity = getCmdLineParameterFloat(argc, argv, "gravity", 1.0);
	parameters.dampening = getCmdLineParameterFloat(argc, argv, "dampening", 1.0);


	StellarSolverVisualizer simulation(parameters, numbodies);

	simulation.runSimulation();
}