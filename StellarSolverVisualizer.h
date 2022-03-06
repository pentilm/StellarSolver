#ifndef __STELLARSOLVERVISUALIZER_H__
#define __STELLARSOLVERVISUALIZER_H__


#include <iostream>
#include "SimulationParameters.h"
#include "BarnesHutParticleSystem.h"


#define GLEW_STATIC
#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"



class StellarSolverVisualizer
{
	private:
		int numOfBodies;
		BarnesHutParticleSystem *particles;
		SimulationParameters parameters;

		sf::ContextSettings *settings;
		sf::Window *window;

		GLuint vao;
		GLuint vbo;

		GLuint vertexShader;
		GLuint fragmentShader;
		GLuint shaderProgram;

		void displayDeviceProperties();

	public:
		StellarSolverVisualizer(const SimulationParameters p, const int numBodies);
		StellarSolverVisualizer(const StellarSolverVisualizer &visualizer);
		StellarSolverVisualizer& operator=(const StellarSolverVisualizer &visualizer);
		~StellarSolverVisualizer();

		void runSimulation();
};



#endif