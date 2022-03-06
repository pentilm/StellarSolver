#include "StellarSolverVisualizer.h"

// Shader sources
const GLchar* vertexSource =
    "#version 130\n"
    "in vec2 position;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform mat4 projection;"
    "void main()"
    "{"
    "    gl_Position = projection * view * model *vec4(position, 0.0, 1.0);"
    "}";
const GLchar* fragmentSource =
    "#version 130\n"
    //"out vec4 outColor;"
    "void main()"
    "{"
    "    gl_FragColor = vec4(1.0, 1.0, 1.0, 0.1);"
    "}"; 




StellarSolverVisualizer::StellarSolverVisualizer(const SimulationParameters p, const int numBodies)
{
	numOfBodies = numBodies;
	parameters = p;
	particles = new BarnesHutParticleSystem(parameters, numOfBodies);

	// opengl initialization
	if(parameters.opengl){
		settings = new sf::ContextSettings();
    	settings->depthBits = 24;
    	settings->stencilBits = 8;
    	window = new sf::Window(sf::VideoMode(1000, 1000, 32), "N body Solver", sf::Style::Titlebar | sf::Style::Close, *settings);

    	glewExperimental = GL_TRUE;
    	glewInit();

    	// Create and compile the vertex shader
		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);

		// Create and compile the fragment shader
		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);

		// Link the vertex and fragment shader into a shader program
		shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glBindFragDataLocation(shaderProgram, 0, "outColor");
		glLinkProgram(shaderProgram);
		glUseProgram(shaderProgram);
	}
}


StellarSolverVisualizer::StellarSolverVisualizer(const StellarSolverVisualizer &visualizer)
{
	numOfBodies = visualizer.numOfBodies;
	parameters = visualizer.parameters;

	particles = new BarnesHutParticleSystem(parameters, numOfBodies);

	if(parameters.opengl){
		settings = new sf::ContextSettings();
    	settings->depthBits = 24;
    	settings->stencilBits = 8;
    	window = new sf::Window(sf::VideoMode(1000, 1000, 32), "N body Solver", sf::Style::Titlebar | sf::Style::Close, *settings);

    	glewExperimental = GL_TRUE;
    	glewInit();

    	// Create and compile the vertex shader
		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);

		// Create and compile the fragment shader
		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);

		// Link the vertex and fragment shader into a shader program
		shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glBindFragDataLocation(shaderProgram, 0, "outColor");
		glLinkProgram(shaderProgram);
		glUseProgram(shaderProgram);
	}
}


StellarSolverVisualizer& StellarSolverVisualizer::operator=(const StellarSolverVisualizer &visualizer)
{
	if(this != &visualizer){
		numOfBodies = visualizer.numOfBodies;
		parameters = visualizer.parameters;

		delete particles;
		particles = new BarnesHutParticleSystem(parameters, numOfBodies);

		if(parameters.opengl){
			delete settings;
			delete window;

			settings = new sf::ContextSettings();
    		settings->depthBits = 24;
    		settings->stencilBits = 8;
    		window = new sf::Window(sf::VideoMode(1000, 1000, 32), "N body Solver", sf::Style::Titlebar | sf::Style::Close, *settings);

    		glewExperimental = GL_TRUE;
    		glewInit();

    		glDeleteProgram(shaderProgram);
			glDeleteShader(fragmentShader);
			glDeleteShader(vertexShader);

			// Create and compile the vertex shader
			vertexShader = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertexShader, 1, &vertexSource, NULL);
			glCompileShader(vertexShader);

			// Create and compile the fragment shader
			fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
			glCompileShader(fragmentShader);

			// Link the vertex and fragment shader into a shader program
			shaderProgram = glCreateProgram();
			glAttachShader(shaderProgram, vertexShader);
			glAttachShader(shaderProgram, fragmentShader);
			glBindFragDataLocation(shaderProgram, 0, "outColor");
			glLinkProgram(shaderProgram);
			glUseProgram(shaderProgram);
		}
	}
	
	return *this;
}


StellarSolverVisualizer::~StellarSolverVisualizer()
{
	delete particles;

	if(parameters.opengl){
		delete settings;
		delete window;

		glDeleteProgram(shaderProgram);
		glDeleteShader(fragmentShader);
		glDeleteShader(vertexShader);
	}
}


void StellarSolverVisualizer::displayDeviceProperties()
{
	// Set up CUDA device 
	cudaDeviceProp properties;

	cudaGetDeviceProperties(&properties,0);

	int fact = 1024;
	int driverVersion, runtimeVersion;

	cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

	std::cout << "************************************************************************" << std::endl;
	std::cout << "                          GPU Device Properties                         " << std::endl;
	std::cout << "************************************************************************" << std::endl;
	std::cout << "Name:                                    " << properties.name << std::endl;
	std::cout << "CUDA driver/runtime version:             " << driverVersion/1000 << "." << (driverVersion%100)/10 << "/" << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;
	std::cout << "CUDA compute capabilitiy:                " << properties.major << "." << properties.minor << std::endl;
	std::cout << "Number of multiprocessors:               " << properties.multiProcessorCount << std::endl;                           
	std::cout << "GPU clock rate:                          " << properties.clockRate/fact << " (MHz)" << std::endl;
	std::cout << "Memory clock rate:                       " << properties.memoryClockRate/fact << " (MHz)" << std::endl;
	std::cout << "Memory bus width:                        " << properties.memoryBusWidth << "-bit" << std::endl;
	std::cout << "Theoretical memory bandwidth:            " << (properties.memoryClockRate/fact*(properties.memoryBusWidth/8)*2)/fact <<" (GB/s)" << std::endl;
	std::cout << "Device global memory:                    " << properties.totalGlobalMem/(fact*fact) << " (MB)" << std::endl;
	std::cout << "Shared memory per block:                 " << properties.sharedMemPerBlock/fact <<" (KB)" << std::endl;
	std::cout << "Constant memory:                         " << properties.totalConstMem/fact << " (KB)" << std::endl;
	std::cout << "Maximum number of threads per block:     " << properties.maxThreadsPerBlock << std::endl;
	std::cout << "Maximum thread dimension:                [" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << "]" << std::endl;
	std::cout << "Maximum grid size:                       [" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << "]" << std::endl;
	std::cout << "**************************************************************************" << std::endl;
	std::cout << "                                                                          " << std::endl;
	std::cout << "**************************************************************************" << std::endl;
}


void StellarSolverVisualizer::runSimulation()
{
	displayDeviceProperties();
	
	particles->reset();

    for(int i=0;i<parameters.iterations;i++){ 
    	particles->update();

    	if(parameters.opengl){
    		const float* vertices = particles->getOutputBuffer();

		   	glGenVertexArrays(1, &vao);
		    glBindVertexArray(vao);

			glGenBuffers(1, &vbo);   //generate a buffer
		  	glBindBuffer(GL_ARRAY_BUFFER, vbo);   //make buffer active
			glBufferData(GL_ARRAY_BUFFER, 2*particles->getNumParticles()*sizeof(float), vertices, GL_DYNAMIC_DRAW); //copy data to active buffer 

		    // Specify the layout of the vertex data
		    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
		    glEnableVertexAttribArray(posAttrib);
		    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);

            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            glEnable(GL_BLEND);

            // model, view, and projection matrices
            glm::mat4 model = glm::mat4(1.0f);
            glm::mat4 view = glm::mat4(1.0f);
            // view = glm::rotate(view, float(2*i), glm::vec3(0.0f, 1.0f, 0.0f)); 
            glm::mat4 projection = glm::ortho(-25.0f, 25.0f, -25.0f, 25.0f, -10.0f, 10.0f);

            // link matrices with shader program
            GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
            GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
			GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
			glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

		    // Clear the screen to black
		    glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
		    glClear(GL_COLOR_BUFFER_BIT);

		    // Draw points
		    glDrawArrays(GL_POINTS, 0, particles->getNumParticles());

		    // Swap buffers
		    window->display();

		    glDeleteBuffers(1, &vbo);

		    glDeleteVertexArrays(1, &vao);
		}
	}

	if(parameters.opengl){
		window->close(); 
	}
}
