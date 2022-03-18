# StellarSolver: High-Performance N-Body Simulation with CUDA and Barnes-Hut Algorithm

## Overview

StellarSolver is a comprehensive tool designed for simulating the n-body problem, utilizing the Barnes-Hut algorithm powered by CUDA. The project provides visualization through OpenGL, following Nvidia's CUDA toolkit examples. Currently, the visualization process is executed by the host, transferring data back at each time-step without utilizing CUDA-OpenGL interoperability due to system restrictions during development.

## Prerequisites

StellarSolver necessitates the installation of Nvidia's CUDA toolkit on a system with a CUDA-capable device and a GCC compiler. The visualization component uses OpenGL, SFML, GLEW (OpenGL Extension Wrangler Library), and GLM (OpenGL Mathematics).

For CUDA installation, refer to the [Nvidia CUDA download page](https://developer.nvidia.com/cuda-downloads) and the CUDA Quick Start Guide. On Ubuntu, install SFML and GLEW by executing the following commands:

```bash
sudo apt-get install libsfml-dev
sudo apt-get install libglew-dev
```

GLM, a collection of header files, can be acquired [here](http://glm.g-truc.net/0.9.8/index.html). Ensure to update the makefile INCLUDE variable to set the path to the GLM directory.

## Compilation

To compile the code, execute the following commands:

```bash
make clean
make build
```

## Execution

StellarSolver offers multiple command-line arguments for customization.

The standard execution of the Barnes-Hut algorithm with OpenGL visualization:

```bash
./app -barnes-hut -opengl
```

Execution with benchmark statistics for 500 iterations:

```bash
./app -barnes-hut -benchmark -iterations=500
```

Additional command-line options are detailed below:

* `-disk` : Use a simple disk model (default).
* `-plummer` : Use a Plummer model.
* `-colliding-disks` : Use two colliding disks.
* `-opengl` : Enable OpenGL visualization.
* `-benchmark` : Output time statistics.
* `-debug` : Run debug tests.
* `-iterations=<n>` : Define the number of iterations (defaults to 50).
* `-gravity=<n>` : Adjust the gravity parameter (defaults to 1.0).
* `-dampening=<n>` : Adjust the velocity dampening parameter (defaults to 1.0).

## Additional Notes

Ensure to manually match the 'numbodies' variable in main.cpp and the 'blockSize' variables in kernels.cu and particle_cuda.cu. For instance, if you set `numbodies = 64*64` in main.cpp, also set `blockSize = 64` in kernels.cu, and `blockSize = 64, gridSize = 64` in particle_cuda.cu.
