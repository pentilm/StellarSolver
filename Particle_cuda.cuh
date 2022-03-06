#ifndef __PARTICLE_CUDA_CUH__
#define __PARTICLE_CUDA_CUH__


void SetDrawArray(float *ptr, float *x, float *y, int n);
void ResetArrays(int *mutex, float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, int n, int m);
void ComputeBoundingBox(int *mutex, float *x, float *y, float *left, float *right, float *bottom, float *top, int n);
void BuildQuadTree(float *x, float *y, float *mass, int *count, int *start, int *child, int *index, float *left, float *right, float *bottom, float *top, int n, int m);
void ComputeCentreOfMass(float *x, float *y, float *mass, int *index, int n);
void SortParticles(int *count, int *start, int *sorted, int *child, int *index, int n);
void CalculateForces(float* x, float *y, float *vx, float *vy, float *ax, float *ay, float *mass, int *sorted, int *child, float *left, float *right, int n, float g);
void IntegrateParticles(float *x, float *y, float *vx, float *vy, float *ax, float *ay, int n, float dt, float d);
void FillOutputArray(float *x, float *y, float *out, int n);


#endif