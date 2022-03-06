#ifndef __DEBUG_H__
#define __DEBUG_H__


#define DEBUG 1
#define TIMING 0

#define DEBUG_PRINT(stream, statement) \
	do { if(DEBUG) (stream) << "DEBUG: "<< __FILE__<<"("<<__LINE__<<") " << (statement) << std::endl;} while(0)

#define TIMING_PRINT(stream, statement, time) \
	do { if(TIMING) (stream) << "TIMING: "<<__FILE__<<"("<<__LINE__<<") " << (statement) << (time) << std::endl;} while(0)


void DEBUG_RUN_TESTS(float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child, float *left, float *right, float *bottom, float *top, int n, int m);

#endif