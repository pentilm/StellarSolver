#include <iostream>
#include <limits>
#include "debug.h"


void DEBUG_RUN_TESTS(float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child, float *left, float *right, float *bottom, float *top, int n, int m)
{
	long test;  // parameter determining whether test passed or failed


	//***********************************************************************
	// x, y, & mass array tests
	//***********************************************************************





	//***********************************************************************
	// count array tests
	//***********************************************************************

	// test 1
	test = 1;
	for(int i=n;i<m;i++){
		int c = count[i];
		for(int j=0;j<4;j++){
			if(child[4*i + j] != -1){
				c -= count[child[4*i + j]];
			}
		}
		if(c != 0){
			test = 0;
		}
	}
	if(test == 1){
		std::cout<<"COUNT ARRAY TEST 1 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"COUNT ARRAY TEST 1 RESULT = FAIL: "<<std::endl;
	}



	//***********************************************************************
	// child array tests
	//***********************************************************************

	// test 1
	test = 0;
	for(int j=0;j<4*m;j++){
		if(child[j] >=0 && child[j] < n){
			test += child[j];
		}
	}
	if(test == (long)n*(long)(n-1)/2){
		std::cout<<"CHILD ARRAY TEST 1 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"CHILD ARRAY TEST 1 RESULT = FAIL: "<<std::endl;
	}
	

	// test 2
	test = 1;
	for(int i=0;i<4*m;i++){
		if(child[i] >= m){
			test = 0;
		}
	}
	if(test == 1){
		std::cout<<"CHILD ARRAY TEST 2 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"CHILD ARRAY TEST 2 RESULT = FAIL: "<<std::endl;
	}


	// test 3
	test = 1;
	for(int i=0;i<4*m;i++){
		if(child[i] < -1){
			test = 0;
		}
	}
	if(test == 1){
		std::cout<<"CHILD ARRAY TEST 3 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"CHILD ARRAY TEST 3 RESULT = FAIL: "<<std::endl;
	}


	// test 4
	int min = 2*n;
	for(int i=0;i<4;i++){
		if(child[i] < min){
			min = child[i];
		}
	}
	test = 1;
	for(int i=4;i<4*(min-1);i++){
		if(child[i] != -1){
			test = 0;
		}
	}
	if(test == 1){
		std::cout<<"CHILD ARRAY TEST 4 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"CHILD ARRAY TEST 4 RESULT = FAIL: "<<std::endl;
	}



	//***********************************************************************
	// sorted array tests
	//***********************************************************************

	// test 1
	test = 0;
	for(int i=0;i<m;i++){
		test += sorted[i];
	}
	if(test == (long)n*(long)(n-1)/2){
		std::cout<<"SORTED ARRAY TEST 1 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"SORTED ARRAY TEST 1 RESULT = FAIL: "<<std::endl;
	}


	// test 2
	int *test_array = new int[n];
	for(int i=0;i<n;i++){
		test_array[i] = -1;
	}
	for(int i=0;i<n;i++){
		test_array[sorted[i]] = 1;
	}
	test = 1;
	for(int i=0;i<n;i++){
		if(test_array[i] != 1){
			test = 0;
		}
	}
	delete [] test_array;
	if(test == 1){
		std::cout<<"SORTED ARRAY TEST 2 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"SORTED ARRAY TEST 2 RESULT = FAIL: "<<std::endl;
	}



	//***********************************************************************
	// left, right, bottom, and top bounding region tests
	//***********************************************************************

	// test 1
	test = 1;
	if(*left > *right){
		test = 0;
	}
	if(*bottom > *top){
		test = 0;
	}
	if(test == 1){
		std::cout<<"BOUNDING BOX TEST 1 RESULT = PASS: "<<std::endl;
	}
	else{
		std::cout<<"BOUNDING BOX TEST 1 RESULT = FAIL: "<<std::endl;
	}


	
}