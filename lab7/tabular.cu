
#include <stdio.h>
#include "../utils/common.h"

#define PI 3.141592f

/*
 * Kernel: tabular function
 */
__global__ void tabular(float *a, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		float x = PI * (float)i / (float)n;
		float s = sinf(x);
		float c = cosf(x);
		a[i] = sqrtf(abs(s * s - c * c));
	}
}

/*
 * Kernel: tabular function using streams
 */
__global__ void tabular_streams(float *a, int n, int offset) {
	int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    float x = PI * (float)i / (float)n;
    float s = sinf(x);
    float c = cosf(x);
    a[i] = sqrtf(abs(s * s - c * c));
  }
}

/*
 * Error measure
 */
float maxError(float *a, int n) {
	float maxE = 0;
	for (int i = 0; i < n; i++) {
		float error = fabs(a[i] - 1.0f);
		if (error > maxE)
			maxE = error;
	}
	return maxE;
}

/*
 * Main: tabular function
 */
int main(void) {
	
  // main params
  uint MB = 1024*1024; 
  uint n = 256*MB;
	int blockSize = 256;

	// streams
	
	// allocate pinned host memory and device memory
	
	// create events and streams
	
	// baseline case - sequential transfer and execute
	
	// asynchronous version 1: loop over {copy, kernel, copy}
	
	// asynchronous version 2: loop over copy, loop over kernel, loop over copy
	
	// cleanup
	
	return 0;
}
