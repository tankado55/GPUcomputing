
#include <stdio.h>
#include "../utils/common.h"

#define PI 3.141592f
#define NSTREAM 4

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

    dim3 grid  ((nElem + block.x - 1) / block.x);
	
  // main params
    uint MB = 1024*1024;
    uint nElem = 256*MB;
    int blockSize = 256;
    size_t nBytes = nElem * sizeof(float);
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

	// streams
    cudaStream_t stream[NSTREAM];
	
	// allocate pinned host memory and device memory
    float *h_A, *hostRef, *gpuRef;
    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&gpuRef, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&hostRef, nBytes, cudaHostAllocDefault));

    float *d_A;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
	
	// create events and streams
    for (int i = 0; i < NSTREAM; ++i)
        CHECK(cudaStreamCreate(&stream[i]));

    CHECK(cudaEventRecord(start, 0));
	
	// baseline case - sequential transfer and execute
    
	// asynchronous version 1: loop over {copy, kernel, copy}
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]));
    }
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        tabular_streams<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], iElem, ioffset);
    }

    // enqueue asynchronous transfers from the device
    for (int i = 0; i < NSTREAM; ++i) {
        int ioffset = i * iElem;
        CHECK(cudaMemcpyAsync(&gpuRef[ioffset], &d_A[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float execution_time;
    CHECK(cudaEventElapsedTime(&execution_time, start, stop));

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM, execution_time, (nBytes * 2e-6) / execution_time );
    printf(" speedup                : %f \n", ((itotal - execution_time) * 100.0f) / itotal);
	
	// asynchronous version 2: loop over copy, loop over kernel, loop over copy
	
	// cleanup

    // free device global memory
  CHECK(cudaFree(d_A));

  // free host memory
  CHECK(cudaFreeHost(h_A));
  CHECK(cudaFreeHost(hostRef));
  CHECK(cudaFreeHost(gpuRef));

  // destroy events
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  // destroy streams
  for (int i = 0; i < NSTREAM; ++i)
    CHECK(cudaStreamDestroy(stream[i]));

  CHECK(cudaDeviceReset());
	
	return 0;
}
