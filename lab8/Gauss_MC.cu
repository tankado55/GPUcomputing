
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "../GPUcomputing/utils/common.h"

#define TRIALS_PER_THREAD 10000
#define BLOCKS  264
#define THREADS 264
#define PI 3.1415926535 // known value of pi

float Gauss_CPU(long trials, float a, float b, float max) {
	long s = 0;
	for (long i = 0; i < trials; i++) {
		float x = (b-a)*(rand() / (float) RAND_MAX)+a;
		float y = (rand() / (float) RAND_MAX);
		s += (y <= expf(-x*x/2));
	}
	return s / (float)trials;
}

__global__ void Gauss_GPU() {
	//# TODO
}

int main(int argc, char *argv[]) {

	float host[BLOCKS * THREADS];
	float *dev;
	float a = -1;
	float b = 2;
	float max = 1.0f/sqrt(2*PI);
	float A = (b-a)*max;
	float P_true = 0.818594;

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// CPU procedure
	float iStart = seconds();
	long N = THREADS * BLOCKS * TRIALS_PER_THREAD;
	float P_cpu = Gauss_CPU(N,a,b,max);
	float iElaps = seconds() - iStart;
	P_cpu = P_cpu*A;
	printf("CPU elapsed time: %.5f (sec)\n", iElaps);
	printf("CPU estimate of P = %f [error of %f]\n", P_cpu, abs(P_cpu - P_true));

	// GPU procedure
    curandGenerator_t gen;
    // Create pseudo-random number generator 
	CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	
	//# TODO

	
	printf("GPU elapsed time: %.5f (sec)\n", seconds);
	printf("GPU estimate of P = %f [error of %f ]\n", P, abs(P - P_true));
	printf("Speedup = %f\n", iElaps/seconds);
	cudaFree(dev);
	cudaFree(devStates);
	return 0;
}
