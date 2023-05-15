
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "../GPUcomputing/utils/common.h"

#define IDX2R(r,c,D) ( r * D + c) 
#define IDX2C(r,c,D) ( c * D + r )

#define N          (1<<10)

void generate_random_vector(int, double**);
void generate_rand_posdefinite_mat(int, double**);
void plot_mat(int, double*, char);
void plot_vec(int, double*, char); 
double norm2(int, double *);

/*
 * This sample implements a conjugate gradient solver on GPU using CUBLAS
 */
int main(int argc, char **argv) {
  int n = N;
	double *A, *dA;      // matrix N x N  (square)
	double *x, *dx;      // vector N x 1 
	double *b, *db;      // vector N x 1
	double *dr, *dr1;    // vector N x 1
	double *dp;          // vector N x 1
	double *dAxp, *dAxr; // vector N x 1
	
	cublasHandle_t handle;
	device_name();

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Generate instance: matrix A and vector b
	srand(time(NULL));
	generate_rand_posdefinite_mat(n, &A);      // random symmetric matrix A
	generate_random_vector(n, &b);            // random verctor b
	generate_random_vector(n, &x);            // random initial solution
	//plot_mat(n, A,'A');

	// Allocate device memory
	CHECK(cudaMalloc((void **)&dA, n * n * sizeof(double)));
	CHECK(cudaMalloc((void **)&dx, n * sizeof(double)));
	CHECK(cudaMalloc((void **)&db, n * sizeof(double)));
	CHECK(cudaMalloc((void **)&dr, n * sizeof(double)));
	CHECK(cudaMalloc((void **)&dr1, n * sizeof(double)));
	CHECK(cudaMalloc((void **)&dp, n * sizeof(double)));
	CHECK(cudaMalloc((void **)&dAxp, n * sizeof(double)));
	CHECK(cudaMalloc((void **)&dAxr, n * sizeof(double)));

	// Create the cuBLAS handle
	CHECK_CUBLAS(cublasCreate(&handle));
	int version;
	CHECK_CUBLAS(cublasGetVersion(handle, &version));
	printf("Using CUBLAS Version: %d\n", version);
	
	// Transfer inputs to the device, column-major order
	CHECK_CUBLAS(cublasSetMatrix(n, n, sizeof(double), A, n, dA, n));
	CHECK_CUBLAS(cublasSetVector(n, sizeof(double), b, 1, db, 1));
	CHECK_CUBLAS(cublasSetVector(n, sizeof(double), x, 1, dx, 1));

	// CG
	double beta = 0.0f;
	double alpha = 0.0f;
	double one = 1.0f, minusOne = -1.0f, zero = 0.0f;
	double num, den = 0, tmp;
	int k = 0, maxit = 2000;

    // my
    double alpha[n];

    //# r0 = b
    cublasScopy(handle, n, r, 1, b, 1);

						//# r0 = b − 𝐴∗x0   
    CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, d_A, n, dx, 1, &beta, dAxp, 1));  //dAxp é un vector

	                      //# p0 = r0
    double *p0 = r0;

    for (int i = 0; i < n; ++i) {
        //# p𝑘^𝑇 ∗ r𝑘 (num)
        cublasDdot(handle, CUBLAS_OP_T, dp, 1, dr, 1, &num);
		
        //# 𝐴 ∗ p𝑘
        cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, dA, n, dp, 1, &beta, dr1, 1);
		
        //# p𝑘^𝑇 ∗ 𝐴 ∗ p𝑘  (den)
        cublasDdot(handle, CUBLAS_OP_T, pd[n * i], 1, dr1, 1, den);

		//# 𝛼𝑘 = num/den
        a[i] = num/den;
		
        //# x(𝑘+1) = x𝑘 + 𝛼𝑘 * p𝑘

			                //# r(𝑘+1) = b − 𝛼𝑘 * 𝐴 ∗ p𝑘
								  		//# 𝐴 ∗ r(𝑘+1)     
			                //# p𝑘^𝑇 ∗ 𝐴 ∗ r(𝑘+1)  (num)
			                //# 𝛽𝑘 = num/den
			                 //# r1 = r(𝑘+1)
			               	//# r1 = r1 - 𝛽𝑘 * p𝑘
			               	 //# p(𝑘+1) = r(𝑘+1) - 𝛽𝑘 * p𝑘
    }
		                    
		


	// final solution
	double *y = (double *) malloc(sizeof(double) * n);
	CHECK_CUBLAS(cublasGetVector(n, sizeof(double), dx, 1, x, 1));
	cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, dA, n, dx, 1, &zero, db, 1);   // b = 𝐴∗𝑥 
	CHECK_CUBLAS(cublasGetVector(n, sizeof(double), db, 1, y, 1));                // y = b (approx solution)

	//plot norms of the vectors
	printf("norm b = %f\n", norm2(n, b));   
	printf("norm y = %f\n", norm2(n, y));   
	//plot_vec(n, b, 'b');
	//plot_vec(n, y, 'y');

  free(A);
  free(x);
  free(b);
  cudaFree(dA);
  cudaFree(dx);
	cudaFree(db);
  cudaFree(dr);
	cudaFree(dr1);
	cudaFree(dp);
	cudaFree(dAxp);
	cudaFree(dAxr);
}

void generate_rand_posdefinite_mat(int n, double **A) {
	double *a = (double *) malloc(sizeof(double) * n * n);
	double *r = (double *) malloc(sizeof(double) * n * n);

	// generate a random matrix
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) 
			r[i*n+j] = (double)rand() / RAND_MAX;
		
	// compute the product with its transpose (positive definite matrix)
	for (int i = 0; i < n; i++)
  	for (int j = i; j < n; j++) {
   		a[i*n+j] = 0;
   		for (int k = 0; k < n; k++) 
    		a[i*n+j] += r[i*n+k]*r[j*n+k];
			a[j*n+i] = a[i*n+j];
  	}
	*A = a;
}

void plot_mat(int n, double *A, char name) {
  printf("\nShow mat %c...\n", name);
	for(int r = 0; r < n; ++r){
    for(int c = 0; c < n; ++c)
			printf("%4.1f ", A[IDX2R(r,c,n)]);
    printf("\n");
	} 
  printf("\n");
}

double norm2(int n, double *x) {
	double norm = 0;
	for(int i = 0; i < n; ++i)
		norm += x[i]*x[i];
	norm = sqrt(norm);
  return norm; 
}

void plot_vec(int n, double *x, char name) {
  printf("\nShow vec %c...\n", name);
	for(int i = 0; i < n; ++i)
			printf("%4.1f ", x[i]);
  printf("\n");
}

void generate_random_vector(int n, double **x) {
	double *z = (double *) malloc(sizeof(double) * n);

	for (int i = 0; i < n; i++)
		z[i] = (double)rand() / RAND_MAX;
	*x = z;
}