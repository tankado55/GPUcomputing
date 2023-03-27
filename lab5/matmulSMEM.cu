#include <stdio.h>
#include <stdlib.h>
#include "../utils/common.h"

#define IDX(i,j,n) (i*n+j)
#define ABS(x,y) (x-y>=0?x-y:y-x)
#define N 1024
#define P 1024
#define M 1024
#define BLOCK_SIZE 32


/*
 * Kernel for matrix product with static SMEM
 *      C  =  A  *  B
 *    (NxM) (MxP) (PxM)
 */
__global__ void matmulSMEMstatic(float* A, float* B, float* C) {
	
	//# TODO

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int smemA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int smemB[BLOCK_SIZE][BLOCK_SIZE];
    float sum = 0.0;
    
    for (int i = 0; i < (P + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        // 2D to linear: skippo righe + colonne
        smemA[threadIdx.y][threadIdx.x] = A[(row * (P + BLOCK_SIZE - 1) / BLOCK_SIZE) + threadIdx.y + (i * BLOCK_SIZE)];
        // setto colonna, skippo righe locali, skippo per ciclo
        smemB[threadIdx.y][threadIdx.x] = B[col + (threadIdx.x * (P + BLOCK_SIZE - 1) / BLOCK_SIZE) + i * BLOCK_SIZE];

        __syncthreads();
        

        for (int i = 0; i < blockDim.x; ++i){
            sum += (smemA[threadIdx.x][i] * smemB[i][threadIdx.y]);
        }
    }
 
    
	
    __syncthreads();

    C[row * N + col] = sum;


}

/*
 * Kernel for matrix product using dynamic SMEM
 */
__global__ void matmulSMEMdynamic(float* A, float* B, float* C, const uint SMEMsize) {
	
	//# TODO

    // indexes

}

/*
 * Kernel for naive matrix product
 */
__global__ void matmul(float* A, float* B, float* C) {
	// indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < N) && (col < M)) {
		float sum = 0;
		for (int k = 0; k < P; k++)
			sum += A[IDX(row, k, P)] * B[IDX(k, col, M)];
		C[IDX(row, col, M)] = sum;
	}
}

/*
 *  matrix product on CPU
 */
void matmulCPU(float* A, float* B, float* C) {
	for (int row = 0; row < N; row++)
		for (int col = 0; col < M; col++) {
			float sum = 0;
			for (int k = 0; k < P; k++)
				sum += A[IDX(row, k, P)] * B[IDX(k, col, M)];
			C[IDX(row, col, M)] = sum;
		}
}

/*
 * Test the device
 */
unsigned long testCUDADevice(void) {
	int dev = 0;

	cudaDeviceSetCacheConfig (cudaFuncCachePreferEqual);
	cudaDeviceProp deviceProp;
	cudaSetDevice(dev);
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device %d: \"%s\"\n", dev, deviceProp.name);
	printf("Total amount of shared memory available per block: %lu KB\n",
			deviceProp.sharedMemPerBlock / 1024);
	return deviceProp.sharedMemPerBlock;
}

/*
 * elementwise comparison between two mqdb
 */
void checkResult(float *A, float *B) {
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N*M; i++)
		if (ABS(A[i], B[i]) > epsilon) {
			match = 0;
			printf("   * Arrays do not match!\n");
			break;
		}
	if (match)
		printf("   Arrays match\n\n");
}

/*
 * MAIN
 */
int main(void) {
	 // Kernels for matrix product
	 //      C  =  A  *  B
	 //    (NxM) (NxP) (PxM)
	printf("N = %d, M = %d, K = %d\n", N, M, P);
	uint rowA = N, rowB = P;
	uint colA = P, colB = M;
	uint rowC = N, colC = M;
	float *A, *B, *C, *C1;
	float *dev_A, *dev_B, *dev_C;

	// dims
	unsigned long Asize = rowA * colA * sizeof(float);
	unsigned long Bsize = rowB * colB * sizeof(float);
	unsigned long Csize = rowC * colC * sizeof(float);
	
	// malloc host memory
	A = (float*) malloc(Asize);
	B = (float*) malloc(Bsize);
	C = (float*) malloc(Csize);
	C1 = (float*) malloc(Csize);

	// device SMEM available/ test device shared memory
	unsigned long maxSMEMbytes = testCUDADevice();
	

	// malloc device memory
	CHECK(cudaMalloc((void** )&dev_A, Asize));
	CHECK(cudaMalloc((void** )&dev_B, Bsize));
	CHECK(cudaMalloc((void** )&dev_C, Csize));
	printf("Total amount of allocated memory on GPU %.2f MB\n\n", (float)(Asize + Bsize + Csize)/(1024.0*1024.0));

	// fill the matrices A and B
	for (int i = 0; i < N * P; i++) A[i] = 1.0;
	for (int i = 0; i < P * M; i++) B[i] = 1.0;

	/***********************************************************/
	/*                       CPU matmul                       */
	/***********************************************************/
	printf("\n   *** CPU & NAIVE KERNEL ***\n\n");
	double start = seconds();
	matmulCPU(A, B, C);
	printf("   matmul elapsed time CPU = %f\n\n", seconds() - start);


	// copy matrices A and B to the GPU
	CHECK(cudaMemcpy(dev_A, A, Asize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_B, B, Bsize, cudaMemcpyHostToDevice));

	/***********************************************************/
	/*                    GPU naive matmul                     */
	/***********************************************************/
	// grid block dims = shared mem dims = BLOCK_SIZE
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	start = seconds();
	matmul<<<grid, block>>>(dev_A, dev_B, dev_C);
	CHECK(cudaDeviceSynchronize());
	printf("   Kernel naive matmul elapsed time GPU = %f\n", seconds() - start);

	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C1, dev_C, Csize, cudaMemcpyDeviceToHost));
	checkResult(C,C1);
    CHECK(cudaMemset((void *) dev_C, 0, Csize));

	/***********************************************************/
	/*              GPU matmulSMEM static SMEM               */
	/***********************************************************/
	// grid block dims = shared mem dims = BLOCK_SIZE
	printf("\n   *** USING STATIC SMEM ***\n\n");
	start = seconds();
	matmulSMEMstatic<<<grid, block>>>(dev_A, dev_B, dev_C);
	CHECK(cudaDeviceSynchronize());
	printf("   Kernel matmulSMEM static elapsed time GPU = %f\n", seconds() - start);
	
	// amount of SMEM used
	uint SMEMsize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
		if (maxSMEMbytes < SMEMsize)
			printf("   Shared memory usage WARNING: available: %lu, required: %d bytes\n",	maxSMEMbytes, SMEMsize);
		else
			printf("   Total amount of shared memory required per block %.1f KB\n", (float) SMEMsize / (float) 1024);
	
	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C1, dev_C, Csize, cudaMemcpyDeviceToHost));
	checkResult(C,C1);
    CHECK(cudaMemset((void *) dev_C, 0, Csize));
	
	/***********************************************************/
	/*            GPU matmulSMEMD dynamic SMEM                */
	/***********************************************************/
	// set cache size
	cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
	printf("\n   *** USING DYNAMIC SMEM ***\n\n");

	// try with various SMEM sizes
	uint sizes[] = {8, 16, 32};
	for (int i = 0; i < 3; i++) {
		uint blockSize = sizes[i];
		block.x = blockSize;
		block.y = blockSize;
		grid.x = (M + block.x - 1) / block.x;
		grid.y = (N + block.y - 1) / block.y;
		uint SMEMsize = blockSize * blockSize;
		uint SMEMbyte = 2 * SMEMsize * sizeof(float);
		start = seconds();
		matmulSMEMdynamic<<< grid, block, SMEMbyte >>>(dev_A, dev_B, dev_C, SMEMsize);
		CHECK(cudaDeviceSynchronize());
		printf("   Kernel matmulSMEM dynamic (SMEM size %d) elapsed time GPU = %f\n", blockSize, seconds() - start);

		// amount of SMEM used
		if (maxSMEMbytes < SMEMbyte)
			printf("   Shared memory usage WARNING: available: %lu, required: %d bytes\n",	maxSMEMbytes, SMEMbyte);
		else
			printf("   Total amount of shared memory required per block %.1f KB\n", (float) SMEMbyte / (float) 1024);

		// copy the array 'C' back from the GPU to the CPU
		CHECK(cudaMemcpy(C1, dev_C, Csize, cudaMemcpyDeviceToHost));
		checkResult(C,C1);
        CHECK(cudaMemset((void *) dev_C, 0, Csize));
	}

	// free the memory allocated on the GPU
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
