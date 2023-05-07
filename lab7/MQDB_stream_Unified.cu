
#include "../utils/MQDB/mqdb.h"
#include "../utils/common.h"

#define BLOCK_SIZE 16     // block size
#define TEST_CPU 0

/*
 * Kernel for standard (naive) matrix product
 */
__global__ void matProdKernel(mqdb *A, mqdb *B, mqdb *C, int n) {
	// row & col indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < n) && (col < n)) {
		float val = 0;
		for (int k = 0; k < n; k++)
			val += A->elem[row * n + k] * B->elem[k * n + col];
		C->elem[row * n + col] = val;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb
 */
__global__ void mqdbBlockProd(mqdb *A, mqdb *B, mqdb *C, uint sdim, uint d, uint n) {
	//printf("%d/n", sdim);
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	

	// jump to the right block sub-matrix
	uint  offset = (n+1)*sdim;

	// each thread computes an entry of the product matrix
	if ((row < d) && (col < d)) {
		float val = 0;
		for (int k = 0; k < d; ++k)
			val += A->elem[row * n + k + offset] * B->elem[k * n + col + offset];
		C->elem[row * n + col + offset] = val;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb
 */
__global__ void mqdbBlockProdStream(mqdb *A, mqdb *B, mqdb *C) {
	//printf("%d/n", sdim);
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	/*
	int d = A


	// each thread computes an entry of the product matrix
	if ((row < d) && (col < d)) {
		float val = 0;
		for (int k = 0; k < d; ++k)
			val += A->elem[row * n + k] * B->elem[k * n + col];
		C->elem[row * n + col] = val;
	}*/
}


/*
 * Test on MQDB kernels using Unified Memory
 */
void testKernelsMQDB_unified(uint n, uint k, cudaEvent_t start, cudaEvent_t stop) {

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid  ((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
	mqdb *A_u, *B_u, *C_u_naive, *C_u_mqdb, *C_u_stream;

	//A_u = mqdbConst(n, k, 0, 1);
	//B_u = mqdbConst(n, k, 0, 1);
	//C_u_naive = mqdbConst(n, k, 0, 1);
	//C_u_mqdb = mqdbConst(n, k, 0, 1);
	//C_u_stream = mqdbConst(n, k, 0, 1);
	
	cudaMallocManaged(&A_u, sizeof(mqdb));
	cudaMallocManaged(&B_u, sizeof(mqdb));
	cudaMallocManaged(&C_u_naive, sizeof(mqdb));
	cudaMallocManaged(&C_u_mqdb, sizeof(mqdb));
	cudaMallocManaged(&C_u_stream, sizeof(mqdb));

	cudaMallocManaged(&(A_u->blkSize), k * sizeof(int));
	cudaMallocManaged(&(A_u->elem), n * n * sizeof(float));
	cudaMallocManaged(&(B_u->blkSize), k * sizeof(int));
	cudaMallocManaged(&(B_u->elem), n * n * sizeof(float));
	cudaMallocManaged(&(C_u_naive->blkSize), k * sizeof(int));
	cudaMallocManaged(&(C_u_naive->elem), n * n * sizeof(float));
	cudaMallocManaged(&(C_u_mqdb->blkSize), k * sizeof(int));
	cudaMallocManaged(&(C_u_mqdb->elem), n * n * sizeof(float));
	cudaMallocManaged(&(C_u_stream->blkSize), k * sizeof(int));
	cudaMallocManaged(&(C_u_stream->elem), n * n * sizeof(float));

	// matrix instance generation - Unified Memory

    genRandDimsUnified(A_u, n, k, 0);
    genRandDimsUnified(B_u, n, k, 0);
    genRandDimsUnified(C_u_naive, n, k, 0);
    genRandDimsUnified(C_u_mqdb, n, k, 0);
    genRandDimsUnified(C_u_stream, n, k, 0);
	
    // random fill mat entries
	fillBlocksUnified(A_u, n, k, 'C', 1);
	fillBlocksUnified(B_u, n, k, 'C', 1);
	fillBlocksUnified(C_u_naive, n, k, 'C', 1);
	fillBlocksUnified(C_u_mqdb, n, k, 'C', 1);
	fillBlocksUnified(C_u_stream, n, k, 'C', 1);

    
  

	/***********************************************************/
	/*                     GPU mat product                     */
	/***********************************************************/
	
  printf("Kernel (naive) mat product...\n");
  CHECK(cudaEventRecord(start, 0));
  matProdKernel<<<grid, block>>>(A_u, B_u, C_u_naive, n);
  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  float naive_time;
  CHECK(cudaEventElapsedTime(&naive_time, start, stop));
	
	/***********************************************************/
	/*                     GPU MQDB product                    */
	/***********************************************************/
	
  	printf("Kernel MQDB product...\n");
  	CHECK(cudaEventRecord(start, 0));
  	int sdim = 0;

	for (int i = 0; i < k; ++i) {
		int currentBlockSize = A_u->blkSize[i];
		dim3 gridMQDB((currentBlockSize + block.x - 1) / block.x, (currentBlockSize + block.y - 1) / block.y);
		mqdbBlockProd<<<gridMQDB, block>>>(A_u, B_u, C_u_mqdb, sdim, currentBlockSize, n);
		sdim += currentBlockSize;
		cudaDeviceSynchronize();
	}

	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	float mqdb_time;
	CHECK(cudaEventElapsedTime(&mqdb_time, start, stop));
	

  	/***********************************************************/
	/*             GPU MQDB product using streams              */
	/***********************************************************/
	
	printf("Kernel MQDB product using streams...\n");
	cudaStream_t stream[k];
	
	for (int i = 0; i < k; ++i) {
		CHECK(cudaStreamCreate(&stream[i]));
	}

  	CHECK(cudaEventRecord(start, 0));

	sdim = 0;
	for (int i = 0; i < k; ++i) {
		uint  offset = (n+1)*sdim;
		int currentBlockSize = A_u->blkSize[i];
		dim3 gridMQDB((currentBlockSize + block.x - 1) / block.x, (currentBlockSize + block.y - 1) / block.y);
		mqdbBlockProdStream<<<gridMQDB, block, 0, stream[i]>>>(A_u + offset, B_u + offset, C_u_stream + offset);
		sdim += currentBlockSize;
		//cudaDeviceSynchronize();
	}

	// clean up streams and events
	CHECK(cudaEventRecord(stop, 0));
  	CHECK(cudaEventSynchronize(stop));
  	float stream_time;
  	CHECK(cudaEventElapsedTime(&stream_time, start, stop));

	// check kernel error
  	CHECK(cudaGetLastError());

	// check device results
  	checkResult(*C_u_naive, *C_u_mqdb);
  	checkResult(*C_u_naive, *C_u_stream);
  	checkResult(*C_u_mqdb, *C_u_stream);
	

  	// destroy streams
  	for (int i = 0; i < k; ++i)
    	CHECK(cudaStreamDestroy(stream[i]));

	// free device global memory
  	CHECK(cudaFree(&A_u->blkSize));
  	CHECK(cudaFree(&A_u->elem));
  	CHECK(cudaFree(&B_u->blkSize));
  	CHECK(cudaFree(&B_u->elem));

  	CHECK(cudaDeviceReset());
}

/*
 * main function
 */
int main(int argc, char *argv[]) {
  
  // set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting mqdb product at ", argv[0]);
	printf("device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint n = 64; //8*1024;         // matrix size
	uint min_k = 20;         // min num of blocks
	uint max_k = 30;         // max num of blocks

	// multiple tests for k = # diag blocks
	for (uint k = min_k; k <= max_k; k+=5) {
		printf("\n*****   k = %d --- (avg block size = %f)\n",k,(float)n/k);
		testKernelsMQDB_unified(n, k, start, stop);
	}

    cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}


