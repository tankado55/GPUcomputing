
#include <stdio.h>
#include "../utils/common.h"


// Dimensione del blocco
#define BDIMX 32
#define BDIMY 32

// macro x conversione indici lineari
#define INDEX(rows, cols, stride) (rows * stride + cols)

// prototipi funzioni
void initialData(float*, const int);
void printData(float*, int, int);
void checkResult(float*, float*, int, int);
void transposeHost(float*, float*, const int, const int);

/*
 * Kernel per il calcolo della matrice trasposta usando la shared memory
 */
__global__ void transposeSmem(float *out, float *in, int nrows, int ncols) {
	// static shared memory
	__shared__ float tile[BDIMY][BDIMX];

	// coordinate matrice originale
	//unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	//unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

	// trasferimento dati dalla global memory alla shared memory
	if (y < nrows && x < ncols)
		tile[threadIdx.y][threadIdx.x] = in[INDEX(y, x, ncols)];

	// thread synchronization
	__syncthreads();

	// offset blocco trasposto
	y = blockIdx.x * blockDim.x + threadIdx.y;
	x = blockIdx.y * blockDim.y + threadIdx.x;

	// controlli invertiti nelle dim riga colonna
	if (y < ncols && x < nrows)
		out[y*nrows + x] = tile[threadIdx.x][threadIdx.y];
}

//# naive: access data in rows
__global__ void copyRow(float *out, float *in, const int nrows,	const int ncols) {
	// matrix coordinate (ix,iy)
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	// transpose with boundary test
	if (row < nrows && col < ncols)
		out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
}

//# naive: access data in cols
__global__ void copyCol(float *out, float *in, const int nrows,	const int ncols) {
	// matrix coordinate (ix,iy)
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	// transpose with boundary test
	if (row < nrows && col < ncols)
		out[INDEX(row, col, ncols)] = in[INDEX(col, row, nrows)];
}

//# MAIN
int main(int argc, char **argv) {
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting transpose at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	bool iprint = 0;

	// set up array size
	int nrows = 1 << 14;
	int ncols = 1 << 14;

	if (argc > 1)
		iprint = atoi(argv[1]);
	if (argc > 2)
		nrows = atoi(argv[2]);
	if (argc > 3)
		ncols = atoi(argv[3]);

	printf("\nMatrice con nrows = %d ncols = %d\n", nrows, ncols);
	size_t ncells = nrows * ncols;
	size_t nBytes = ncells * sizeof(float);

	// allocate host memory
	float *A_h = (float *) malloc(nBytes);

	// Allocate Unified Memory – accessible from CPU or GPU
	float *A, *AT;
	cudaMallocManaged(&A, nBytes);
	cudaMallocManaged(&AT, nBytes);

	//  initialize host array
	initialData(A, nrows * ncols);
	if (iprint)
		printData(A, nrows, ncols);

	//  transpose at host side
	transposeHost(A_h, A, nrows, ncols);

	
  printf("*** KERNEL: col copy  ***\n");
	// tranpose gmem
  memset(AT, 0, nBytes);
  dim3 block(BDIMX, BDIMY, 1);
	dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y, 1);
	double iStart = seconds();
	copyCol<<<grid, block>>>(AT, A, nrows, ncols);
	CHECK(cudaDeviceSynchronize());
	double iElaps = seconds() - iStart;

	// check result
	checkResult(A_h, AT, nrows, ncols);

	double ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
	printf("col copy elapsed %f sec\n <<< grid (%d,%d) block (%d,%d)>>> "
			"effective bandwidth %f GB\n\n", iElaps, grid.x, grid.y, block.x,	block.y, ibnd);

  
  printf("*** KERNEL: row copy  ***\n");
	// tranpose gmem
  memset(AT, 0, nBytes);

	iStart = seconds();
	copyRow<<<grid, block>>>(AT, A, nrows, ncols);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;

	// check result
	checkResult(A_h, AT, nrows, ncols);

	ibnd = 2 * ncells * sizeof(float) / 1e9 / iElaps;
	printf("row copy elapsed %f sec\n <<< grid (%d,%d) block (%d,%d)>>> "
			"effective bandwidth %f GB\n\n", iElaps, grid.x, grid.y, block.x,	block.y, ibnd);



	printf("*** KERNEL: transposeSmem ***\n");
	// tranpose smem
	memset(AT, 0, nBytes);

	iStart = seconds();
	transposeSmem<<<grid, block>>>(AT, A, nrows, ncols);
	CHECK(cudaDeviceSynchronize());
	double iElapsSMEM = seconds() - iStart;

	if (iprint)
		printData(AT, ncols, nrows);

	checkResult(A_h, AT, nrows, ncols);
	ibnd = 2 * ncells * sizeof(float) / 1e9 / iElapsSMEM;
	printf("transposeSmem elapsed %f sec\n <<< grid (%d,%d) block (%d,%d)>>> "
			"effective bandwidth %f GB\n", iElapsSMEM, grid.x, grid.y, block.x,
			block.y, ibnd);

	printf("SPEEDUP = %f\n", iElaps/iElapsSMEM);

	// free host and device memory
	CHECK(cudaFree(A));
	CHECK(cudaFree(AT));
	free(A_h);

	// reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}

void initialData(float *in, const int size) {
	for (int i = 0; i < size; i++)
		in[i] = i; // (float)(rand()/INT_MAX) * 10.0f;
	return;
}

void printData(float *in, int nrows, int ncols) {
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++)
			printf("%3.0f ", in[INDEX(i, j, ncols)]);
		printf("\n");
	}
}

void transposeHost(float *out, float *in, const int nrows, const int ncols) {
	for (int iy = 0; iy < nrows; ++iy)
		for (int ix = 0; ix < ncols; ++ix)
			out[INDEX(ix, iy, nrows)] = in[INDEX(iy, ix, ncols)];
}

void checkResult(float *hostRef, float *gpuRef, int rows, int cols) {
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int index = INDEX(i, j, cols);
			if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
				match = 0;
				printf("different on (%d, %d) (offset=%d) element in "
						"transposed matrix: host %f gpu %f\n", i, j, index,
						hostRef[index], gpuRef[index]);
				break;
			}
		}
		if (!match)
			break;
	}

	if (!match)
		printf("Arrays do not match.\n");
}
