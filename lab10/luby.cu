%%cuda --name Luby.cu

#include <iostream>
#include "coloring.h"
#include "../GPUcomputing/utils/graph/graph_d.h"
#include "../GPUcomputing/utils/common.h"

using namespace std;

#define THREADxBLOCK 128

Coloring* LubyGreedy(GraphStruct *str) {
	// set coloring struct

	Coloring* col;
	CHECK(cudaMallocManaged(&col, sizeof(Coloring)));
	uint n = str->nodeSize;
	col->uncoloredNodes = true;

	// cudaMalloc for arrays of struct Coloring
	CHECK(cudaMallocManaged( &(col->coloring), n * sizeof(uint)));
	memset(col->coloring,0,n);

	// allocate space on the GPU for the random states
	curandState_t* states;
	uint* weigths;
	cudaMalloc((void**) &states, n * sizeof(curandState_t));
	cudaMalloc((void**) &weigths, n * sizeof(uint));
	dim3 threads ( THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );
	uint seed = 0;
	init <<< blocks, threads >>> (seed, states, weigths, n);

	// start coloring (dyn. parall.)
	LubyJPcolorer <<< 1, 1 >>> (col, str, weigths);

	cudaFree(states);
	cudaFree(weigths);
	return col;
}

/**
 * find an IS
 */
__global__ void findIS (Coloring* col, GraphStruct *str, uint* weights) {
	
	// # TODO
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < str->nodeSize) {
            int max = 0;
			for (int i; i < str.edgeSize; i++) {
                if (weights[i] > weights[global_idx]) {
                    max = global_idx;
                }
            }
            if (str[global_idx] == max) {
                col->coloring[numOfColors] = global_idx;
            }
	}


}


/**
 *  this GPU kernel takes an array of states, and an array of ints, and puts a random int into each
 */
__global__ void init (uint seed, curandState_t* states, uint* numbers, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > n)
			return;
	curand_init(seed, idx, 0, &states[idx]);
	numbers[idx] = curand(&states[idx])%n*n;
}


/**
 * Luby IS & Jones−Plassmann colorer
 */
__global__ void LubyJPcolorer (Coloring* col, GraphStruct *str, uint* weights) {
	dim3 threads (THREADxBLOCK);
	dim3 blocks ((str->nodeSize + threads.x - 1) / threads.x, 1, 1 );

	// loop on ISs covering the graph
	col->numOfColors = 0;
	while (col->uncoloredNodes) {
		col->uncoloredNodes = false;
		col->numOfColors++;
		findIS <<< blocks, threads >>> (col, str, weights);
		cudaDeviceSynchronize();
	}
}

/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void printColoring (Coloring* col, GraphStruct* str, bool verbose) {
	node n = str->nodeSize;
	cout << "** Graph (num node: " << n << ", num edges: " << str->edgeSize << ")" << endl;
	cout << "** Coloring (num colors: " << col->numOfColors << ")" << endl;
	if (verbose) {
		for (int i = 1; i <= col->numOfColors; i++) {
			cout << "   color(" << i << ")" << "-> ";
			for (int j = 0; j < n; j++)
				if (col->coloring[j] == i)
					cout << j << " ";
			cout << "\n";
		}
		cout << "\n";
	}
}


