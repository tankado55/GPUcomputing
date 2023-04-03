#include "../utils/common.h"

void initialData(float *ip, const int size) {
  int i;

  for (i = 0; i < size; i++)
    ip[i] = (float)( rand() & 0xFF ) / 10.0f;
  return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++)
      ic[ix] = ia[ix] + ib[ix];

    ia += nx;
    ib += nx;
    ic += nx;
  }
  return;
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
      break;
    }
  }

  if (!match)
    printf("Arrays do not match.\n\n");
}

//# grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx, int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;

  if (ix < nx && iy < ny)
    MatC[idx] = MatA[idx] + MatB[idx];
}

//# MAIN
int main(int argc, char **argv) {
  printf("%s Starting ", argv[0]);

  // set up data size of matrix
  int nx, ny;
  int ishift = 14;
  if  (argc > 1) ishift = atoi(argv[1]);
  nx = ny = 1 << ishift;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  //# malloc unified host memory
  
  // TODO
  float *d_MatA, *d_MatB, *d_MatC;
  cudaMallocManaged(&d_MatA, nBytes);
  cudaMallocManaged(&d_MatB, nBytes);
  cudaMallocManaged(&d_MatC, nBytes);

  //# invoke kernel at host side
  int dimx = 32;
  int dimy = 32;
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  //# after warm-up, time with unified memory
  double iStart = seconds();
  
  // INVOKE THE KERNEL
  sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  CHECK(cudaDeviceSynchronize()); 

  double iElaps = seconds() - iStart;
  printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps, grid.x, grid.y, block.x, block.y);
  //# check kernel error
  CHECK(cudaGetLastError());

  //# free device global memory
  CHECK(cudaFree(d_MatA));
  CHECK(cudaFree(d_MatB));
  //CHECK(cudaFree(gpuRef));

  // reset device
  CHECK(cudaDeviceReset());

  return (0);
}