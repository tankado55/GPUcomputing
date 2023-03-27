<<<<<<< HEAD
#include <stdio.h>
#include <iostream>
#include <unistd.h>


using namespace std;

__global__ void helloFromGPU (void) {
  int tID = threadIdx.x;
  printf("Hello World from GPU (I'am thread %d)!\n", tID);
}

int main(void) {
<<<<<<< HEAD
  //# hello from GPU 
  cout << "Hello World from CPU!" << endl;
  cudaSetDevice(0);
  helloFromGPU <<<1, 10>>>();
  cudaDeviceSynchronize();
  sleep(10);
  return 0;
=======
    // hello from GPU 
    cout << "Hello World from CPU!" << endl;
    cudaSetDevice(1);
    helloFromGPU <<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
>>>>>>> bd4ceea (a lot)
=======
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void helloFromGPU (void) {
    int tID = threadIdx.x;
    printf("Hello World from GPU (I'am thread = %d)!\n", tID);
}

int main(void) {
    // hello from GPU 
    cout << "Hello World from CPU!" << endl;
    cudaSetDevice(0);
    helloFromGPU <<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
>>>>>>> main
}