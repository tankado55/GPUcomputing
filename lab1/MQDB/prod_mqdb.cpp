#include "mqdb.h"

void mqdbProd(mqdb A, mqdb B, mqdb C) {
	int n = 0;
	for (uint i = 0; i < A.nBlocks; i++)
		n += A.blkSize[i];
	
	uint blockOffset = 0;
  for (uint i = 0; i < A.nBlocks; i++) {
			//printf("mydebug: %d\n", blockOffset);		
      for (uint r = blockOffset; r < A.blkSize[i] + blockOffset; r++) {
					//printf("mydebug r: %d\n", r);					 
        for (uint c = blockOffset; c < A.blkSize[i] + blockOffset; c++) {
			    double sum = 0;
			    for (uint l = 0; l < A.blkSize[i]; l++){
				    double a = A.elem[r * n + blockOffset + l];
				    double b = B.elem[(l + blockOffset) * n + c];
				    sum += a*b;
			    }
			  C.elem[r * n + c] = (float)sum;				
		  }
    }
		blockOffset = blockOffset + A.blkSize[i];
  }	
		
}