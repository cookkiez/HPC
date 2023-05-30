/**
 * cuda solution for solving large systems of linear equations
*/

#include <math.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#include "../mtx_sparse.h"

void printHelp(const char *progName) {
  fprintf(stderr, "Usage: %s [-n iterations] [-e margin] FILE\n", progName);
}

__global__ void vecSumCoefGPU(double *vecOut, double *vecInA, double *vecInB, int n, double coef){
    // only to be used for addition
    // [TESTED] works on vector of any size
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    while (tid < n) {
        vecOut[tid] = vecInA[tid] + coef * vecInB[tid];
        tid += gridDim.x * blockDim.x;
    }

}

__global__ void vecSubtrCoefGPU(double *vecOut, double *vecInA, double *vecInB, int n, double coef){
    // only to be used for subtraction
    // [TESTED] works on vector of any size
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    while (tid < n) {
        vecOut[tid] = vecInA[tid] - coef * vecInB[tid];
        tid += gridDim.x * blockDim.x;
    }
}

void vecSumCoef(double *vecOut, double *vecInA, double *vecInB, bool subtract, int n, double coef) {
  for (int i = 0; i < n; i++)
    vecOut[i] = (subtract) ? (vecInA[i] - coef * vecInB[i]) : (vecInA[i] + coef * vecInB[i]);
}

__global__ void vecLinProdGPU(double *vecOut, double *vecIn, int n, double coef) {
    // [TESTED] works on vector of any size
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while (tid < n) {
        vecOut[tid] = coef * vecIn[tid];
        tid += gridDim.x * blockDim.x;
    }

}

void vecLinProd(double *vecOut, double *vecIn, int n, double coef) {
  for (int i = 0; i < n; i++)
    vecOut[i] = coef * vecIn[i];
}

__global__ void vecDotProductGPU(double *a, double *b, double *p, int n)
  // result in d_p, but needs to be summed!
  // [TESTED] 
  // GPU thread organization
  // dim3 gridsize((size-1)/threadsperblock + 1);
  // dim3 blocksize(threadsperblock);
  // at the end sum values of all blocks (for loop over gridsize.x)
{
    __shared__ double part[BLOCK_SIZE];

    part[threadIdx.x] = 0.0;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < n)
    {
        part[threadIdx.x] += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

	int floorPow2 = blockDim.x;
	while (floorPow2 & (floorPow2-1))
		floorPow2 &= floorPow2-1;

	if (blockDim.x != floorPow2)
	{
		if (threadIdx.x >= floorPow2)
			part[threadIdx.x - floorPow2] += part[threadIdx.x];
        __syncthreads();
	}

    int idxStep;
	for(idxStep = floorPow2 >> 1; idxStep > 32 ; idxStep >>= 1 )
	{
		if (threadIdx.x < idxStep)
			part[threadIdx.x] += part[threadIdx.x+idxStep];
        __syncthreads();
	}
	for( ; idxStep > 0 ; idxStep >>= 1 )
	{
		if (threadIdx.x < idxStep)
			part[threadIdx.x] += part[threadIdx.x+idxStep];
	}

    if (threadIdx.x == 0)
        p[blockIdx.x] = part[0];

}

double vecDotProduct(double *vecInA, double *vecInB, int n) {
  double prod = 0.f;
  for (int i = 0; i < n; i++)
    prod += vecInA[i] * vecInB[i];
  return prod;
}

void mtxVecProduct_JDS(double *vecOut, struct mtx_JDS *mtx, double *vecIn, int n) {
  // Init value of output vector
  for (int i = 0; i < n; i++)
    vecOut[i] = 0; // Set output vector to zero
  
  // Multiply each non zero
  int jag_start, jag_end, jag_index, row_in_jag, row_ptr = 0, row_offset;
  for (int s = mtx->max_el_in_row; s > 0; s--) { // Each jag
    jag_index = mtx->max_el_in_row - s;
    jag_start = mtx->jagged_ptr[jag_index]; // Start jag where pointer points to
    jag_end = (jag_index + 1 < mtx->max_el_in_row) ? mtx->jagged_ptr[jag_index + 1] : mtx->num_elements; // End jag where next pointer points to
    row_in_jag = (jag_end - jag_start) / s;
    
    for (int i = jag_start; i < jag_end; i++) {
      row_offset = ((i - jag_start) % row_in_jag);
      vecOut[mtx->row_permute[row_ptr + row_offset]] += mtx->data[i] * vecIn[mtx->col[i]];
    }
    row_ptr += row_in_jag;
  }
}

void mtxVecProduct_JDSrow(double *vecOut, struct mtx_JDS *mtx, double *vecIn, int n) {
  // Init value of output vector
  memset(vecOut, 0, n * sizeof(double)); // Set output vector to zero

  // Multiply each non zero
  for (int r = 0; r < mtx->max_el_in_row; r++) {
    for (int i = mtx->jagged_ptr[r]; i < ((r + 1 < mtx->max_el_in_row) ? mtx->jagged_ptr[r + 1] : mtx->num_nonzeros); i++) {
      if (mtx->row_permute[r] < n)
        vecOut[mtx->row_permute[r]] += mtx->data[i] * vecIn[mtx->col[i]];
    }
  }
}

int main(int argc, char *argv[]) {
  uint32_t iterations = 1000;
  double margin = 1e-8;
  bool showIntermediateResult = false;
  int c;
  while ((c = getopt(argc, argv, "n:e:s")) != -1) {
    switch (c) {
      case 'n':
        sscanf(optarg, "%o", &iterations);
        break;
      case 'e':
        sscanf(optarg, "%lf", &margin);
        break;
      case 's':
        showIntermediateResult = true;
        break;
      case 'h':
      case '?':
      default:
        printHelp(argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if (optind >= argc) {
    printHelp(argv[0]);
    exit(EXIT_FAILURE);
  }

  // Init random
  srand(time(NULL));

  // Open file with matrix A
  FILE *file;
  if ((file = fopen(argv[optind], "r")) == NULL) {
    fprintf(stderr, "Failed to open: %s \n", argv[optind]);
    exit(EXIT_FAILURE);
  }
  FILE *file2;
  if ((file2 = fopen(argv[optind], "r")) == NULL) {
    fprintf(stderr, "Failed to open: %s \n", argv[optind]);
    exit(EXIT_FAILURE);
  }
  // Create matrix structures
  struct mtx_COO mtxCOO, mtxCOO_t;
  struct mtx_CSR mtxCSR, mtxCSR_t;
  struct mtx_ELL mtxELL, mtxELL_t;
  struct mtx_JDS mtxJDS, mtxJDS_t;

  if (mtx_COO_create_from_file(&mtxCOO, file, false) != 0) {
    fprintf(stderr, "Failed to extract COO matrix from file\n");
    exit(EXIT_FAILURE);
  }
  if (mtx_COO_create_from_file(&mtxCOO_t, file2, true) != 0) {
    fprintf(stderr, "Failed to extract COO matrix from file\n");
    exit(EXIT_FAILURE);
  }
  
  mtx_CSR_create_from_mtx_COO(&mtxCSR, &mtxCOO);
  mtx_CSR_create_from_mtx_COO(&mtxCSR_t, &mtxCOO_t);
  mtx_ELL_create_from_mtx_CSR(&mtxELL, &mtxCSR);
  mtx_ELL_create_from_mtx_CSR(&mtxELL_t, &mtxCSR_t);
  mtx_JDS_create_from_mtx_CSR(&mtxJDS, &mtxCSR);
  mtx_JDS_create_from_mtx_CSR(&mtxJDS_t, &mtxCSR_t);

  /*printf("COO data: ");
  vecPrint(mtxCOO.data, mtxCOO.num_nonzeros);
  printf("COO col: ");
  vecPrintInt(mtxCOO.col, mtxCOO.num_nonzeros);
  printf("COO row: ");
  vecPrintInt(mtxCOO.row, mtxCOO.num_nonzeros);*/

  int vecSize = mtxCOO.num_cols;

  printf("Vec size: %d\n", vecSize);
  
  // Temporary
  double *vec_tmp = (double*)malloc(vecSize * sizeof(double));
  if (!vec_tmp) {
    fprintf(stderr, "Failed to allocate memory for vec_temp\n");

    // Memory cleanup
    mtx_COO_free(&mtxCOO);
    mtx_COO_free(&mtxCOO_t);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);
    mtx_JDS_free(&mtxJDS);
    mtx_JDS_free(&mtxJDS_t);

    exit(EXIT_FAILURE);
  }

  // Setup initial values

  // s = x_0 (for calculation b = As)
  double *vec_x = (double*)malloc(vecSize * sizeof(double));
  if (!vec_x) {
    fprintf(stderr, "Failed to allocate memory for vec_x\n");

    // Memory cleanup
    mtx_COO_free(&mtxCOO);
    mtx_COO_free(&mtxCOO_t);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);
    mtx_JDS_free(&mtxJDS);
    mtx_JDS_free(&mtxJDS_t);
    free(vec_tmp);

    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < vecSize; i++)
    vec_x[i] = 1; // Initial s
  //vec_x[0] = 2; vec_x[1] = 1;
  printf("Correct solution: ");
  vecPrint(vec_x, vecSize);
  
  // b = Ax_0 (generate constant)
  double *vec_b = (double*)malloc(vecSize * sizeof(double));
  if (!vec_b) {
    fprintf(stderr, "Failed to allocate memory for vec_b\n");

    // Memory cleanup
    mtx_COO_free(&mtxCOO);
    mtx_COO_free(&mtxCOO_t);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);
    mtx_JDS_free(&mtxJDS);
    mtx_JDS_free(&mtxJDS_t);
    free(vec_tmp);
    free(vec_x);

    exit(EXIT_FAILURE);
  }
  mtxVecProduct_JDS(vec_b, &mtxJDS, vec_x, vecSize);
  printf("Vector b: ");
  vecPrint(vec_b, vecSize);
  
  // x_0 = random
  for (int i = 0; i < vecSize; i++)
    vec_x[i] = rand() % 10 + 1; // Initial x_0 value (generate randomly [0, 9])
  printf("x_0: ");
  vecPrint(vec_x, vecSize);
  
  // r_0 = b - Ax_0
  double *vec_r = (double*)malloc(vecSize * sizeof(double));
  if (!vec_r) {
    fprintf(stderr, "Failed to allocate memory for vec_r\n");

    // Memory cleanup
    mtx_COO_free(&mtxCOO);
    mtx_COO_free(&mtxCOO_t);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);
    mtx_JDS_free(&mtxJDS);
    mtx_JDS_free(&mtxJDS_t);
    free(vec_tmp);
    free(vec_x);
    free(vec_b);

    exit(EXIT_FAILURE);
  }
  mtxVecProduct_JDS(vec_tmp, &mtxJDS, vec_x, vecSize); // Ax_0
  vecSumCoef(vec_r, vec_b, vec_tmp, true, vecSize, 1); // r = b - Ax_0
  
  // r_dash_0 = r_0
  double *vec_r_dash = (double*)malloc(vecSize * sizeof(double));
  if (!vec_r_dash) {
    fprintf(stderr, "Failed to allocate memory for vec_r_dash\n");

    // Memory cleanup
    mtx_COO_free(&mtxCOO);
    mtx_COO_free(&mtxCOO_t);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);
    mtx_JDS_free(&mtxJDS);
    mtx_JDS_free(&mtxJDS_t);
    free(vec_tmp);
    free(vec_x);
    free(vec_b);
    free(vec_r);

    exit(EXIT_FAILURE);
  }
  memcpy(vec_r_dash, vec_r, vecSize * sizeof(double)); 

  // p_0 = r_0
  double *vec_p = (double*)malloc(vecSize * sizeof(double));
  if (!vec_p) {
    fprintf(stderr, "Failed to allocate memory for vec_p\n");

    // Memory cleanup
    mtx_COO_free(&mtxCOO);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);
    mtx_JDS_free(&mtxJDS);
    mtx_JDS_free(&mtxJDS_t);
    free(vec_tmp);
    free(vec_x);
    free(vec_b);
    free(vec_r);
    free(vec_r_dash);

    exit(EXIT_FAILURE);
  }
  memcpy(vec_p, vec_r, vecSize * sizeof(double)); 

  // p_dash_0 = r_dash_0
  double *vec_p_dash = (double*)malloc(vecSize * sizeof(double));
  if (!vec_p_dash) {
    fprintf(stderr, "Failed to allocate memory for vec_p_dash\n");

    // Memory cleanup
    mtx_COO_free(&mtxCOO);
    mtx_COO_free(&mtxCOO_t);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);
    mtx_JDS_free(&mtxJDS);
    mtx_JDS_free(&mtxJDS_t);
    free(vec_tmp);
    free(vec_x);
    free(vec_b);
    free(vec_r);
    free(vec_r_dash);
    free(vec_p);

    exit(EXIT_FAILURE);
  }
  memcpy(vec_p_dash, vec_r_dash, vecSize * sizeof(double));

  // b_t * b
  double bb = vecDotProduct(vec_b, vec_b, vecSize);

  // r_t_0 * r_0
  double rtr = vecDotProduct(vec_r, vec_r, vecSize);

  // calculate margin^2
  margin = powf(margin, 2) * bb;


    // start CUDA timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

  // Main loop
  double alpha, beta, rr, rrn;
  uint32_t k;
  
  for (k = 0; k < iterations && rtr > margin; k++) {
    // r_dash_t_k * r_k
    rr = vecDotProduct(vec_r_dash, vec_r, vecSize);

    // Ap_k
    mtxVecProduct_JDS(vec_tmp, &mtxJDS, vec_p, vecSize);
    
    // alpha = (r_dash_t * r_t) / (p_dash_t * A * p)
    alpha = rr / vecDotProduct(vec_p_dash, vec_tmp, vecSize);
    // x_k+1 = x_k + alpha * p_k
    vecSumCoef(vec_x, vec_x, vec_p, false, vecSize, alpha);
    if (showIntermediateResult) {
      printf("x_k+1: ");
      vecPrint(vec_x, vecSize);
    }
    
    // r_k+1 = r_k - alpha * A * p_k
    vecSumCoef(vec_r, vec_r, vec_tmp, true, vecSize, alpha);

    // A_t p_Dash
    //printf("\n----TRANSPOSE Multiplication----\n");
    mtxVecProduct_JDS(vec_tmp, &mtxJDS_t, vec_p_dash, vecSize);
    
    // r_dash_k+1 = r_dash_k - alpha * A_t * p_dash_k
    vecSumCoef(vec_r_dash, vec_r_dash, vec_tmp, true, vecSize, alpha);
    
    // beta = (r_t_k+1 * r_k+1) / (r_t_k * r_k)
    rrn = vecDotProduct(vec_r_dash, vec_r, vecSize); // r_dash_t_k+1 * r_k+1
    beta = rrn / rr;
    rr = rrn;
    // r_t * r
    rtr = vecDotProduct(vec_r, vec_r, vecSize);

    // p_k+1 = r_k+1 + beta * p_k
    vecSumCoef(vec_p, vec_r, vec_p, false, vecSize, beta);
    
    // p_dash_k+1 = r_dash_k+1 + beta * p_dash_k
    vecSumCoef(vec_p_dash, vec_r_dash, vec_p_dash, false, vecSize, beta);
    //printf("TEST: %lf %.15lf\n", rtr, margin);
    //vecPrint(vec_r, vecSize);
  }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dtime = 0;
    cudaEventElapsedTime(&dtime, start, stop);

  // Print result
  printf("Iterations: %o/%o\nResult: ", k, iterations);
  vecPrint(vec_x, vecSize);
  printf("Elapsed: %.lf ms\n", dtime);
  
  // Clear memory
  mtx_COO_free(&mtxCOO);
  mtx_COO_free(&mtxCOO_t);
  mtx_CSR_free(&mtxCSR);
  mtx_CSR_free(&mtxCSR_t);
  mtx_ELL_free(&mtxELL);
  mtx_ELL_free(&mtxELL_t);
  mtx_JDS_free(&mtxJDS);
  mtx_JDS_free(&mtxJDS_t);
  free(vec_tmp);
  free(vec_b);
  free(vec_x);
  free(vec_r);
  free(vec_r_dash);
  free(vec_p);
  free(vec_p_dash);

  return 0;
}
