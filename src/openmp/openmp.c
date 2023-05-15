/**
 * OpenMP solution for solving large systems of linear equations
*/

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../mtx_sparse.h"

void printHelp(const char *progName) {
  fprintf(stderr, "Usage: %s [-n iterations] [-e margin] FILE\n", progName);
}

void vecPrint(float *vecIn, int n) {
  printf("[");
  for (int i = 0; i < n; i++) {
    if (i == 0)
      printf("%f", vecIn[i]);
    else
      printf(",%f", vecIn[i]);
  }
  printf("]\n");
}

void vecSumCoef(float *vecOut, float *vecInA, float *vecInB, bool subtract, int n, float coef) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++)
    vecOut[i] = (subtract) ? (vecInA[i] - coef * vecInB[i]) : (vecInA[i] + coef * vecInB[i]);
}

void vecLinProd(float *vecOut, float *vecIn, int n, float coef) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++)
    vecOut[i] = coef * vecIn[i];
}

float vecDotProduct(float *vecInA, float *vecInB, int n) {
  float prod = 0.f;
  #pragma omp parallel for reduction (+:prod)
  for (int i = 0; i < n; i++)
    prod += vecInA[i] * vecInB[i];
  return prod;
}

void mtxVecProduct_JDS(float *vecOut, struct mtx_JDS *mtx, float *vecIn, int n) {
  // Init value of output vector
  memset(vecOut, 0, n * sizeof(float)); // Set output vector to zero

  // Multiply each non zero
  #pragma omp parallel for
  for (int r = 0; r < mtx->max_el_in_row; r++) {
    for (int i = mtx->jagged_ptr[r]; i < ((r + 1 < mtx->max_el_in_row) ? mtx->jagged_ptr[r + 1] : mtx->num_nonzeros); i++) {
      if (mtx->row_permute[r] < n)
        vecOut[mtx->row_permute[r]] += mtx->data[i] * vecIn[mtx->col[i]];
    }
  }
}

int main(int argc, char *argv[]) {
  uint32_t iterations = 1000;
  float margin = 0.005;
  bool showIntermediateResult = false;

  int c;
  while ((c = getopt(argc, argv, "n:e:s")) != -1) {
    switch (c) {
      case 'n':
        sscanf(optarg, "%o", &iterations);
        break;
      case 'e':
        sscanf(optarg, "%f", &margin);
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

  printf("%o\n", iterations);

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
  if (mtx_COO_create_from_file(&mtxCOO, file, false) != 0)
    exit(EXIT_FAILURE);
  if (mtx_COO_create_from_file(&mtxCOO_t, file, true) != 0)
    exit(EXIT_FAILURE);

  mtx_CSR_create_from_mtx_COO(&mtxCSR, &mtxCOO);
  mtx_CSR_create_from_mtx_COO(&mtxCSR_t, &mtxCOO_t);
  mtx_ELL_create_from_mtx_CSR(&mtxELL, &mtxCSR);
  mtx_ELL_create_from_mtx_CSR(&mtxELL_t, &mtxCSR_t);
  mtx_JDS_create_from_mtx_CSR(&mtxJDS, &mtxCSR);
  mtx_JDS_create_from_mtx_CSR(&mtxJDS_t, &mtxCSR_t);

  int vecSize = mtxCOO.num_cols;

  // Temporary
  float *vec_tmp = (float*)malloc(vecSize * sizeof(float));

  // Setup initial values

  // s = x_0 (for calculation b = As)
  float *vec_x = (float*)malloc(vecSize * sizeof(float));
  for (int i = 0; i < vecSize; i++)
    vec_x[i] = 1; // Initial s
  //vec_x[0] = 2; vec_x[1] = 1;
  printf("Correct solution: ");
  vecPrint(vec_x, vecSize);

  // b = Ax_0 (generate constant)
  float *vec_b = (float*)malloc(vecSize * sizeof(float));
  mtxVecProduct_JDS(vec_b, &mtxJDS, vec_x, vecSize);

  // x_0 = random
  for (int i = 0; i < vecSize; i++)
    vec_x[i] = rand() % 10 + 1; // Initial x_0 value (generate randomly [0, 9])
  printf("x_0: ");
  vecPrint(vec_x, vecSize);

  // r_0 = b - Ax_0
  float *vec_r = (float*)malloc(vecSize * sizeof(float));
  mtxVecProduct_JDS(vec_tmp, &mtxJDS, vec_x, vecSize); // Ax_0
  vecSumCoef(vec_r, vec_b, vec_tmp, true, vecSize, 1); // r = b - Ax_0

  // r_dash_0 = r_0
  float *vec_r_dash = (float*)malloc(vecSize * sizeof(float));
  memcpy(vec_r_dash, vec_r, vecSize * sizeof(float)); 

  // p_0 = r_0
  float *vec_p = (float*)malloc(vecSize * sizeof(float));
  memcpy(vec_p, vec_r, vecSize * sizeof(float)); 

  // p_dash_0 = r_dash_0
  float *vec_p_dash = (float*)malloc(vecSize * sizeof(float));
  memcpy(vec_p_dash, vec_r_dash, vecSize * sizeof(float));

  // b_t * b
  float bb = vecDotProduct(vec_b, vec_b, vecSize);

  // r_t_0 * r_0
  float rtr = vecDotProduct(vec_r, vec_r, vecSize) / bb;

  // calculate margin^2
  margin = powf(margin, 2);
  
  double time_start = omp_get_wtime();

  // Main loop
  float alpha, beta, rr, rrn;
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
    mtxVecProduct_JDS(vec_tmp, &mtxJDS_t, vec_p_dash, vecSize);

    // r_dash_k+1 = r_dash_k - alpha * A_t * p_dash_k
    vecSumCoef(vec_r_dash, vec_r_dash, vec_tmp, true, vecSize, alpha);

    // beta = (r_t_k+1 * r_k+1) / (r_t_k * r_k)
    rrn = vecDotProduct(vec_r_dash, vec_r, vecSize); // r_dash_t_k+1 * r_k+1
    beta = rrn / rr;
    rr = rrn;

    // r_t * r
    rtr = vecDotProduct(vec_r, vec_r, vecSize) / bb;

    // p_k+1 = r_k+1 + beta * p_k
    vecSumCoef(vec_p, vec_r, vec_p, false, vecSize, beta);

    // p_dash_k+1 = r_dash_k+1 + beta * p_dash_k
    vecSumCoef(vec_p_dash, vec_r_dash, vec_p_dash, false, vecSize, beta);
  }

  double time_end = omp_get_wtime();

  // Print result
  printf("Iterations: %o\nResult: ", k);
  vecPrint(vec_x, vecSize);
  printf("Elapsed: %.5lf s\n", (double)(time_end - time_start));
  
  // Clear memory
  mtx_COO_free(&mtxCOO);
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
