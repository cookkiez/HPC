/**
 * Sequential solution for solving large systems of linear equations
*/

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

void vecSumCoef(float *vecOut, float *vecInA, float *vecInB, bool subtract, int n, float coef) {
  for (int i = 0; i < n; i++)
    vecOut[i] = (subtract) ? (vecInA[i] - coef * vecInB[i]) : (vecInA[i] + coef * vecInB[i]);
}

float vecDotProduct(float *vecInA, float *vecInB, int n) {
  float prod = 0.f;
  for (int i = 0; i < n; i++)
    prod += vecInA[i] * vecInB[i];
  return prod;
}

void mtxVecProduct_JDS(float *vecOut, struct mtx_JDS *mtx, float *vecIn, int n) {
  // Init value of output vector
  memset(vecOut, 0, n * sizeof(float)); // Set output vector to zero

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
  float margin = 0.005;

  int c;
  while ((c = getopt(argc, argv, "n:e:")) != -1) {
    switch (c) {
      case 'n':
        sscanf(optarg, "%o", &iterations);
        break;
      case 'e':
        sscanf(optarg, "%f", &margin);
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

  // Create matrix structures
  struct mtx_COO mtxCOO, mtxCOO_t;
  struct mtx_CSR mtxCSR, mtxCSR_t;
  struct mtx_ELL mtxELL, mtxELL_t;
  struct mtx_JDS mtxJDS, mtxJDS_t;
  if (mtx_COO_create_from_file(&mtxCOO, file) != 0)
    exit(EXIT_FAILURE);
  
  // Create tansposed matrix
  mtxCOO_t.col = mtxCOO.row;
  mtxCOO_t.row = mtxCOO.col;
  mtxCOO_t.data = mtxCOO.data;
  mtxCOO_t.num_nonzeros = mtxCOO.num_nonzeros;
  mtxCOO_t.num_cols = mtxCOO.num_rows;
  mtxCOO_t.num_rows = mtxCOO.num_cols;

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

  // b = Ax_0 (generate constant)
  float *vec_b = (float*)malloc(vecSize * sizeof(float));
  mtxVecProduct_JDS(vec_b, &mtxJDS, vec_x, vecSize);

  // x_0 = random
  for (int i = 0; i < vecSize; i++)
    vec_x[i] = rand() % 10; // Initial x_0 value (generate randomly [0, 9])

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

  // r_dash_t_0 * r_0
  float rr = vecDotProduct(vec_r_dash, vec_r, vecSize), rrn;

  // Main loop
  float alpha, beta; 
  for (uint32_t k = 0; k < iterations && rr > margin; k++) {
    // Ap_k
    mtxVecProduct_JDS(vec_tmp, &mtxJDS, vec_p, vecSize);

    // alpha = (r_dash_t * r_t) / (p_dash_t * A * p)
    alpha = rr / vecDotProduct(vec_p_dash, vec_tmp, vecSize);

    // x_k+1 = x_k + alpha * p_k
    vecSumCoef(vec_x, vec_x, vec_p, false, vecSize, alpha);

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

    // p_k+1 = r_k+1 + beta * p_k
    vecSumCoef(vec_p, vec_r, vec_p, false, vecSize, beta);

    // p_dash_k+1 = r_dash_k+1 + beta * p_dash_k
    vecSumCoef(vec_p_dash, vec_r_dash, vec_p_dash, false, vecSize, beta);
  }

  // Print result
  printf("[");
  for (int i = 0; i < vecSize; i++) {
    if (i == 0)
      printf("%f", vec_x[i]);
    else
      printf(",%f", vec_x[i]);
  }
  printf("]\n");


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
