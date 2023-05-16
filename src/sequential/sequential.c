/**
 * Sequential solution for solving large systems of linear equations
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

#include "../mtx_sparse.h"

void printHelp(const char *progName) {
  fprintf(stderr, "Usage: %s [-n iterations] [-e margin] FILE\n", progName);
}

void vecSumCoef(double *vecOut, double *vecInA, double *vecInB, bool subtract, int n, double coef) {
  for (int i = 0; i < n; i++)
    vecOut[i] = (subtract) ? (vecInA[i] - coef * vecInB[i]) : (vecInA[i] + coef * vecInB[i]);
}

void vecLinProd(double *vecOut, double *vecIn, int n, double coef) {
  for (int i = 0; i < n; i++)
    vecOut[i] = coef * vecIn[i];
}

double vecDotProduct(double *vecInA, double *vecInB, int n) {
  double prod = 0.f;
  for (int i = 0; i < n; i++)
    prod += vecInA[i] * vecInB[i];
  return prod;
}

int mtxVecProduct_JDS(double *vecOut, struct mtx_JDS *mtx, double *vecIn, int n) {
  // Init value of output vector
  memset(vecOut, 0, n * sizeof(double)); // Set output vector to zero
  
  int curr_els = mtx->max_el_in_row;  
  int rows_computed = 0;
  for (int i = 0; i < mtx->max_el_in_row; i++) {
    int jag_start = mtx->jagged_ptr[i];
    int jag_end = mtx->jagged_ptr[i + 1];
    int curr_els_in_jag = jag_end - jag_start;
    int curr_els_in_row = curr_els_in_jag / curr_els;
    for (int data_ix = jag_start; data_ix < jag_end; data_ix += curr_els_in_row) {
      for (int row_ix = 0; row_ix < curr_els_in_row; row_ix++) {
        double d = mtx->data[data_ix + row_ix];
        int d_col = mtx->col[data_ix + row_ix];
        int d_row = mtx->row_permute[rows_computed + row_ix];
        vecOut[d_row] += d * vecIn[d_col]; 
      }
    }
    rows_computed += curr_els_in_row;
    curr_els--;
  }
  //vecPrint(vecOut, mtx->num_rows);
  return 0;
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
  
  if (mtx_COO_create_from_file(&mtxCOO_t, file2, true) != 0)
    exit(EXIT_FAILURE);

  mtx_CSR_create_from_mtx_COO(&mtxCSR, &mtxCOO);
  mtx_CSR_create_from_mtx_COO(&mtxCSR_t, &mtxCOO_t);
  mtx_ELL_create_from_mtx_CSR(&mtxELL, &mtxCSR);
  mtx_ELL_create_from_mtx_CSR(&mtxELL_t, &mtxCSR_t);
  mtx_JDS_create_from_mtx_CSR(&mtxJDS, &mtxCSR);
  mtx_JDS_create_from_mtx_CSR(&mtxJDS_t, &mtxCSR_t);
  int vecSize = mtxCOO.num_cols;

  // Temporary
  double *vec_tmp = (double*)malloc(vecSize * sizeof(double));

  // Setup initial values

  // s = x_0 (for calculation b = As)
  double *vec_x = (double*)malloc(vecSize * sizeof(double));
  for (int i = 0; i < vecSize; i++)
    vec_x[i] = 1; // Initial s
  //vec_x[0] = 2; vec_x[1] = 1;
  printf("Correct solution: ");
  vecPrint(vec_x, vecSize);
  
  // b = Ax_0 (generate constant)
  double *vec_b = (double*)malloc(vecSize * sizeof(double));
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
  mtxVecProduct_JDS(vec_tmp, &mtxJDS, vec_x, vecSize); // Ax_0
  vecSumCoef(vec_r, vec_b, vec_tmp, true, vecSize, 1); // r = b - Ax_0
  
  // r_dash_0 = r_0
  double *vec_r_dash = (double*)malloc(vecSize * sizeof(double));
  memcpy(vec_r_dash, vec_r, vecSize * sizeof(double)); 

  // p_0 = r_0
  double *vec_p = (double*)malloc(vecSize * sizeof(double));
  memcpy(vec_p, vec_r, vecSize * sizeof(double)); 

  // p_dash_0 = r_dash_0
  double *vec_p_dash = (double*)malloc(vecSize * sizeof(double));
  memcpy(vec_p_dash, vec_r_dash, vecSize * sizeof(double));

  // b_t * b
  double bb = vecDotProduct(vec_b, vec_b, vecSize);

  // r_t_0 * r_0
  double rtr = vecDotProduct(vec_r, vec_r, vecSize);

  // calculate margin^2
  margin = powf(margin, 2) * bb;
  
  clock_t time_start = clock();

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
    rtr = vecDotProduct(vec_r, vec_r, vecSize) / bb;

    // p_k+1 = r_k+1 + beta * p_k
    vecSumCoef(vec_p, vec_r, vec_p, false, vecSize, beta);
    
    // p_dash_k+1 = r_dash_k+1 + beta * p_dash_k
    vecSumCoef(vec_p_dash, vec_r_dash, vec_p_dash, false, vecSize, beta);
    //printf("TEST: %lf %.15lf\n", rtr, margin);
    //vecPrint(vec_r, vecSize);
  }

  clock_t time_end = clock();

  // Print result
  printf("Iterations: %o/%o\nResult: ", k, iterations);
  vecPrint(vec_x, vecSize);
  printf("Elapsed: %lf s\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);
  
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
