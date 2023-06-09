/**
 * OpenMPI solution for solving large systems of linear equations
*/

#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../mtx_sparse.h"

#define PROCESS_ROOT 0

void printHelp(const char *progName) {
  fprintf(stderr, "Usage: %s [-n iterations] [-e margin] FILE\n", progName);
}

void vecSumCoef(double *vecOut, double *vecInA, double *vecInB, bool subtract, int n, double coef) {
  #pragma omp parallel for schedule(guided, 1)
  for (int i = 0; i < n; i++)
    vecOut[i] = (subtract) ? (vecInA[i] - coef * vecInB[i]) : (vecInA[i] + coef * vecInB[i]);
}

void vecLinProd(double *vecOut, double *vecIn, int n, double coef) {
  #pragma omp parallel for schedule(guided, 1)
  for (int i = 0; i < n; i++)
    vecOut[i] = coef * vecIn[i];
}

double vecDotProduct(double *vecInA, double *vecInB, int n) {
  double prod = 0;
  #pragma omp parallel for reduction(+:prod) schedule(guided, 1)
  for (int i = 0; i < n; i++)
    prod += vecInA[i] * vecInB[i];
  return prod;
}

void mtxVecProduct_JDS(double *vecOut, struct mtx_JDS *mtx, double *vecIn, int n) {
  // Init value of output vector
  memset(vecOut, 0, n * sizeof(double)); // Set output vector to zero

  // Multiply each non zero
  // dynamic schedule because of unequal load per row (rows have different sizes)
  #pragma omp parallel for schedule(dynamic, 1)
  for (int r = 0; r < mtx->max_el_in_row; r++) {
    for (int i = mtx->jagged_ptr[r]; i < ((r + 1 < mtx->max_el_in_row) ? mtx->jagged_ptr[r + 1] : mtx->num_nonzeros); i++) {
      if (mtx->row_permute[r] < n)
        vecOut[mtx->row_permute[r]] += mtx->data[i] * vecIn[mtx->col[i]];
    }
  }
}

int main(int argc, char *argv[]) {
  uint32_t iterations = 1000;
  double margin = 0.005;
  bool showIntermediateResult = false;

  // TODO: Cleanup
  int rank; // process rank 
	int num_p; // total number of processes 
	MPI_Status status; // message status
	MPI_Request request; // MPI request
	int flag = 0; // request status flag
	char node_name[MPI_MAX_PROCESSOR_NAME]; //node name
	int name_len; //true length of node name
    
	MPI_Init(&argc, &argv); // initialize MPI 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get process rank 
	MPI_Comm_size(MPI_COMM_WORLD, &num_p); // get number of processes
	MPI_Get_processor_name(node_name, &name_len); // get node name
	
	printf("Process %d/%d running on %s\n", rank, num_p, node_name);

  // Main process
  int vecSize = 0;
  struct mtx_JDS mtxJDS, mtxJDS_t;
  if (rank == 0) {
    // Read arguments
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

    // Create matrix structures
    struct mtx_COO mtxCOO, mtxCOO_t;
    struct mtx_CSR mtxCSR, mtxCSR_t;
    struct mtx_ELL mtxELL, mtxELL_t;

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

    vecSize = mtxCOO.num_cols;

    // Clear memory
    mtx_COO_free(&mtxCOO);
    mtx_CSR_free(&mtxCSR);
    mtx_CSR_free(&mtxCSR_t);
    mtx_ELL_free(&mtxELL);
    mtx_ELL_free(&mtxELL_t);

    // TODO: Move generation of vector b here and broadcast it to all processes

    // TODO: Load calculate load share
  }

  // Send settings to all processes
  MPI_Bcast(&iterations, 1, MPI_UINT32_T, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&margin, 1, MPI_DOUBLE, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&vecSize, 1, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);

  // TODO: Don't forget to init mtxJDS so other processes can receive data
  // Notes (ideas, not fatch):
  //  to each process copy:
  //  - selected part of value
  //  - selected par of column (same position & size)
  //  - jagged_ptr values of column count that will be processed by this process (scatter count array between processes)
  //  - row_permute needs to be accounted on root process when puting things back together
  // - margin needs to be devided by number of processes ?

  // TODO: Scatterv (matrix rows)

  // Init random
  srand(time(NULL));
  
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
  
  double time_start;
  if (rank == 0) {
    time_start = MPI_Wtime();
  }

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

  // TODO: gatherv (result)

  if (rank == 0) {
    double time_end = MPI_Wtime();

    // Print result
    printf("Iterations: %o/%o\nResult: ", k, iterations);
    vecPrint(vec_x, vecSize);
    printf("Elapsed: %.5lf s\n", (double)(time_end - time_start));
  }
  
  MPI_Finalize();

  // Clear memory
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
