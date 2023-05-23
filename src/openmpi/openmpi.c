/**
 * OpenMPI solution for solving large systems of linear equations
*/
#include <mpi.h>
#include <getopt.h>
#include <math.h>
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
  
  int curr_els = mtx->max_el_in_row;  
  int rows_computed = 0;
  for (int i = 0; i < mtx->max_el_in_row; i++) {
    int jag_start = mtx->jagged_ptr[i];
    int jag_end = mtx->jagged_ptr[i + 1];
    if (i == mtx->max_el_in_row - 1) { jag_end = mtx->num_elements; }
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
}

void init_proc_mtx(struct mtx_JDS *mtx_new, int rows, int max_els, 
                  int num_cols, int num_nonzeros, int num_elements) {
  //printf("INIT: %d %d %d %d %d\n", rows, max_els, num_cols, num_nonzeros, num_elements);
  mtx_new->num_rows = rows;
  mtx_new->max_el_in_row = max_els;
  mtx_new->num_cols = num_cols;
  mtx_new->num_nonzeros = num_nonzeros;
  mtx_new->num_elements = num_elements;
  mtx_new->data = (double *)calloc(num_elements, sizeof(double));
  mtx_new->col = (int *) calloc(num_elements, sizeof(int));
  mtx_new->row_permute = (int *) calloc(rows, sizeof(int));
  mtx_new->jagged_ptr = (int *) calloc(max_els, sizeof(int));
}


void get_matrix_from_file(struct mtx_JDS *proc_mtxJDS, bool transposed, char *argv[]){
    struct mtx_COO mtxCOO;
    struct mtx_CSR mtxCSR;
    struct mtx_ELL mtxELL;

      // Open file with matrix A
      FILE *file;
      if ((file = fopen(argv[optind], "r")) == NULL) {
        fprintf(stderr, "Failed to open: %s \n", argv[optind]);
        exit(EXIT_FAILURE);
      }
      if (mtx_COO_create_from_file(&mtxCOO, file, transposed) != 0)
        exit(EXIT_FAILURE);

    mtx_CSR_create_from_mtx_COO(&mtxCSR, &mtxCOO);
    mtx_ELL_create_from_mtx_CSR(&mtxELL, &mtxCSR);
    mtx_JDS_create_from_mtx_CSR(proc_mtxJDS, &mtxCSR);
    
    // Clear memory
    mtx_COO_free(&mtxCOO);
    mtx_CSR_free(&mtxCSR);
    mtx_ELL_free(&mtxELL);
}

void get_matrix(struct mtx_JDS *proc_mtxJDS, bool transposed, int rank, int num_p, char *argv[]) {
  int proc_jag_start, proc_jag_end, proc_actual_rows;
  int proc_num_rows, proc_num_cols, proc_num_els, proc_max_el_in_row, proc_num_nonzeros;
  int *jags_start = malloc(sizeof(int)* num_p);
  int *jags_end = malloc(sizeof(int)* num_p);
  int *actual_rows = malloc(sizeof(int)* num_p);

  if(rank == 0) {
    // Create matrix structures
    get_matrix_from_file(proc_mtxJDS, transposed, argv);
    proc_num_rows = proc_mtxJDS->num_rows;
    proc_num_cols = proc_mtxJDS->num_cols;
    proc_num_els = proc_mtxJDS->num_elements;
    proc_max_el_in_row = proc_mtxJDS->max_el_in_row;
    proc_num_nonzeros = proc_mtxJDS->num_nonzeros;

    //Send matrix to other processes
    memset(jags_start, 0,  num_p * sizeof(int));
    memset(jags_end, 0,  num_p * sizeof(int));
    //jags_start[0] = 0;
    int curr_proc = 0;
    int curr_rows = 0;
    int rows_per_processor = proc_mtxJDS->num_rows / num_p;
    int rows_per_proc_remainder = proc_mtxJDS->num_rows % num_p;
    //printf("%d %d %d %d %d %d\n\n", rows_per_processor, rows_per_proc_remainder, proc_mtxJDS->jagged_ptr[proc_mtxJDS->max_el_in_row - 1], num_p, proc_mtxJDS->num_rows, proc_mtxJDS->max_el_in_row);
    for (int j = 0; j < proc_mtxJDS->max_el_in_row; j++) {
      int els_in_row = proc_mtxJDS->max_el_in_row - j;
      bool trigger = false;
      //printf("%d %d %d %d\n", els_in_row, j, proc_mtxJDS->jagged_ptr[j], proc_mtxJDS->jagged_ptr[j + 1]);
      for (int row = proc_mtxJDS->jagged_ptr[j]; row < ((j + 1 < proc_mtxJDS->max_el_in_row) ? proc_mtxJDS->jagged_ptr[j + 1] : proc_mtxJDS->num_nonzeros); row += els_in_row) {        
        curr_rows++;
        if (curr_rows == rows_per_processor && rows_per_proc_remainder > 0) { 
          rows_per_proc_remainder--;
          curr_rows++;
          row += els_in_row;
        }
        //printf("%d %d %d\n", row, curr_rows, curr_proc);
        if (curr_rows >= rows_per_processor) {
          jags_end[curr_proc] = row; 
          actual_rows[curr_proc] = curr_rows;
          curr_proc++;
          if (curr_proc >= num_p) { trigger = true; break; } 
          jags_start[curr_proc] = row; 
          curr_rows = 0;
        }
      }
      if (trigger) { break; }
    }
    jags_end[curr_proc - 1] = proc_mtxJDS->num_elements;
    //printf("\n\nERROR\n\n");
    //printf("%d %d %d %d,\n", proc_mtxJDS->jagged_ptr[proc_mtxJDS->max_el_in_row - 1], curr_proc, proc_mtxJDS->max_el_in_row, proc_mtxJDS->num_elements);
    // printf("MATRIX:\n");
    // vecPrint(proc_mtxJDS->data, proc_mtxJDS->num_nonzeros);
    // vecPrintInt(proc_mtxJDS->col, proc_mtxJDS->num_nonzeros);
    // vecPrintInt(proc_mtxJDS->row_permute, proc_mtxJDS->num_rows);
    // vecPrintInt(proc_mtxJDS->jagged_ptr, proc_mtxJDS->max_el_in_row);
    // printf("DIVISION: TRansposed: %d\n", transposed);
    // vecPrintInt(jags_start, num_p);
    // vecPrintInt(jags_end, num_p);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&proc_num_rows, 1, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&proc_num_cols, 1, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&proc_num_els, 1, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&proc_max_el_in_row, 1, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&proc_num_nonzeros, 1, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  if (rank != 0) {
    init_proc_mtx(proc_mtxJDS, proc_num_rows, proc_max_el_in_row, 
                  proc_num_cols, proc_num_nonzeros, proc_num_els);
  }
  //printf("\n\nGOT OUT OF INIT\n\n");
  // MPI_Scatterv(mtxJDS->row_permute, sendcounts_rows, displs, MPI_INT, &proc_mtxJDS->row_permute, proc_mtxJDS->num_rows, MPI_INT, 0, MPI_COMM_WORLD);
  // MPI_Scatter(sendcounts_rows, 1, MPI_INT, &proc_mtxJDS->num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatter(jags_start, 1, MPI_INT, &proc_jag_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(jags_end, 1, MPI_INT, &proc_jag_end, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(actual_rows, 1, MPI_INT, &proc_actual_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  //my_jags = (int*)malloc(proc_max_el_in_row * sizeof(int));
  //int my_jag = 0;
  //for (int i = 0; i < num_p; i++) { displs[i] = -1;}
  //displs[0] = 0;
  //printf("MAX EL IN ROW:_ %d %d %d\n", proc_max_el_in_row, proc_mtxJDS->jagged_ptr[0], proc_mtxJDS->jagged_ptr[1]);
  //MPI_Bcast(sendcounts_jags, proc_max_el_in_row, MPI_INT, 0, MPI_COMM_WORLD);
  //MPI_Scatterv(proc_mtxJDS->jagged_ptr, sendcounts_jags, displs, MPI_INT, &my_jags, proc_max_el_in_row, MPI_INT, 0, MPI_COMM_WORLD);
  //printf("PRINT JAGS :");
  //vecPrintInt(my_jags, proc_max_el_in_row);

  MPI_Bcast(proc_mtxJDS->jagged_ptr, proc_max_el_in_row, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(proc_mtxJDS->row_permute, proc_num_rows, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(proc_mtxJDS->col, proc_num_els, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(proc_mtxJDS->data, proc_num_els, MPI_DOUBLE, PROCESS_ROOT, MPI_COMM_WORLD);
  
  for (int i = 0; i < proc_max_el_in_row; i++) {
    if (proc_mtxJDS->jagged_ptr[i] < proc_jag_start) { proc_mtxJDS->jagged_ptr[i] = proc_jag_start; }
    else if (proc_mtxJDS->jagged_ptr[i] > proc_jag_end) { proc_mtxJDS->jagged_ptr[i] = proc_jag_end; }
  }
  // int num_els = proc_jag_end - proc_jag_start;
  // int num_cols = proc_mtxJDS->num_cols;
  // int num_nonzeros = num_els;
  // int num_rows = proc_actual_rows;
  // struct mtx_JDS temp_mtxJDS; 
  // init_proc_mtx(&temp_mtxJDS, num_rows, max_el_in_row, num_cols, num_nonzeros, num_els);
  // printf("TEST: %d:%d %d\n\n", proc_jag_start, proc_jag_end, max_el_in_row);
  // vecPrintInt(proc_mtxJDS->jagged_ptr, proc_max_el_in_row);
  // vecPrintInt(proc_mtxJDS->col, proc_num_els);
  // vecPrint(proc_mtxJDS->data, proc_num_els);
  // vecPrintInt(proc_mtxJDS->row_permute, proc_num_rows);
  // int data_ix = 0;
  // temp_mtxJDS.jagged_ptr[0] = 0;
  // int jag_ix = 0;
  // for (int r = 0; r < proc_mtxJDS->max_el_in_row; r++) {
  //   bool trigger = false;
  //   for (int i = proc_mtxJDS->jagged_ptr[r]; i < ((r + 1 < proc_mtxJDS->max_el_in_row) ? proc_mtxJDS->jagged_ptr[r + 1] : proc_mtxJDS->num_nonzeros); i++) {
  //     if (proc_mtxJDS->row_permute[r] < proc_mtxJDS->num_cols){
  //       int row = proc_mtxJDS->row_permute[r];
  //       temp_mtxJDS.row_permute[r] = row;
  //       temp_mtxJDS.jagged_ptr[jag_ix] = proc_mtxJDS->jagged_ptr[r] - proc_jag_start;
  //       int el = proc_mtxJDS->data[i];
  //       int col = proc_mtxJDS->col[i];
  //       temp_mtxJDS.data[data_ix] = el;
  //       temp_mtxJDS.col[data_ix] = col;
  //       data_ix++;
  //       trigger = true;
  //     }
  //   }
  //   if (trigger) { jag_ix++; }
  // }
  // while (jag_ix < max_el_in_row && temp_mtxJDS.jagged_ptr[jag_ix] == 0) { temp_mtxJDS.jagged_ptr[jag_ix] = proc_jag_end - proc_jag_start; jag_ix++; }

  // vecPrintInt(temp_mtxJDS.jagged_ptr, max_el_in_row);
  // vecPrintInt(temp_mtxJDS.col, num_els);
  // vecPrint(temp_mtxJDS.data, num_els);
  // vecPrintInt(temp_mtxJDS.row_permute, num_rows);
}

int main(int argc, char *argv[]) {
  uint32_t iterations = 1000;
  double margin = 1e-8;
  bool showIntermediateResult = true;
  int rank; // process rank 
	int num_p; // total number of processes 
  double *vec_b, *vec_x;
  
	MPI_Init(&argc, &argv); // initialize MPI 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get process rank 
	MPI_Comm_size(MPI_COMM_WORLD, &num_p); // get number of processes
	
	printf("Process %d/%d\n", rank, num_p);

  // Main processD
  int vecSize = 0;
  struct mtx_JDS proc_mtxJDS, proc_mtxJDS_t, mtxJDS;
  
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
    get_matrix_from_file(&mtxJDS, false, argv);
  }
  get_matrix(&proc_mtxJDS, false, rank, num_p, argv);
  get_matrix(&proc_mtxJDS_t, true, rank, num_p, argv);
  //printf("AFTER\n:");
  // vecPrintInt(proc_mtxJDS.jagged_ptr, proc_mtxJDS.max_el_in_row);
  // vecPrintInt(proc_mtxJDS.col, proc_mtxJDS.num_cols);
  // vecPrint(proc_mtxJDS.data, proc_mtxJDS.num_elements);
  // vecPrintInt(proc_mtxJDS.row_permute, proc_mtxJDS.num_rows);
  // vecPrintInt(proc_mtxJDS_t.jagged_ptr, proc_mtxJDS_t.max_el_in_row);
  // vecPrintInt(proc_mtxJDS_t.col, proc_mtxJDS_t.num_cols);
  // vecPrint(proc_mtxJDS_t.data, proc_mtxJDS_t.num_elements);
  // vecPrintInt(proc_mtxJDS_t.row_permute, proc_mtxJDS_t.num_rows);
  vecSize = proc_mtxJDS.num_cols;
  vec_b = (double*)malloc(vecSize * sizeof(double));
  vec_x = (double*)malloc(vecSize * sizeof(double));
  if (rank == 0) {
    // Setup initial values
    // s = x_0 (for calculation b = As)
    for (int i = 0; i < vecSize; i++)
      vec_x[i] = 1; // Initial s
    printf("Correct solution: ");
    vecPrint(vec_x, vecSize);

    // vecPrintInt(mtxJDS.jagged_ptr, mtxJDS.max_el_in_row);
    // vecPrintInt(mtxJDS.col, mtxJDS.num_cols);
    // vecPrint(mtxJDS.data, mtxJDS.num_elements);
    // vecPrintInt(mtxJDS.row_permute, mtxJDS.num_rows);

    // b = Ax_0 (generate constant)
    mtxVecProduct_JDS(vec_b, &mtxJDS, vec_x, vecSize);
    printf("B vector: ");
    vecPrint(vec_b, vecSize);

    // Init random
    srand(time(NULL));

    // x_0 = random
    for (int i = 0; i < vecSize; i++)
      vec_x[i] = 0;//rand() % 10 + 1; // Initial x_0 value (generate randomly [0, 9])
    printf("x_0: ");
    vecPrint(vec_x, vecSize);
  }

  // Send settings to all processes
  MPI_Bcast(&iterations, 1, MPI_UINT32_T, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&margin, 1, MPI_DOUBLE, PROCESS_ROOT, MPI_COMM_WORLD);
  // MPI_Bcast(&vecSize, 1, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(vec_b, vecSize, MPI_DOUBLE, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(vec_x, vecSize, MPI_DOUBLE, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  // vecPrint(vec_x, vecSize);
  // vecPrint(vec_b, vecSize);
  // TODO: Don't forget to init mtxJDS so other processes can receive data
  // Notes (ideas, not fatch):
  //  to each process copy:
  //  - selected part of value
  //  - selected par of column (same position & size)
  //  - jagged_ptr values of column count that will be processed by this process (scatter count array between processes)
  //  - row_permute needs to be accounted on root process when puting things back together
  // - margin needs to be devided by number of processes?

  // TODO: Scatterv (matrix rows)
  //printf("PROCESS: %d, %d %d\n ", rank, proc_mtxJDS.num_rows, vecSize);
  //vecPrintInt(proc_mtxJDS.jagged_ptr, proc_mtxJDS.max_el_in_row);
  //vecPrintInt(proc_mtxJDS.row_permute, proc_mtxJDS.num_rows);
  //vecPrint(proc_mtxJDS.data, proc_mtxJDS.num_elements);
  printf("\n\n");
  // Temporary

  double *vec_tmp = (double*)malloc(vecSize * sizeof(double));
  
  // // r_0 = b - Ax_0
  double *vec_r = (double*)malloc(vecSize * sizeof(double));
  mtxVecProduct_JDS(vec_tmp, &proc_mtxJDS, vec_x, vecSize); // Ax_0
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
  double local_margin = margin / num_p;
  double time_start;
  if (rank == 0) {
    time_start = MPI_Wtime();
  }

  // Main loop
  double alpha, beta, rr, rrn;
  uint32_t k;
  for (k = 0; k < iterations && rtr > local_margin; k++) {
    // r_dash_t_k * r_k
    rr = vecDotProduct(vec_r_dash, vec_r, vecSize);

    // Ap_k
    mtxVecProduct_JDS(vec_tmp, &proc_mtxJDS, vec_p, vecSize);

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
    mtxVecProduct_JDS(vec_tmp, &proc_mtxJDS_t, vec_p_dash, vecSize);

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

  for (int p = 1; p < num_p; p++) {
    // ptr to send data, send data size, send data type, receiver, message tag,
		// ptr to received data, received data size, recevied data type, sender, message tag,
		// communicator, status
    double *vec_temp_sum = (double*)malloc(vecSize * sizeof(double));
		MPI_Sendrecv(vec_x, vecSize, MPI_DOUBLE, 0, 1,
					 vec_temp_sum, vecSize, MPI_DOUBLE, p, 1, 
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    if (rank == 0) { vecSumCoef(vec_x, vec_x, vec_temp_sum, false, vecSize, 1); }
  }
  // TODO: gatherv (result)
  printf("PROCESS: %d\n", rank);
  vecPrint(vec_x, vecSize);
  // printf("MATRIX AT END OF PROCESSING\n:");
  // vecPrintInt(proc_mtxJDS.jagged_ptr, proc_mtxJDS.max_el_in_row);
  // vecPrintInt(proc_mtxJDS.col, proc_mtxJDS.num_cols);
  // vecPrint(proc_mtxJDS.data, proc_mtxJDS.num_elements);
  // vecPrintInt(proc_mtxJDS.row_permute, proc_mtxJDS.num_rows);
  // vecPrintInt(proc_mtxJDS_t.jagged_ptr, proc_mtxJDS_t.max_el_in_row);
  // vecPrintInt(proc_mtxJDS_t.col, proc_mtxJDS_t.num_cols);
  // vecPrint(proc_mtxJDS_t.data, proc_mtxJDS_t.num_elements);
  // vecPrintInt(proc_mtxJDS_t.row_permute, proc_mtxJDS_t.num_rows);
  if (rank == 0) {
    double time_end = MPI_Wtime();
    mtx_JDS_free(&mtxJDS);
    // Print result
    printf("Iterations: %o/%o\nResult: ", k, iterations);
    vecPrint(vec_x, vecSize);
    printf("Elapsed: %.5lf s\n", (double)(time_end - time_start));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  mtx_JDS_free(&proc_mtxJDS_t);
  mtx_JDS_free(&proc_mtxJDS);

  MPI_Finalize();

  // Clear memory 
  free(vec_tmp);
  free(vec_b);
  free(vec_x);
  free(vec_r);
  free(vec_r_dash);
  free(vec_p);
  free(vec_p_dash);

  return 0;
}
