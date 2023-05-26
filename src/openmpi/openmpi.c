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
  //vecPrintInt(mtx->jagged_ptr, mtx->max_el_in_row + 1);
  //int curr_els = mtx->max_el_in_row;  
  int rows_computed = 0;
  for (int i = 0; i < mtx->max_el_in_row; i++) {
    int jag_start = mtx->jagged_ptr[i];
    //printf("MULTIPLY TEST %d %d\n", i, mtx->max_el_in_row); fflush(stdout);
    int jag_end = mtx->jagged_ptr[i + 1];
    int curr_els = mtx->max_el_in_row - i;
    if (i == mtx->max_el_in_row - 1 && jag_end < jag_start) { jag_end = mtx->num_elements; }
    int curr_els_in_jag = jag_end - jag_start;
    int curr_rows_in_jag = curr_els_in_jag / curr_els;
    // printf("JAGS AND STUFF : %d %d %d %d %d %d %d \n", i, jag_start, jag_end, curr_els_in_jag, curr_rows_in_jag, curr_els, rows_computed);
    // fflush(stdout);
    for (int data_ix = jag_start; data_ix < jag_end; data_ix += curr_rows_in_jag) {
      //printf("ROWS AND STUFF: %d\n", data_ix); fflush(stdout);
      for (int row_ix = 0; row_ix < curr_rows_in_jag; row_ix++) {
        // printf("JAGS AND STUFF : %d %d %d %d %d %d %d \n", i, jag_start, jag_end, curr_els_in_jag, curr_rows_in_jag, curr_els, rows_computed);
        
        double d = mtx->data[data_ix + row_ix];
        int d_col = mtx->col[data_ix + row_ix];
        int d_row = mtx->row_permute[rows_computed + row_ix];
        
        // fflush(stdout);
        vecOut[d_row] += d * vecIn[d_col]; 
      }
    }
    rows_computed += curr_rows_in_jag;
    curr_els--;
    //printf("%d\n", curr_els);
    fflush(stdout);
  }
  fflush(stdout);
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
    //struct mtx_ELL mtxELL;

      // Open file with matrix A
      FILE *file;
      if ((file = fopen(argv[optind], "r")) == NULL) {
        fprintf(stderr, "Failed to open: %s \n", argv[optind]);
        exit(EXIT_FAILURE);
      }
      if (mtx_COO_create_from_file(&mtxCOO, file, transposed) != 0){
       printf("FAILED TO GENERATE matrix\n");
       fflush(stdout);
       exit(EXIT_FAILURE);
      }
        

    mtx_CSR_create_from_mtx_COO(&mtxCSR, &mtxCOO);
    mtx_JDS_create_from_mtx_CSR(proc_mtxJDS, &mtxCSR);
    
    // Clear memory
    mtx_COO_free(&mtxCOO);
    mtx_CSR_free(&mtxCSR);
}

struct mtx_JDS get_matrix(struct mtx_JDS *proc_mtxJDS, bool transposed, int rank, int num_p, char *argv[], int* all_rows) {
  int proc_jag_start, proc_jag_end;
  int proc_num_rows, proc_num_cols, proc_num_els, proc_max_el_in_row, proc_num_nonzeros;
  int *jags_start = malloc(sizeof(int)* num_p);
  int *jags_end = malloc(sizeof(int)* num_p);
  // printf("PROCES IN GET MATRIX: %d %d\n", rank, transposed);
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
    int jags_per_processor = proc_mtxJDS->max_el_in_row / num_p;
    int jags_per_proc_remainder = proc_mtxJDS->max_el_in_row % num_p;
    int jag_ix = 0;
    for (int p = 0; p < num_p; p++) {
      //printf("%d, %d %d %d %d\n", p, jag_ix, jags_per_processor, jags_per_proc_remainder, proc_mtxJDS->max_el_in_row);
      jags_start[p] = (jag_ix < proc_mtxJDS->max_el_in_row) ? proc_mtxJDS->jagged_ptr[jag_ix] : proc_mtxJDS->num_elements; 
      if (jags_per_proc_remainder > 0) { jag_ix++; jags_per_proc_remainder--; }
      jag_ix += jags_per_processor;
      if (jag_ix >= proc_mtxJDS->max_el_in_row) { jags_end[p] = proc_mtxJDS->num_elements; }
      else { jags_end[p] = proc_mtxJDS->jagged_ptr[jag_ix]; }
    }
    // vecPrintInt(jags_start, num_p);
    // vecPrintInt(jags_end, num_p);
  }
  // printf("TEST: %d %d\n", rank, transposed);
  // fflush(stdout);
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
  *all_rows = proc_num_rows;
  MPI_Scatter(jags_start, 1, MPI_INT, &proc_jag_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(jags_end, 1, MPI_INT, &proc_jag_end, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(proc_mtxJDS->jagged_ptr, proc_max_el_in_row, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(proc_mtxJDS->row_permute, proc_num_rows, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(proc_mtxJDS->col, proc_num_els, MPI_INT, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Bcast(proc_mtxJDS->data, proc_num_els, MPI_DOUBLE, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  struct mtx_JDS temp_mtxJDS;
  temp_mtxJDS.num_elements = proc_jag_end - proc_jag_start;
  temp_mtxJDS.data = (double *)calloc(proc_mtxJDS->num_elements, sizeof(double));
  temp_mtxJDS.col = (int *) calloc(proc_mtxJDS->num_elements, sizeof(int));
  temp_mtxJDS.row_permute = (int *) calloc(proc_mtxJDS->num_rows, sizeof(int));
  temp_mtxJDS.jagged_ptr = (int *) calloc(proc_mtxJDS->max_el_in_row, sizeof(int));
  bool first = true;
  int jag_ix = 0;
  int data_ix = 0;
  int row_ix = 0;
  int computed_rows = 0;
  
  // fflush(stdout);
  // vecPrintInt(proc_mtxJDS->jagged_ptr, proc_max_el_in_row + 1);
  // vecPrintInt(proc_mtxJDS->col, proc_mtxJDS->num_elements);
  // vecPrint(proc_mtxJDS->data, proc_mtxJDS->num_elements);
  for (int i = 0; i < proc_max_el_in_row; i++) {    
    int curr_els_in_row = proc_max_el_in_row - i;
    int temp_jag_ix = proc_mtxJDS->jagged_ptr[i];
    int end_temp_jag_ix = proc_mtxJDS->jagged_ptr[i + 1];
    if (i == proc_max_el_in_row - 1) { end_temp_jag_ix = proc_mtxJDS->num_elements; }
    int jag_len = end_temp_jag_ix - temp_jag_ix;
    int rows_in_jag = jag_len / curr_els_in_row;
    // printf("%d %d %d %d %d %d %d %d %d %d\n", i,
    //     curr_els_in_row, temp_jag_ix, end_temp_jag_ix, first, temp_jag_ix >= proc_jag_start,
    //     temp_jag_ix <= proc_jag_end, proc_jag_start, proc_jag_end, rows_in_jag);
    // fflush(stdout);
    if (temp_jag_ix >= proc_jag_start && end_temp_jag_ix > proc_jag_start
       && end_temp_jag_ix <= proc_jag_end) { 
      if(first && temp_jag_ix == proc_jag_start) { 
        if (proc_jag_start == proc_jag_end) { temp_mtxJDS.max_el_in_row = 0; }
        else { temp_mtxJDS.max_el_in_row = curr_els_in_row; } 
        first = false; 
      }
      temp_mtxJDS.jagged_ptr[jag_ix] = data_ix;
      //printf("TEEST %d %d %d %d\n", jag_ix, data_ix, temp_jag_ix, end_temp_jag_ix);
      jag_ix++;
      if (end_temp_jag_ix <= proc_jag_end) {
        for (int d = temp_jag_ix; d < end_temp_jag_ix; d++) {
          temp_mtxJDS.data[data_ix] = proc_mtxJDS->data[d];
          temp_mtxJDS.col[data_ix] = proc_mtxJDS->col[d];
          //printf("%d %d %f %d\n", data_ix, d, proc_mtxJDS->data[d], proc_mtxJDS->col[d]);
          data_ix++;
        }
        for (int r = 0; r < rows_in_jag; r++) {
          temp_mtxJDS.row_permute[row_ix] = proc_mtxJDS->row_permute[r + computed_rows];
          row_ix++;
        }
      }
      if (jag_ix < temp_mtxJDS.max_el_in_row) { 
        temp_mtxJDS.jagged_ptr[jag_ix] = data_ix + temp_mtxJDS.jagged_ptr[jag_ix - 1];
      }
    }
    computed_rows += rows_in_jag;
  }
  while (jag_ix >= temp_mtxJDS.max_el_in_row) { jag_ix--; }
  if (proc_jag_start == proc_jag_end) { temp_mtxJDS.max_el_in_row = 0; }
  //printf("\n2 %d\n", temp_mtxJDS.max_el_in_row);
  //vecPrintInt(temp_mtxJDS.jagged_ptr, temp_mtxJDS.max_el_in_row + 1);
  // fflush(stdout);
  for (int i = temp_mtxJDS.max_el_in_row; (i >= 0) && (temp_mtxJDS.jagged_ptr[i] == 0); i--) {
    // printf("\njag: %d\n", i);
    temp_mtxJDS.jagged_ptr[i] = temp_mtxJDS.jagged_ptr[jag_ix];
    //fflush(stdout);
  }
  temp_mtxJDS.num_rows = row_ix;
  temp_mtxJDS.num_cols = temp_mtxJDS.max_el_in_row;
  temp_mtxJDS.num_elements = data_ix;
  // printf("PROCESS in matrix: %d %d %d %d\n", rank, transposed, temp_mtxJDS.max_el_in_row, temp_mtxJDS.num_elements);
  // vecPrintInt(temp_mtxJDS.jagged_ptr, temp_mtxJDS.max_el_in_row + 1);
  // vecPrintInt(temp_mtxJDS.col, data_ix);
  // vecPrint(temp_mtxJDS.data, data_ix);
  // vecPrintInt(temp_mtxJDS.row_permute, row_ix);
  // printf("\n %d %d\n", temp_mtxJDS.num_rows,  temp_mtxJDS.num_cols);
  // fflush(stdout);
  return temp_mtxJDS;
}

int main(int argc, char *argv[]) {
  uint32_t iterations = 1000;
  double margin = 1e-8;
  bool showIntermediateResult = false;
  int rank; // process rank 
	int num_p; // total number of processes 
  double *vec_b, *vec_x;
  // printf("Initialization\n");
  // fflush(stdout);
	MPI_Init(&argc, &argv); // initialize MPI 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get process rank 
	MPI_Comm_size(MPI_COMM_WORLD, &num_p); // get number of processes
	
	//printf("Process %d/%d\n", rank, num_p);
  //fflush(stdout);
  // Main process
  int vecSize = 0;
  struct mtx_JDS proc_mtxJDS, proc_mtxJDS_t, mtxJDS;
  double init_time;
  if (rank == 0) {
    init_time = MPI_Wtime();
  }
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
  MPI_Barrier(MPI_COMM_WORLD);
  //printf("Getting matrices, process : %d\n", rank);
  //fflush(stdout);
  int all_rows = 0;
  proc_mtxJDS = get_matrix(&proc_mtxJDS, false, rank, num_p, argv, &all_rows);
  proc_mtxJDS_t = get_matrix(&proc_mtxJDS_t, true, rank, num_p, argv, &all_rows);
  //printf("Got matrices, process : %d\n", rank);
  //fflush(stdout);
  if (rank == 0) {
    double end_init_time = MPI_Wtime();
    printf("Init time : %.5lf s\n", (double)(end_init_time - init_time));
    fflush(stdout);
  }
  // printf("AFTER %d\n", all_rows);
  // printf("PROCESS : %d\n", rank);
  // fflush(stdout);
  // vecPrintInt(proc_mtxJDS.jagged_ptr, proc_mtxJDS.max_el_in_row + 1);
  // vecPrintInt(proc_mtxJDS.col, proc_mtxJDS.num_cols);
  // vecPrint(proc_mtxJDS.data, proc_mtxJDS.num_elements);
  // vecPrintInt(proc_mtxJDS.row_permute, proc_mtxJDS.num_rows);
  // fflush(stdout);
  // vecPrintInt(proc_mtxJDS_t.jagged_ptr, proc_mtxJDS_t.max_el_in_row);
  // vecPrintInt(proc_mtxJDS_t.col, proc_mtxJDS_t.num_cols);
  // vecPrint(proc_mtxJDS_t.data, proc_mtxJDS_t.num_elements);
  // vecPrintInt(proc_mtxJDS_t.row_permute, proc_mtxJDS_t.num_rows);
  MPI_Barrier(MPI_COMM_WORLD);
  vecSize = all_rows;
  // printf("PROCESS %d size %d\n", rank, vecSize);
  // fflush(stdout);
  vec_b = (double*) malloc(vecSize * sizeof(double));
  vec_x = (double*) malloc(vecSize * sizeof(double));
  //if (rank == 0) {
    // Setup initial values
    // s = x_0 (for calculation b = As)
    //printf("PROCESS: %d %d\n ", rank, proc_mtxJDS.num_rows);
    for (int i = 0; i < vecSize; i++)
      vec_x[i] = 1; // Initial s
    
    //printf("Correct solution: \n");
    // vecPrint(vec_x, vecSize);
    //fflush(stdout);
    //vecPrintInt(proc_mtxJDS.jagged_ptr, proc_mtxJDS.max_el_in_row);
    //vecPrintInt(proc_mtxJDS.col, proc_mtxJDS.num_cols);
    //vecPrint(proc_mtxJDS.data, proc_mtxJDS.num_elements);
    //vecPrintInt(proc_mtxJDS.row_permute, proc_mtxJDS.num_rows);
    // printf("BEFORE MULTIPLY\n");
    // fflush(stdout);
    // b = Ax_0 (generate constant)
    mtxVecProduct_JDS(vec_b, &proc_mtxJDS, vec_x, vecSize);
    // printf("AFTER MULTIPLY\n");
    // fflush(stdout);

    // printf("B vector: ");
    // //vecPrint(vec_b, vecSize);
    // fflush(stdout);
    // Init random
    srand(time(NULL));

    // x_0 = random
    for (int i = 0; i < vecSize; i++)
      vec_x[i] = 0;//rand() % 10 + 1; // Initial x_0 value (generate randomly [0, 9])
    //printf("x_0: ");
    //vecPrint(vec_x, vecSize);
  //}
  // Send settings to all processes

  MPI_Bcast(&margin, 1, MPI_DOUBLE, PROCESS_ROOT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // TODO: Scatterv (matrix rows)
  // printf("PROCESS: %d %d \n ", rank, proc_mtxJDS.num_rows);
  // vecPrintInt(proc_mtxJDS.jagged_ptr, proc_mtxJDS.max_el_in_row);
  // vecPrintInt(proc_mtxJDS.row_permute, proc_mtxJDS.num_rows);
  // vecPrint(proc_mtxJDS.data, proc_mtxJDS.num_elements);
  // vecPrint(vec_x, vecSize);
  // vecPrint(vec_b, vecSize);
  printf("Started computing process: %d\n", rank);
  // fflush(stdout);
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
  for (k = 0; proc_mtxJDS.num_rows > 0 && k < iterations && rtr > local_margin; k++) {
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
      fflush(stdout);
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
    //printf("Iteration: %d\n", k);
    //fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    double time_end = MPI_Wtime();
    printf("Compute time: %.5lf s\n", (double)(time_end - time_start));
  }
  // printf("PROCESS a: %d %d %d\n", rank, vecSize, proc_mtxJDS.num_rows);
  // vecPrintInt(proc_mtxJDS.jagged_ptr, proc_mtxJDS.max_el_in_row + 1);
  // //fflush(stdout); 
  // vecPrint(vec_x, vecSize);
  if (rank == 0) {
    for (int p = 1; p < num_p; p++) {
      double *vec_temp_sum = (double*)malloc(vecSize * sizeof(double));
      MPI_Status status;
      MPI_Recv(vec_temp_sum, vecSize, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status );
      vecSumCoef(vec_x, vec_x, vec_temp_sum, false, vecSize, 1); 
      //vecPrint(vec_x, vecSize);
    }
  } else {
    MPI_Send(vec_x, vecSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  //printf("PROCESS d: %d\n", rank);
  fflush(stdout);
  if (rank == 0) {
    double time_end = MPI_Wtime();
    mtx_JDS_free(&mtxJDS);
    double error = 0;
    for (int i = 0; i < vecSize; i++) { error += fabs(1 - fabs(vec_x[i])); }
    // Print result0
    printf("Iterations: %o/%o\nResult: ", k, iterations);
    // vecPrint(vec_x, vecSize);
    printf("Error: %.7f, Elapsed (compute + building result): %.5lf s\n", error, (double)(time_end - time_start));
    fflush(stdout);
  }
  //MPI_Barrier(MPI_COMM_WORLD);
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
  fflush(stdout);
  return 0;
}
