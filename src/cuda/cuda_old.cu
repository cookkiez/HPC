// module load CUDA/10.1.243-GCC-8.3.0
// nvcc -Xcompiler -o SparseMV SparseMV.cu mtx_sparse.c
// srun --reservation=fri --gpus=1 SparseMV data/scircuit.mtx 
// srun --reservation=fri --gpus=1 SparseMV data/pdb1HYS.mtx
// srun --reservation=fri -G1 -n1 sparseMV data/pwtk.mtx

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "../mtx_sparse.h"
#include "helper_cuda.h"

#define THREADS_PER_BLOCK 256
#define REPEAT 1

__global__ void mELLxVec(int *col, float *data, float *vin, float *vout, int rows, int elemsinrow) {		
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if(gid < rows)
	{
		float sum = 0.0f;
		int idx;
		for (int j = 0; j < elemsinrow; j++)
		{
			idx = j * rows + gid;
            sum += data[idx] * vin[col[idx]];
		}
		vout[gid] = sum;
	}
}

__global__ void mJDSVecPar(int *col, float *data, int * jagged_ptr, int *row_permute, int max_el_in_row,
                           int num_rows, int num_cols, int num_elements, float *vin, float *vout){
    // Hint: Each thread calculates its own product, then add it to the vector with atomicAdd
}

void multiplyVecMatJDS(float *r, int max_el_in_row, int *jagged_ptr, float *data, int *col, int *row_permute, 
                      float *multiplyVector, int num_rows, bool transpose) {
    int curr_els = max_el_in_row;  
    int rows_computed = 0;
    for (int i = 0; i < max_el_in_row - 1; i++) {
        int jag_start = jagged_ptr[i];
        int jag_end = jagged_ptr[i + 1];
        int curr_els_in_jag = jag_end - jag_start;
        int curr_els_in_row = curr_els_in_jag / curr_els;
        for (int data_ix = jag_start; data_ix < jag_end; data_ix += curr_els_in_row) {
            for (int row_ix = 0; row_ix < curr_els_in_row; row_ix++) {
                float d = data[data_ix + row_ix];
                int d_col = col[data_ix + row_ix];
                int d_row = row_permute[rows_computed + row_ix];
                //printf("%f, %f, %d, %d\n", d, multiplyVector[d_col],  d_row, d_col);
                if (transpose) { r[d_col] += (d * multiplyVector[d_row]); } 
                else { r[d_row] += (d * multiplyVector[d_col]); }
            }
        }
        rows_computed += curr_els_in_row;
        curr_els--;
    }
    printf("Mat x Vec \n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f %f \n ", r[i], multiplyVector[i]);
    }
    printf("\n");
}

float dotProduct(float *a, float *b, int num_rows) {
    float out = 0.0;
    for (int i = 0; i < num_rows; i++) {
        out += a[i] * b[i];
        printf("%f, %f\n", a[i], b[i]);
    }
    printf("%f\n\n", out);

    return out;
}


void sumVectors (float *out, float *x, float *p, bool plus, int num_rows, float coeff) { 
    printf("SUM\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f %f %f %f %d\n ", out[i], x[i], p[i], coeff, plus);
        out[i] = (plus) ? x[i] + coeff * p[i] : x[i] - coeff * p[i];
        printf("%f\n ", out[i]);
    }
    printf("\n");
}

void computeResidual(float *r, int max_el_in_row, int *jagged_ptr, float *data, int *col, int *row_permute, 
                     float *base, float *multiplyVector, int num_rows, int coeff) {
    float *temp = (float *) calloc(num_rows, sizeof(float));
    multiplyVecMatJDS(temp, max_el_in_row, jagged_ptr, data, col, row_permute, 
                      multiplyVector, num_rows, false);
    sumVectors(r, base, temp, false, num_rows, coeff);
}


void mJDSVecSeq(int *col, float *data, int *jagged_ptr, int *row_permute, int max_el_in_row,
                int num_rows, int num_cols, int num_elements, float *b, float *x, int iters,
                float epsilon) {
    float *r = (float *) calloc(num_rows, sizeof(float));    
    float *r_dash = (float *) calloc(num_rows, sizeof(float));
    float *p = (float *) calloc(num_rows, sizeof(float));    
    float *p_dash = (float *) calloc(num_rows, sizeof(float)); 

    computeResidual(r, max_el_in_row, jagged_ptr, data, col, row_permute, 
        b, x, num_rows, 1.0);
    for (int i = 0; i < num_rows; i++) { r_dash[i] = r[i]; p[i] = r[i]; p_dash[i] = r[i]; }

    int k = 0;
    float rDotProduct = dotProduct(r_dash, r, num_rows);
    printf("%f\n", rDotProduct);
    while (k < iters && fabs(rDotProduct) > epsilon) {
        float *mulMatVec = (float *) calloc(num_rows, sizeof(float));
        float *mulMatVecTrans = (float *) calloc(num_rows, sizeof(float));
        printf("// A * pk\n");
        multiplyVecMatJDS(mulMatVec, max_el_in_row, jagged_ptr, data, col, row_permute, 
                      p, num_rows, false);
        //printf("%f\n", rDotProduct);
        float alpha_k = rDotProduct / dotProduct(p_dash, mulMatVec, num_rows);
        printf("alpha: %f\n", alpha_k);
        printf("// xk + alpha_k * pk\n");
        sumVectors(x, x, p, true, num_rows, alpha_k);
        printf("// rk - alpha_k * A*pk\n");
        sumVectors(r, r, mulMatVec, false, num_rows, alpha_k);
        printf("// A^t * p_dash_k\n");
        multiplyVecMatJDS(mulMatVecTrans, max_el_in_row, jagged_ptr, data, col, row_permute, 
                      p_dash, num_rows, true); 
        printf("// r_dash_k - alpha_k * A^t * p_dash_k\n");
        sumVectors(r_dash, r_dash, mulMatVecTrans, false, num_rows, alpha_k);
        float beta_k = dotProduct(r_dash, r, num_rows) / rDotProduct;
        printf("beta: %f\n", beta_k);
        printf("// rk + beta_k * pk\n");
        sumVectors(p, r, p, true, num_rows, beta_k);
        printf("// r_dash_k + betak_k * p_dash_k\n");
        sumVectors(p_dash, r_dash, p_dash, true, num_rows, beta_k);
        printf("// r^t * r\n");
        rDotProduct = dotProduct(r_dash, r, num_rows);
        printf("Finished iteration: %d\n", k);
        k++;
        
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
  struct mtx_COO h_mtxCOO, h_mtxCOO_t;
  struct mtx_CSR h_mtxCSR, h_mtxCSR_t;
  struct mtx_ELL h_mtxELL, h_mtxELL_t;
  struct mtx_JDS h_mtxJDS, h_mtxJDS_t;

  if (mtx_COO_create_from_file(&h_mtxCOO, file, false) != 0)
    exit(EXIT_FAILURE);
  
  if (mtx_COO_create_from_file(&h_mtxCOO_t, file2, true) != 0)
    exit(EXIT_FAILURE);

  mtx_CSR_create_from_mtx_COO(&h_mtxCSR, &h_mtxCOO);
  mtx_CSR_create_from_mtx_COO(&h_mtxCSR_t, &h_mtxCOO_t);
  mtx_ELL_create_from_mtx_CSR(&h_mtxELL, &h_mtxCSR);
  mtx_ELL_create_from_mtx_CSR(&h_mtxELL_t, &h_mtxCSR_t);
  mtx_JDS_create_from_mtx_CSR(&h_mtxJDS, &h_mtxCSR);
  mtx_JDS_create_from_mtx_CSR(&h_mtxJDS_t, &h_mtxCSR_t);
  int vecSize = mtxCOO.num_cols;

  // Free unnessesary matrices
  mtx_COO_free(&h_mtxCOO);
  mtx_CSR_free(&h_mtxCSR);
  mtx_CSR_free(&h_mtxCSR_t);
  mtx_ELL_free(&h_mtxELL);
  mtx_ELL_free(&h_mtxELL_t);

  // TODO: Generate b and calculate bb, margin (then transfer them into device memory)

  // TODO: Calculate init values for vectors and transfer values to device memory
  // Hint: Maybe look at functions in openmp.c, because they are clear-written

  // allocate vectors
  float *h_vecIn = (float *)malloc(h_mCOO.num_cols * sizeof(float));
  float* h_vecOutJDSSeq = (float*)calloc(h_mCSR.num_rows, sizeof(float));
  for (int i = 0; i < h_mCOO.num_cols; i++) {
      //h_vecOutJDSSeq[i] = 5;
      h_vecIn[i] = 1.0;
  }

  // compute with COO
  float *h_vecOutCOO_cpu = (float *)calloc(h_mCOO.num_rows, sizeof(float));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (repeat = 0; repeat < REPEAT; repeat++)
  {
      for (int i = 0; i < h_mCOO.num_rows; i++)
          h_vecOutCOO_cpu[i] = 0.0;
      for (int i = 0; i < h_mCOO.num_nonzeros; i++)
          h_vecOutCOO_cpu[h_mCOO.row[i]] += h_mCOO.data[i] * h_vecIn[h_mCOO.col[i]];
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float dtimeCOO_cpu = 0;
  cudaEventElapsedTime(&dtimeCOO_cpu, start, stop);
  
  //float *h_vecOutCSR_gpu = (float *)calloc(h_mCSR.num_rows, sizeof(float));
  float *h_vecOutELL_gpu = (float *)calloc(h_mELL.num_rows, sizeof(float));
  float* h_vecOutCSRpar = (float *)calloc(h_mCSR.num_rows, sizeof(float));
  
  int iters = 30;
  float epsilon = 1e-10;
  mJDSVecSeq(h_mJDS.col, h_mJDS.data, h_mJDS.jagged_ptr, h_mJDS.row_permute, h_mJDS.max_el_in_row,
              h_mJDS.num_rows, h_mJDS.num_cols, h_mJDS.num_elements, 
              h_vecOutCOO_cpu, h_vecOutJDSSeq, iters, epsilon);

  // allocate memory on device and transfer data from host 
  // CSR
  //int *d_mCSRrowptr, *d_mCSRcol;
  //float *d_mCSRdata;
  //cudaMalloc((void **)&d_mCSRrowptr, (h_mCSR.num_rows + 1) * sizeof(int));
  //cudaMalloc((void **)&d_mCSRcol, (h_mCSR.num_nonzeros + 1) * sizeof(int));
  //cudaMalloc((void **)&d_mCSRdata, h_mCSR.num_nonzeros * sizeof(float));
  //cudaMemcpy(d_mCSRrowptr, h_mCSR.rowptr, (h_mCSR.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_mCSRcol, h_mCSR.col, h_mCSR.num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_mCSRdata, h_mCSR.data, h_mCSR.num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
  // ELL
  int *d_mELLcol;
  float *d_mELLdata;
  checkCudaErrors(cudaMalloc((void **)&d_mELLcol, h_mELL.num_elements * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_mELLdata, h_mELL.num_elements * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_mELLcol, h_mELL.col, h_mELL.num_elements * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_mELLdata, h_mELL.data, h_mELL.num_elements * sizeof(float), cudaMemcpyHostToDevice));

  // vectors
  float *d_vecIn, *d_vecOut;
  checkCudaErrors(cudaMalloc((void **)&d_vecIn, h_mCOO.num_cols * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_vecOut, h_mCOO.num_rows * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_vecIn, h_vecIn, h_mCSR.num_cols*sizeof(float), cudaMemcpyHostToDevice));

  
// Divide work 
  dim3 blocksize(THREADS_PER_BLOCK);
  
  // CSR
  //dim3 gridsize_CSR((h_mCSR.num_rows - 1) / blocksize.x + 1);
  
  // ELL
  dim3 gridsize_ELL((h_mELL.num_rows - 1) / blocksize.x + 1);

// CSR execute
  // cudaEventRecord(start);
  // for (repeat = 0; repeat < REPEAT; repeat++)
  // {
  //     mCSRxVec<<<gridsize_CSR, blocksize>>>(d_mCSRrowptr, d_mCSRcol, d_mCSRdata, d_vecIn, d_vecOut, h_mCSR.num_rows);
  // }    
// cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float dtimeCSR_gpu = 0;
  // cudaEventElapsedTime(&dtimeCSR_gpu, start, stop);
  // cudaMemcpy(h_vecOutCSR_gpu, d_vecOut, h_mCSR.num_rows*sizeof(float), cudaMemcpyDeviceToHost);

  // CSRPar execute
  cudaEventRecord(start);
  // int num_threads = 64;
  // int num_blocks = 1;
  // for (repeat = 0; repeat < REPEAT; repeat++)
  // {
  //     dim3 blocksize(num_threads);
  //     dim3 gridsize_CSRpar(num_blocks);
  //     mCSRxVecPar<<<gridsize_CSRpar, blocksize>>>(d_mCSRrowptr, d_mCSRcol, d_mCSRdata, d_vecIn, d_vecOut, h_mCSR.num_rows, h_mELL.num_elementsinrow, num_threads);
  //     getLastCudaError("printGPU() execution failed\n");
  // }
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float dtimeCSRPar_gpu = 0;
  // cudaEventElapsedTime(&dtimeCSRPar_gpu, start, stop);
  // cudaMemcpy(h_vecOutCSRpar, d_vecOut, h_mCSR.num_rows * sizeof(float), cudaMemcpyDeviceToHost);


// ELL write, execute, read
  cudaEventRecord(start);
  for (repeat = 0; repeat < REPEAT; repeat++)
  {
      mELLxVec<<<gridsize_ELL, blocksize>>>(d_mELLcol, d_mELLdata, d_vecIn, d_vecOut, h_mELL.num_rows, h_mELL.num_elementsinrow);   
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float dtimeELL_gpu = 0;
  cudaEventElapsedTime(&dtimeELL_gpu, start, stop);
  checkCudaErrors(cudaMemcpy(h_vecOutELL_gpu, d_vecOut, h_mELL.num_rows*sizeof(float), cudaMemcpyDeviceToHost));
  
  // release device memory
  // cudaFree(d_mCSRrowptr);
  // cudaFree(d_mCSRcol);
  // cudaFree(d_mCSRdata);
  checkCudaErrors(cudaFree(d_mELLcol));
  checkCudaErrors(cudaFree(d_mELLdata));
  checkCudaErrors(cudaFree(d_vecIn));
  checkCudaErrors(cudaFree(d_vecOut));

  // output
  printf("Matrix: %s, size: %d x %d, nonzero: %d, max elems in row: %d\n", argv[1],h_mCOO.num_rows, h_mCOO.num_cols, h_mCOO.num_nonzeros, h_mELL.num_elementsinrow);
  int errorsJDS_seq = 0;
  int errorsJDS_par = 0;
  int errorsELL_gpu = 0;
  for(int i = 0; i < h_mCOO.num_rows; i++)
  {
      if (fabs(h_vecIn[i] - h_vecOutJDSSeq[i]) > 1e-4 ){
          errorsJDS_seq++;
          printf("%f, %f\n", h_vecIn[i], h_vecOutJDSSeq[i]);
      }
          
      // if (fabs(h_vecOutCOO_cpu[i] - h_vecOutCSRpar[i]) > 1e-4){
      //     errorsCSRPar_gpu++;
      //     //printf("%d, %.4f, %.4f, %.4f \n", i, h_vecOutCOO_cpu[i], h_vecOutCSRpar[i], fabs(h_vecOutCOO_cpu[i] - h_vecOutCSRpar[i]));
      // }
          
      if (fabs(h_vecIn[i]-h_vecOutELL_gpu[i]) > 1e-4 )
          errorsELL_gpu++;
  }
  printf("Errors: %d(JDS), %d(JDS_par), %d(ELL_gpu)\n", errorsJDS_seq, errorsJDS_par, errorsELL_gpu);
  //printf("Times: %.1f ms(COO_cpu), %.1f ms(CSR_gpu), %.1f ms (CSRPar_gpu), %.1f ms(ELL_gpu)\n\n", dtimeCOO_cpu, dtimeCSR_gpu, dtimeCSRPar_gpu, dtimeELL_gpu);
  printf("Times: %.1f ms(COO_cpu), %.1f ms(ELL_gpu)\n\n", dtimeCOO_cpu, dtimeELL_gpu);
  
  // release host memory
  mtx_JDS_free(&mtxJDS);
  mtx_JDS_free(&mtxJDS_t);

  // free(h_vecIn);
  // free(h_vecOutCOO_cpu);
  // free(h_vecOutCSR_gpu);
  // free(h_vecOutELL_gpu);

	return 0;
}
