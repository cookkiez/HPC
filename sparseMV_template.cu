// module load CUDA/10.1.243-GCC-8.3.0
// nvcc -Xcompiler -o SparseMV SparseMV.cu mtx_sparse.c
// srun --reservation=fri --gpus=1 SparseMV data/scircuit.mtx 
// srun --reservation=fri --gpus=1 SparseMV data/pdb1HYS.mtx
// srun --reservation=fri -G1 -n1 sparseMV data/pwtk.mtx

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "mtx_sparse.h"
#include "helper_cuda.h"

#define THREADS_PER_BLOCK 256
#define REPEAT 1

__global__ void mELLxVec(int *col, float *data, float *vin, float *vout, int rows, int elemsinrow){		
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


int main(int argc, char *argv[])
{
    FILE *f;
    struct mtx_COO h_mCOO;
    struct mtx_CSR h_mCSR;
    struct mtx_ELL h_mELL;
    struct mtx_JDS h_mJDS;
    int repeat;

    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    // create sparse matrices
    if (mtx_COO_create_from_file(&h_mCOO, f) != 0)
        exit(1);
    mtx_CSR_create_from_mtx_COO(&h_mCSR, &h_mCOO);
    mtx_ELL_create_from_mtx_CSR(&h_mELL, &h_mCSR);
    mtx_JDS_create_from_mtx_CSR(&h_mJDS, &h_mCSR);
    printf("%d", h_mJDS.num_elements);
    for (int i = 0; i < h_mJDS.num_elements; i++) {
        printf("%f, ", h_mJDS.data[i]);
    }
    printf("\n");
    // for (int i = 0; i < h_mJDS->max; i++) {
    //     printf("%f, ", h_mJDS->data[i]);
    // }
    // printf("\n");
    // allocate vectors
    // float *h_vecIn = (float *)malloc(h_mCOO.num_cols * sizeof(float));
    // for (int i = 0; i < h_mCOO.num_cols; i++)
    //     h_vecIn[i] = 1.0;
    // float *h_vecOutCOO_cpu = (float *)calloc(h_mCOO.num_rows, sizeof(float));
    // float *h_vecOutCSR_gpu = (float *)calloc(h_mCSR.num_rows, sizeof(float));
    // float *h_vecOutELL_gpu = (float *)calloc(h_mELL.num_rows, sizeof(float));
    // float* h_vecOutCSRpar = (float*)calloc(h_mCSR.num_rows, sizeof(float));

    // // compute with COO
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    // for (repeat = 0; repeat < REPEAT; repeat++)
    // {
    //     for (int i = 0; i < h_mCOO.num_rows; i++)
    //         h_vecOutCOO_cpu[i] = 0.0;
    //     for (int i = 0; i < h_mCOO.num_nonzeros; i++)
    //         h_vecOutCOO_cpu[h_mCOO.row[i]] += h_mCOO.data[i] * h_vecIn[h_mCOO.col[i]];
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float dtimeCOO_cpu = 0;
    // cudaEventElapsedTime(&dtimeCOO_cpu, start, stop);

    // // allocate memory on device and transfer data from host 
    // // CSR
    // int *d_mCSRrowptr, *d_mCSRcol;
    // float *d_mCSRdata;
    // cudaMalloc((void **)&d_mCSRrowptr, (h_mCSR.num_rows + 1) * sizeof(int));
    // cudaMalloc((void **)&d_mCSRcol, (h_mCSR.num_nonzeros + 1) * sizeof(int));
    // cudaMalloc((void **)&d_mCSRdata, h_mCSR.num_nonzeros * sizeof(float));
    // cudaMemcpy(d_mCSRrowptr, h_mCSR.rowptr, (h_mCSR.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_mCSRcol, h_mCSR.col, h_mCSR.num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_mCSRdata, h_mCSR.data, h_mCSR.num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    // // ELL
    // int *d_mELLcol;
    // float *d_mELLdata;
    // cudaMalloc((void **)&d_mELLcol, h_mELL.num_elements * sizeof(int));
    // cudaMalloc((void **)&d_mELLdata, h_mELL.num_elements * sizeof(float));
    // cudaMemcpy(d_mELLcol, h_mELL.col, h_mELL.num_elements * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_mELLdata, h_mELL.data, h_mELL.num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // // vectors
    // float *d_vecIn, *d_vecOut;
    // cudaMalloc((void **)&d_vecIn, h_mCOO.num_cols * sizeof(float));
    // cudaMalloc((void **)&d_vecOut, h_mCOO.num_rows * sizeof(float));
    // cudaMemcpy(d_vecIn, h_vecIn, h_mCSR.num_cols*sizeof(float), cudaMemcpyHostToDevice);
  
    
	// // Divide work 
    // dim3 blocksize(THREADS_PER_BLOCK);
    
    // // CSR
    // dim3 gridsize_CSR((h_mCSR.num_rows - 1) / blocksize.x + 1);
    
    // // ELL
    // dim3 gridsize_ELL((h_mELL.num_rows - 1) / blocksize.x + 1);

    
	// // CSR execute
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

    // // CSRPar execute
    // cudaEventRecord(start);
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


	// // ELL write, execute, read
    // cudaEventRecord(start);
    // for (repeat = 0; repeat < REPEAT; repeat++)
    // {
    //     mELLxVec<<<gridsize_ELL, blocksize>>>(d_mELLcol, d_mELLdata, d_vecIn, d_vecOut, h_mELL.num_rows, h_mELL.num_elementsinrow);   
    // }
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float dtimeELL_gpu = 0;
    // cudaEventElapsedTime(&dtimeELL_gpu, start, stop);
    // cudaMemcpy(h_vecOutELL_gpu, d_vecOut, h_mELL.num_rows*sizeof(float), cudaMemcpyDeviceToHost);
    
    // // release device memory
    // cudaFree(d_mCSRrowptr);
    // cudaFree(d_mCSRcol);
    // cudaFree(d_mCSRdata);
    // cudaFree(d_mELLcol);
    // cudaFree(d_mELLdata);
    // cudaFree(d_vecIn);
    // cudaFree(d_vecOut);

    // // output
    // printf("Matrix: %s, size: %d x %d, nonzero: %d, max elems in row: %d\n", argv[1],h_mCOO.num_rows, h_mCOO.num_cols, h_mCOO.num_nonzeros, h_mELL.num_elementsinrow);
    // int errorsCSR_gpu = 0;
    // int errorsCSRPar_gpu = 0;
    // int errorsELL_gpu = 0;
    // for(int i = 0; i < h_mCOO.num_rows; i++)
    // {
    //     if (fabs(h_vecOutCOO_cpu[i] - h_vecOutCSR_gpu[i]) > 1e-4 )
    //         errorsCSR_gpu++;
    //     if (fabs(h_vecOutCOO_cpu[i] - h_vecOutCSRpar[i]) > 1e-4){
    //         errorsCSRPar_gpu++;
    //         //printf("%d, %.4f, %.4f, %.4f \n", i, h_vecOutCOO_cpu[i], h_vecOutCSRpar[i], fabs(h_vecOutCOO_cpu[i] - h_vecOutCSRpar[i]));
    //     }
            
    //     if (fabs(h_vecOutCOO_cpu[i]-h_vecOutELL_gpu[i]) > 1e-4 )
    //         errorsELL_gpu++;
    // }
    // printf("Errors: %d(CSR_gpu), %d(CSRPar_gpu), %d(ELL_gpu)\n", errorsCSR_gpu, errorsCSRPar_gpu,errorsELL_gpu);
    // printf("Times: %.1f ms(COO_cpu), %.1f ms(CSR_gpu), %.1f ms (CSRPar_gpu), %.1f ms(ELL_gpu)\n\n", dtimeCOO_cpu, dtimeCSR_gpu, dtimeCSRPar_gpu, dtimeELL_gpu);
    // // release host memory
    mtx_COO_free(&h_mCOO);
    mtx_CSR_free(&h_mCSR);
    mtx_ELL_free(&h_mELL);
    mtx_JDS_free(&h_mJDS);

    // free(h_vecIn);
    // free(h_vecOutCOO_cpu);
    // free(h_vecOutCSR_gpu);
    // free(h_vecOutELL_gpu);

	return 0;
}
