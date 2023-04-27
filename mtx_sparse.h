#ifdef __cplusplus

extern "C" {

#endif

#ifndef MTX_SPARSE
#define MTX_SPARSE

#include <stdio.h>

struct mtx_COO  // COOrdinates
{
    int *row;
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

struct mtx_CSR  // Compressed Sparse Row
{
    int *rowptr;
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

struct mtx_ELL      // ELLiptic (developed by authors of ellipctic package)
{
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int num_elements;
    int num_elementsinrow;    
};

int mtx_COO_create_from_file(struct mtx_COO *mCOO, FILE *f);
int mtx_COO_free(struct mtx_COO *mCOO);

int mtx_CSR_create_from_mtx_COO(struct mtx_CSR *mCSR, struct mtx_COO *mCOO);
int mtx_CSR_free(struct mtx_CSR *mCSR);

int mtx_ELL_create_from_mtx_CSR(struct mtx_ELL *mELL, struct mtx_CSR *mCSR);
int mtx_ELL_free(struct mtx_ELL *mELL);

#endif

#ifdef __cplusplus

}

#endif
