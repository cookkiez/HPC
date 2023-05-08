#include <stdio.h>
#include <stdlib.h>
#include "mtx_sparse.h"

struct mtx_MM
{
    int row;
    int col;
    float data;
};

int mtx_COO_compare(const void * a, const void * b)
{   
    struct mtx_MM aa = *(struct mtx_MM *)a;
    struct mtx_MM bb = *(struct mtx_MM *)b;

    if (aa.row < bb.row)
        return -1;
    else if (aa.row > bb.row)
        return +1;
    else if (aa.col < bb.col)
        return -1;
    else if (aa.col > bb.col)
        return +1;
    else 
        return 0;
}

int mtx_COO_create_from_file(struct mtx_COO *mCOO, FILE *f)
{
    char line[1024];

    // skip comments
    do 
    {
        if (fgets(line, 1024, f) == NULL) 
            return 1;
    } 
    while (line[0] == '%');
    // get matrix size
    if (sscanf(line, "%d %d %d", &(mCOO->num_rows), &(mCOO->num_cols), &(mCOO->num_nonzeros)) != 3)
        return 1;
    // allocate matrix
    struct mtx_MM *mMM = (struct mtx_MM *)malloc(mCOO->num_nonzeros * sizeof(struct mtx_MM));
    mCOO->data = (float *) malloc(mCOO->num_nonzeros * sizeof(float));
    mCOO->col = (int *) malloc(mCOO->num_nonzeros * sizeof(int));
    mCOO->row = (int *) malloc(mCOO->num_nonzeros * sizeof(int));
    // read data
    for (int i = 0; i < mCOO->num_nonzeros; i++)
    {
        fscanf(f, "%d %d %f\n", &mMM[i].row, &mMM[i].col, &mMM[i].data);
        mMM[i].row--;  /* adjust from 1-based to 0-based row/column */
        mMM[i].col--;
    }    
    fclose(f);

    // sort elements
    qsort(mMM, mCOO->num_nonzeros, sizeof(struct mtx_MM), mtx_COO_compare);

    // copy to mtx_COO structures (GPU friendly)
    for (int i = 0; i < mCOO->num_nonzeros; i++)
    {
        mCOO->data[i] = mMM[i].data;
        mCOO->row[i] = mMM[i].row;
        mCOO->col[i] = mMM[i].col;
    }

    free(mMM);

    return 0;
}

int mtx_COO_free(struct mtx_COO *mCOO)
{
    free(mCOO->data);
    free(mCOO->col);
    free(mCOO->row);

    return 0;
}

int mtx_CSR_create_from_mtx_COO(struct mtx_CSR *mCSR, struct mtx_COO *mCOO)
{
    mCSR->num_nonzeros = mCOO->num_nonzeros;
    mCSR->num_rows = mCOO->num_rows;
    mCSR->num_cols = mCOO->num_cols;

    mCSR->data =  (float *)malloc(mCSR->num_nonzeros * sizeof(float));
    mCSR->col = (int *)malloc(mCSR->num_nonzeros * sizeof(int));
    mCSR->rowptr = (int *)calloc(mCSR->num_rows + 1, sizeof(int));
    mCSR->data[0] = mCOO->data[0];
    mCSR->col[0] = mCOO->col[0];
    mCSR->rowptr[0] = 0;
    mCSR->rowptr[mCSR->num_rows] = mCSR->num_nonzeros;
    for (int i = 1; i < mCSR->num_nonzeros; i++)
    {
        mCSR->data[i] = mCOO->data[i];
        mCSR->col[i] = mCOO->col[i];
        if (mCOO->row[i] > mCOO->row[i-1])
        {
            int r = mCOO->row[i];
            while (r > 0 && mCSR->rowptr[r] == 0)
                mCSR->rowptr[r--] = i;
        }
    }

    return 0;
}

int mtx_CSR_free(struct mtx_CSR *mCSR)
{
    free(mCSR->data);
    free(mCSR->col);
    free(mCSR->rowptr);

    return 0;
}

int mtx_ELL_create_from_mtx_CSR(struct mtx_ELL *mELL, struct mtx_CSR *mCSR)
{
    mELL->num_nonzeros = mCSR->num_nonzeros;
    mELL->num_rows = mCSR->num_rows;
    mELL->num_cols = mCSR->num_cols;
    mELL->num_elementsinrow = 0;

    for (int i = 0; i < mELL->num_rows; i++)
        if (mELL->num_elementsinrow < mCSR->rowptr[i+1]-mCSR->rowptr[i]) 
            mELL->num_elementsinrow = mCSR->rowptr[i+1]-mCSR->rowptr[i];
    mELL->num_elements = mELL->num_rows * mELL->num_elementsinrow;
    mELL->data = (float *)calloc(mELL->num_elements, sizeof(float));
    mELL->col = (int *) calloc(mELL->num_elements, sizeof(int));    
    for (int i = 0; i < mELL->num_rows; i++)
    {
        for (int j = mCSR->rowptr[i]; j < mCSR->rowptr[i+1]; j++)
        {            
            int ELL_j = (j - mCSR->rowptr[i]) * mELL->num_rows + i;
            mELL->data[ELL_j] = mCSR->data[j];
            mELL->col[ELL_j] = mCSR->col[j];
        }
    }

    return 0;
}

int mtx_ELL_free(struct mtx_ELL *mELL)
{
    free(mELL->col);
    free(mELL->data);

    return 0;
}

int mtx_JDS_create_from_mtx_CSR(struct mtx_JDS *mJDS, struct mtx_CSR *mCSR) {
    mJDS->num_nonzeros = mCSR->num_nonzeros;
    mJDS->num_rows = mCSR->num_rows;
    mJDS->num_cols = mCSR->num_cols;
    mJDS->num_elements = mCSR->rowptr[mCSR->num_rows];

    int *ordered = (int *)calloc(mCSR->num_rows, sizeof(int));
    mJDS->data = (float *)calloc(mJDS->num_elements, sizeof(float));
    mJDS->col = (int *) calloc(mJDS->num_elements, sizeof(int));
    mJDS->row_permute = (int *) calloc(mJDS->num_rows, sizeof(int));
    int rows = mJDS->num_rows;
    // Get number of elements in each row
    for (int i = 0; i < rows; i++){
        ordered[i] = mCSR->rowptr[i + 1] - mCSR->rowptr[i];
        // printf("%d %d %d %d\n", ordered[i], rows, mCSR->rowptr[i], mCSR->rowptr[i + 1]);
        mJDS->row_permute[i] = i;
    } 
    // printf("SORT\n");
    // Bubble sort over rows
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows - i; j++) {
            if (ordered[j] < ordered[j + 1]) {
                // printf("%d %d %d %d\n", ordered[j], ordered[j + 1], mJDS->row_permute[j], mJDS->row_permute[j + 1]);
                int temp = ordered[j];
                ordered[j] = ordered[j + 1];
                ordered[j + 1] = temp;

                temp = mJDS->row_permute[j];
                mJDS->row_permute[j] = mJDS->row_permute[j + 1];
                mJDS->row_permute[j + 1] = temp;
            }
        }
    }

    mJDS->max_el_in_row = ordered[0];
    mJDS->jagged_ptr = (int *) calloc(ordered[0], sizeof(int));
    int *els_in_jag_row = (int *) calloc(ordered[0], sizeof(int));
    mJDS->jagged_ptr[0] = 0;
    int data_ix = 0;
    int curr_els = ordered[0] + 1;
    int jag_ix = 0;
    for (int row = 0; row < rows; row++) {
        int curr_row = mJDS->row_permute[row];
        if (ordered[row] < curr_els) {
            //printf("%d, %d, %d\n", curr_row, data_ix, curr_els);
            curr_els = ordered[row];
            mJDS->jagged_ptr[jag_ix] = data_ix; 
            els_in_jag_row[jag_ix] = curr_els;
            jag_ix++;
        }
        for (int i = mCSR->rowptr[curr_row]; i < mCSR->rowptr[curr_row + 1]; i++) {
            mJDS->data[data_ix] = mCSR->data[i];
            mJDS->col[data_ix] = mCSR->col[i];
            data_ix++;
        }
    }
    if (mJDS->jagged_ptr[jag_ix] == 0) { mJDS->jagged_ptr[jag_ix] = mJDS->num_elements; }
    
    for (int jag = 0; jag < mJDS->max_el_in_row; jag++) {
        int jag_start = mJDS->jagged_ptr[jag];
        int jag_end = mJDS->jagged_ptr[jag + 1];
        int curr_els = jag_end - jag_start;
        int els_in_row = els_in_jag_row[jag];
        for (int i = 1; i <= els_in_row; i++) {
            for (int j = jag_start + i; j < jag_end - els_in_row; j += els_in_row) {
                int temp = mJDS->col[j];
                mJDS->col[j] = mJDS->col[j + 1];
                mJDS->col[j + 1] = temp;

                float f_temp = mJDS->data[j];
                mJDS->data[j] = mJDS->data[j + 1];
                mJDS->data[j + 1] = f_temp;
            }
        }
    } 

    free(ordered);
    free(els_in_jag_row);
}

int mtx_JDS_free(struct mtx_JDS *mJDS) {
    free(mJDS->col);
    free(mJDS->data);
    free(mJDS->jagged_ptr);
    free(mJDS->row_permute);
    return 0;
}