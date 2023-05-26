#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mtx_sparse.h"

struct mtx_MM
{
    int row;
    int col;
    double data;
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

int mtx_COO_create_from_file(struct mtx_COO *mCOO, FILE *f, bool transposed)
{
    char line[1024];
    // skip comments
    do
    {
        if (fgets(line, 1024, f) == NULL)
            return 1;
    }
    while (line[0] == '%');

    // get matrix  size
    if (sscanf(line, "%d %d %d", &(mCOO->num_rows), &(mCOO->num_cols), &(mCOO->num_nonzeros)) != 3)
        return 1;
    // allocate matrix
    struct mtx_MM *mMM = (struct mtx_MM *)malloc(mCOO->num_nonzeros * sizeof(struct mtx_MM));
    mCOO->data = (double *) malloc(mCOO->num_nonzeros * sizeof(double));
    mCOO->col = (int *) malloc(mCOO->num_nonzeros * sizeof(int));
    mCOO->row = (int *) malloc(mCOO->num_nonzeros * sizeof(int));

    // read data
    for (int i = 0; i < mCOO->num_nonzeros; i++)
    {
        if (transposed)
            fscanf(f, "%d %d %lf\n", &mMM[i].col, &mMM[i].row, &mMM[i].data);
        else
            fscanf(f, "%d %d %lf\n", &mMM[i].row, &mMM[i].col, &mMM[i].data);
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

    mCSR->data =  (double *)malloc(mCSR->num_nonzeros * sizeof(double));
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
    int prev_value = mCSR->rowptr[0];
    for (int i = 1; i < mCSR->num_rows + 1; i++) {
        if (mCSR->rowptr[i] == 0) { mCSR->rowptr[i] = prev_value; }
        else { prev_value = mCSR->rowptr[i]; }
    }

    // printf("CSAR\n");
    // vecPrint(mCSR->data, mCSR->num_nonzeros);
    // vecPrintInt(mCSR->rowptr, mCSR->num_rows + 1);
    // vecPrintInt(mCSR->col, mCSR->num_nonzeros);
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
    mELL->data = (double *)calloc(mELL->num_elements, sizeof(double));
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


void vecPrint(double *vecIn, int n) {
  printf("[");
  for (int i = 0; i < n; i++) {
    if (i == 0)
      printf("%lf", vecIn[i]);
    else
      printf(",%lf", vecIn[i]);
  }
  printf("]\n");
}


void vecPrintInt(int *vecIn, int n) {
  printf("[");
  for (int i = 0; i < n; i++) {
    if (i == 0)
      printf("%d", vecIn[i]);
    else
      printf(",%d", vecIn[i]);
  }
  printf("]\n");
}

int cmpfun(const void *a, const void *b) {
    return ( ((int*)b)[1] - ((int*)a)[1] );
}

int mtx_JDS_create_from_mtx_CSR(struct mtx_JDS *mJDS, struct mtx_CSR *mCSR) {
    mJDS->num_nonzeros = mCSR->num_nonzeros;
    mJDS->num_rows = mCSR->num_rows;
    mJDS->num_cols = mCSR->num_cols;
    mJDS->num_elements = mCSR->rowptr[mCSR->num_rows];
    mJDS->data = (double *)calloc(mJDS->num_elements, sizeof(double));
    mJDS->col = (int *) calloc(mJDS->num_elements, sizeof(int));
    mJDS->row_permute = (int *) calloc(mJDS->num_rows, sizeof(int));
    int rows = mJDS->num_rows;
    // Get number of elements in each row
    for (int i = 0; i < rows; i++){
        //ordered[i] = mCSR->rowptr[i + 1] - mCSR->rowptr[i];
        //printf("%d %d %d %d\n", ordered[i], rows, mCSR->rowptr[i], mCSR->rowptr[i + 1]);
        mJDS->row_permute[i] = i;
    }

    // int *rows = (int *)calloc(mCSR->num_rows, 2 * sizeof(int)); // [index, size]
    // if (!rows) {
    //     fprintf(stderr, "Failed to init rows\n");
    //     free(mJDS->data);
    //     free(mJDS->col);
    //     return 1;
    // }
    // int row_notzero = 0;
    // for (int i = 0; i < mCSR->num_rows; i++) { // Calculate sizes of rows
    //     rows[i * 2] = i; // row index
    //     rows[i * 2 + 1] = mCSR->rowptr[i + 1] - mCSR->rowptr[i]; // row size
    //     if (rows[i * 2 + 1] > 0)
    //         row_notzero++;
    // }
    
    // qsort(rows, mCSR->num_rows, 2 * sizeof(int), cmpfun);

    //printf("SORT\n");
    //Bubble sort over rows
    //printf("%d \n", rows);
    // for (int i = 0; i < rows; i++) {
    //     // printf("Sorting: %d\n", i);
    //     for (int j = 0; j < rows - i - 1; j++) {
    //         if (ordered[j] < ordered[j + 1]) {
    //             //printf("%d %d %d %d %d %d\n", j, j + 1,ordered[j], ordered[j + 1], mJDS->row_permute[j], mJDS->row_permute[j + 1]);
    //             int temp = ordered[j];
    //             ordered[j] = ordered[j + 1];
    //             ordered[j + 1] = temp;
    //             temp = mJDS->row_permute[j];
    //             mJDS->row_permute[j] = mJDS->row_permute[j + 1];
    //             mJDS->row_permute[j + 1] = temp;
    //         }
    //     }
    // }

    int *order_rows = (int *)calloc(mCSR->num_rows, 2 * sizeof(int)); // [index, size]
    if (!order_rows) {
        fprintf(stderr, "Failed to init rows\n");
        free(mJDS->data);
        free(mJDS->col);
        return 1;
    }
    int row_notzero = 0;
    for (int i = 0; i < mCSR->num_rows; i++) { // Calculate sizes of rows
        order_rows[i * 2] = i; // row index
        order_rows[i * 2 + 1] = mCSR->rowptr[i + 1] - mCSR->rowptr[i]; // row size
        if (order_rows[i * 2 + 1] > 0)
            row_notzero++;
    }
    
    qsort(order_rows, mCSR->num_rows, 2 * sizeof(int), cmpfun);
    // for (int i = 0; i < mCSR->num_rows * 2; i+=2) {
    //     ordered[ord] = order_rows[i];
    //     ord++;
    // }
    //free(order_rows);
    //printf("[index, row size]: ");
    //vecPrintInt(order_rows, mCSR->num_rows * 2);
    
    for (int i = 0; i < mCSR->num_rows; i++) {
        mJDS->row_permute[i] = order_rows[i * 2];
    }
    //vecPrintInt(mJDS->row_permute, mCSR->num_rows);
    //mJDS->max_el_in_row = rows[1];
    mJDS->max_el_in_row = order_rows[1];
    //printf("%d\n\n", ordered[0]);
    mJDS->jagged_ptr = (int *) calloc(order_rows[1], sizeof(int));
    int *els_in_jag_row = (int *) calloc(order_rows[1], sizeof(int));
    mJDS->jagged_ptr[0] = 0;

    int data_ix = 0;
    int curr_els = order_rows[1] + 1;
    int jag_ix = 0;
    int prev_ordered = order_rows[1];
    for (int row = 0; row < rows; row++) {
        int curr_row = mJDS->row_permute[row];
        if (order_rows[row * 2 + 1] < curr_els) {
            //printf("%d, %d, %d\n", curr_row, data_ix, curr_els);
            curr_els = order_rows[row * 2 + 1];
            while (curr_els < prev_ordered) {
                // printf("%d %d %d %d\n", curr_els, prev_ordered, jag_ix, data_ix);
                // fflush(stdout);
                mJDS->jagged_ptr[jag_ix] = data_ix;
                els_in_jag_row[jag_ix] = 0;
                jag_ix++; 
                prev_ordered--;
            }
            mJDS->jagged_ptr[jag_ix] = data_ix;
            els_in_jag_row[jag_ix] = curr_els;
            jag_ix++;
            prev_ordered = curr_els - 1;
        }
        for (int i = mCSR->rowptr[curr_row]; i < mCSR->rowptr[curr_row + 1]; i++) {
            data_ix++;
        }
    }
    if (mJDS->jagged_ptr[jag_ix] == 0) { mJDS->jagged_ptr[jag_ix] = mJDS->num_elements; }

    jag_ix = 0;
    int row_in_jag = 0;
    int jag_start = 0;
    int num_rows_jag = 0;
    data_ix = 0;
    curr_els = order_rows[1] + 1;
    for (int row = 0; row < rows; row++) {
        int curr_row = mJDS->row_permute[row];
        if (order_rows[row * 2 + 1] < curr_els) {
            curr_els = order_rows[row * 2 + 1];
            jag_start = mJDS->jagged_ptr[jag_ix];
            jag_ix++;
            row_in_jag = 0;
            if (curr_els == 0) { break; }
            num_rows_jag = (mJDS->jagged_ptr[jag_ix] - jag_start) / curr_els;
            while(num_rows_jag == 0) {
                jag_start = mJDS->jagged_ptr[jag_ix];
                jag_ix++;
                row_in_jag = 0;
                if (curr_els == 0) { break; }
                num_rows_jag = (mJDS->jagged_ptr[jag_ix] - jag_start) / curr_els;
                // printf("%d %d %d\n", num_rows_jag, jag_ix, curr_els);
                // fflush(stdout);
            }
        }
        int cnt = 0;
        for (int i = mCSR->rowptr[curr_row]; i < mCSR->rowptr[curr_row + 1]; i++) {
            int current_ix = jag_start + row_in_jag + num_rows_jag * cnt;
            mJDS->data[current_ix] = mCSR->data[i];
            mJDS->col[current_ix] = mCSR->col[i];
            //printf("%lf, %d %d %d %d %d %d %d\n", mCSR->data[i], mCSR->col[i], i, current_ix, cnt, row_in_jag, num_rows_jag, jag_start);
            data_ix++;
            cnt++;
        }
        row_in_jag++;
    }

    // vecPrintInt(mJDS->jagged_ptr, mJDS->max_el_in_row + 1);
    // vecPrint(mJDS->data, mJDS->num_nonzeros);
    // vecPrintInt(mJDS->row_permute, mJDS->num_rows);
    // vecPrintInt(mJDS->col, mJDS->num_nonzeros);
    // vecPrintInt(els_in_jag_row, ordered[0]);
    // fflush(stdout);
    
    //free(ordered);
    free(els_in_jag_row);
    return 0;
}

int mtx_JDS_free(struct mtx_JDS *mJDS) {
    free(mJDS->col);
    free(mJDS->data);
    free(mJDS->jagged_ptr);
    free(mJDS->row_permute);
    return 0;
}