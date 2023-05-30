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
    // Copy common values
    mJDS->num_nonzeros = mCSR->num_nonzeros;
    mJDS->num_rows = mCSR->num_rows;
    mJDS->num_cols = mCSR->num_cols;
    mJDS->num_elements = mCSR->rowptr[mCSR->num_rows];

    /*printf("CSR data: ");
    vecPrint(mCSR->data, mCSR->num_nonzeros);
    printf("CSR col: ");
    vecPrintInt(mCSR->col, mCSR->num_nonzeros);
    printf("CSR row_ptr: ");
    vecPrintInt(mCSR->rowptr, mCSR->num_rows + 1);*/
    
    if (mJDS->num_nonzeros < 1)
        return 1;

    // Allocate space for data (same as CSR format)
    mJDS->data = (double *)calloc(mJDS->num_elements, sizeof(double));
    if (!(mJDS->data)) {
        fprintf(stderr, "Failed to init mtx data\n");
        return 1;
    }
    
    // Allocated space for column data (same as CSR format)
    mJDS->col = (int *)calloc(mJDS->num_elements, sizeof(int));
    if (!(mJDS->col)) {
        fprintf(stderr, "Failed to init mtx column data\n");
        free(mJDS->data);
        return 1;
    }

    /*
        Filter null rows and reorder rows by size
    */
    int *rows = (int *)calloc(mCSR->num_rows, 2 * sizeof(int)); // [index, size]
    if (!rows) {
        fprintf(stderr, "Failed to init rows\n");
        free(mJDS->data);
        free(mJDS->col);
        return 1;
    }
    int row_notzero = 0;
    for (int i = 0; i < mCSR->num_rows; i++) { // Calculate sizes of rows
        rows[i * 2] = i; // row index
        rows[i * 2 + 1] = mCSR->rowptr[i + 1] - mCSR->rowptr[i]; // row size
        if (rows[i * 2 + 1] > 0)
            row_notzero++;
    }
    
    qsort(rows, mCSR->num_rows, 2 * sizeof(int), cmpfun);
    mJDS->max_el_in_row = rows[1]; // Max elements is size of the first row in sorted table (largest row)

    /*printf("[index, row size]: ");
    vecPrintInt(rows, mCSR->num_rows * 2);*/

    // Count number of rows for each size
    int *countPerSize = (int*)calloc(mJDS->max_el_in_row, sizeof(int));
    if (!countPerSize) {
        fprintf(stderr, "Failed to init countPerSize\n");
        free(mJDS->data);
        free(mJDS->col);
        free(rows);
        return 1;
    }
    for (int s = mJDS->max_el_in_row, i = 0; s >= 1; s--) {
        while(i < (mCSR->num_rows) * 2 && rows[i + 1] == s) {
            countPerSize[mJDS->max_el_in_row - s]++; // Increase counter for size
            i += 2; // Move to next [row index, size]
        }
    }

    /*printf("Count per size: ");
    vecPrintInt(countPerSize, mJDS->max_el_in_row);*/

    // Allocate space for row data
    mJDS->row_permute = (int *)calloc(row_notzero, sizeof(int));
    if (!(mJDS->row_permute)) {
        fprintf(stderr, "Failed to init mtx row permute\n");
        free(mJDS->data);
        free(mJDS->col);
        free(rows);
        free(countPerSize);
        return 1;
    }
    mJDS->jagged_ptr = (int *)calloc(mJDS->max_el_in_row, sizeof(int));
    if (!(mJDS->jagged_ptr)) {
        fprintf(stderr, "Failed to init mtx jagged_ptr\n");
        free(mJDS->data);
        free(mJDS->col);
        free(rows);
        free(countPerSize);
        free(mJDS->row_permute);
        return 1;
    }
    
    // Copy row contents
    for (int i = 0; i < mJDS->max_el_in_row; i++)
        mJDS->jagged_ptr[i] = mJDS->num_elements;

    //printf("El. no %d\n", mJDS->num_elements);
    //printf("Max el in row %d\n", mJDS->max_el_in_row);
    //printf("JDS jagged_ptr [initial]: ");
    //vecPrintInt(mJDS->jagged_ptr, mJDS->max_el_in_row);

    for (int r = 0, pos = 0, written = 0; r < row_notzero && pos < mJDS->num_elements; r++) {
        if (rows[2 * r + 1] < 1) // Only copy non-zero rows
            break;
        for (int i = 0; i < rows[2 * r + 1]; i++) { // Copy row element by element
            mJDS->data[pos + i * countPerSize[mJDS->max_el_in_row - rows[2 * r + 1]]] = mCSR->data[mCSR->rowptr[rows[2 * r]] + i];
            mJDS->col[pos + i * countPerSize[mJDS->max_el_in_row - rows[2 * r + 1]]] = mCSR->col[mCSR->rowptr[rows[2 * r]] + i];
        }
        mJDS->row_permute[r] = rows[2 * r]; // Save row index

        if (r == 0 || rows[2 * r + 1] != rows[2 * (r - 1) + 1]) {
            // Prefill values if there is no rows of certaion size
            for (int s = rows[2 * (r - 1) + 1] - 1; s > rows[2 * r + 1]; s--) {
                mJDS->jagged_ptr[mJDS->max_el_in_row - s] = written;
            }
            mJDS->jagged_ptr[mJDS->max_el_in_row - rows[2 * r + 1]] = written; // Save start pos of row
        }

        if (r + 1 < row_notzero && rows[2 * r + 1] == rows[2 * (r + 1) + 1]) {
            pos++; // If next row has same size start writing next to start of previous row
        } else {
            pos += rows[2 * r + 1]; // Increase write position by row size
        }
        written += rows[2 * r + 1]; // Increase total written values
    }


    free(rows);
    free(countPerSize);
    return 0;
}

int mtx_JDSrow_create_from_mtx_CSR(struct mtx_JDS *mJDS, struct mtx_CSR *mCSR) {
    // Copy common values
    mJDS->num_nonzeros = mCSR->num_nonzeros;
    mJDS->num_rows = mCSR->num_rows;
    mJDS->num_cols = mCSR->num_cols;
    mJDS->num_elements = mCSR->rowptr[mCSR->num_rows];

    /*printf("CSR data: ");
    vecPrint(mCSR->data, mJDS->num_nonzeros);
    printf("CSR col: ");
    vecPrintInt(mCSR->col, mJDS->num_nonzeros);
    printf("CSR row_ptr: ");
    vecPrintInt(mCSR->rowptr, mJDS->num_rows);
    printf("========\n");*/

    if (mJDS->num_nonzeros < 1)
        return 1;

    // Allocate space for data (same as CSR format)
    mJDS->data = (double *)calloc(mJDS->num_elements, sizeof(double));
    if (!(mJDS->data)) {
        fprintf(stderr, "Failed to init mtx data\n");
        return 1;
    }
    
    // Allocated space for column data (same as CSR format)
    mJDS->col = (int *)calloc(mJDS->num_elements, sizeof(int));
    if (!(mJDS->col)) {
        fprintf(stderr, "Failed to init mtx column data\n");
        free(mJDS->data);
        return 1;
    }

    /*
        Filter null rows and reorder rows by size
    */
    int *rows = (int *)calloc(mCSR->num_rows, 2 * sizeof(int)); // [index, size]
    if (!rows) {
        fprintf(stderr, "Failed to init rows\n");
        free(mJDS->data);
        free(mJDS->col);
        return 1;
    }
    int row_notzero = 0;
    for (int i = 0; i < mCSR->num_rows; i++) { // Calculate sizes of rows
        rows[i * 2] = i; // row index
        rows[i * 2 + 1] = mCSR->rowptr[i + 1] - mCSR->rowptr[i]; // row size
        if (rows[i * 2 + 1] > 0)
            row_notzero++;
    }
    
    qsort(rows, mCSR->num_rows, 2 * sizeof(int), cmpfun);
    mJDS->max_el_in_row = rows[1]; // Biggest row is at the start
    /*printf("***\n");
    printf("[index, row size]: ");
    vecPrintInt(rows, mCSR->num_rows * 2);
    printf("***\n");*/
    int *countPerSize = (int*)calloc(mJDS->max_el_in_row, sizeof(int));
    if (!countPerSize) {
        fprintf(stderr, "Failed to init countPerSize\n");
        free(mJDS->data);
        free(mJDS->col);
        free(rows);
        return 1;
    }
    for (int s = mJDS->max_el_in_row, i = 0; s >= 1; s--) {
        while(i < (mCSR->num_rows) * 2 && rows[i + 1] == s) {
            countPerSize[mJDS->max_el_in_row - s]++; // Increase counter for size
            i += 2; // Move to next [row index, size]
        }
    }

    // Allocate space for row data
    mJDS->row_permute = (int *)calloc(row_notzero, sizeof(int));
    if (!(mJDS->row_permute)) {
        fprintf(stderr, "Failed to init mtx row permute\n");
        free(mJDS->data);
        free(mJDS->col);
        free(rows);
        return 1;
    }
    mJDS->jagged_ptr = (int *)calloc(mJDS->max_el_in_row, sizeof(int));
    if (!(mJDS->jagged_ptr)) {
        fprintf(stderr, "Failed to init mtx jagged_ptr\n");
        free(mJDS->data);
        free(mJDS->col);
        free(rows);
        free(mJDS->row_permute);
        return 1;
    }
    // Prepare jagged_ptr
    mJDS->jagged_ptr[0] = 0; // First jag starts at the start
    for (int s = mJDS->max_el_in_row - 1; s >= 1; s--) {
        mJDS->jagged_ptr[mJDS->max_el_in_row - s] = mJDS->jagged_ptr[mJDS->max_el_in_row - s - 1] + (s + 1) * countPerSize[mJDS->max_el_in_row - s - 1];
    }
    for (int r = 0, pos = 0; r < row_notzero; r++) {
        if (rows[2 * r + 1] < 1) // Only copy non-zero rows
            break;
        memcpy(&(mJDS->data[pos]), &(mCSR->data[mCSR->rowptr[rows[2 * r]]]), rows[2 * r + 1] * sizeof(double)); // Copy row data
        memcpy(&(mJDS->col[pos]), &(mCSR->col[mCSR->rowptr[rows[2 * r]]]), rows[2 * r + 1] * sizeof(int)); // Copy row data
        //mJDS->jagged_ptr[r] = pos; // Save start pos of row
        mJDS->row_permute[r] = rows[2 * r]; // Save row index

        pos += rows[2 * r + 1]; // Increase write position by row size
    }

    /*printf("JDS data: ");
    vecPrint(mJDS->data, mJDS->num_nonzeros);
    printf("JDS coll: ");
    vecPrintInt(mJDS->col, mJDS->num_nonzeros);
    printf("JDS jagged_ptr: ");
    vecPrintInt(mJDS->jagged_ptr, row_notzero);
    printf("JDS data: ");
    vecPrintInt(mJDS->row_permute, row_notzero);
    printf("JDS max el in row: ");
    printf("%d\n", mJDS->max_el_in_row);*/

    free(rows);
    free(countPerSize);
    //free(jag_write_ptr);
    return 0;
}

int mtx_JDS_free(struct mtx_JDS *mJDS) {
    free(mJDS->col);
    free(mJDS->data);
    free(mJDS->jagged_ptr);
    free(mJDS->row_permute);
    return 0;
}