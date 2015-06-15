#include <stdio.h>
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mmio.h"

#ifdef __APPLE__

#include "cblas.h"

#else
#include <mkl.h>
#endif

int main(int argc, char **argv) {
    char *filename = NULL;
    int option = 0;

    while ((option = getopt(argc, argv, "f:")) != -1) {
        switch (option) {
            case 'f':
                filename = optarg;
                break;
            default:
                printf("Usage: f2b -f filename \n");
                return 0;
        }
    }
    if (filename == NULL) {
        printf("Usage: f2b -f filename \n");
        return 0;
    }


    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Unable to open file!");
        return 0;
    }

    MM_typecode matcode;

    if (mm_read_banner(file, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    int M, N, nz, i;


    if (mm_is_complex(matcode) && !mm_is_symmetric(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if ((mm_read_mtx_crd_size(file, &M, &N, &nz)) != 0)
        exit(1);

    if (M != N) {
        printf("Sorry, this matrix is not square \n");
        exit(1);
    }

    double *mat_a = (double *) malloc(N * N * sizeof(double));
    memset(mat_a, 0, N * N * sizeof(double));

    int ii, jj;
    double val;
    for (i = 0; i < nz; i++) {
        fscanf(file, "%d %d %lg\n", &ii, &jj, &val);
        ii--;
        jj--;

        mat_a[ii * N + jj] = val;
        if (ii != jj) {
            mat_a[jj * N + ii] = val;
        }
    }

    if (file != stdin)
        fclose(file);

    double *vec_b = (double *) malloc(N * sizeof(double));

    double *vec_u = (double *) malloc(N * sizeof(double));
    for (i = 0; i < N; ++i) {
        vec_u[i] = 1.0;
    }
    cblas_dsymv(CblasRowMajor, CblasUpper, N, 1.0, mat_a, N, vec_u, 1, 0.0, vec_b, 1);

    char outfilename[strlen(filename)+10];
    strcpy(outfilename, filename);
    strcat(outfilename, ".bin");

    FILE *ptr_file = fopen(outfilename, "wb");
    if (!ptr_file) {
        printf("Unable to open file!");
    } else {
        fwrite(&N, sizeof(int), 1, ptr_file);
        fwrite(mat_a, sizeof(double), (size_t) (N * N), ptr_file);
        fwrite(vec_b, sizeof(double), (size_t) N, ptr_file);
        fclose(ptr_file);
    }
    return 0;

}