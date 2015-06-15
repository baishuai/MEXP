#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>

#ifdef __APPLE__

#include "cblas.h"
#include "lapacke.h"

#else
#include <mkl.h>
#endif

double norm_inf_mat_Y11(double *X, int n) {
    double lambda = 0.0, lambda_t;
    int i;
    for (i = 0; i < n; ++i) {
        lambda_t = cblas_dasum(n, X + (i * n), 1);
        if (lambda_t > lambda)
            lambda = lambda_t;
    }
    return lambda;
}

double Random(double min, double max) {

    double random = ((double) rand()) / (double) RAND_MAX;
    double range = max - min;
    return (random * range) + min;
}

int main(int argc, char **argv) {

    int N = -1, method = 1, trans = 0;
    double alpha = 1.0;
    char *filename = NULL;
    int option = 0, i;
    while ((option = getopt(argc, argv, "t:mn:f:")) != -1) {
        switch (option) {
            case 'm':
                method = 2;
                break;
            case 't':
                trans = atoi(optarg);
                break;
            case 'n':
                N = atoi(optarg);
                break;
            case 'f':
                filename = optarg;
                break;
            default:
                printf("Usage: gen [-m] -n num [-t num] -f string \n");
                return 0;
        }
    }
    if (N == -1 || filename == NULL) {
        printf("Usage: gen [-m] -n num [-t num] -f string \n");
        return 0;
    }

    srand((unsigned int) (time(NULL)));
    double *A = (double *) malloc(N * N * sizeof(double));
    for (i = 0; i < N * N; ++i) {
        A[i] = Random(-1, 1);
    }

    assert(trans >= 0);
    if (method == 1) {
        /*
         * method 1: A * A'
         */
        double *C = (double *) malloc(N * N * sizeof(double)), *T = NULL;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, A, N, A, N, 0.0, C, N);
        T = A;
        A = C;
        C = T;
        free(C);
    }
    if (method == 2) {
        /* method 2: (A + A')/2 + D
         * first part is to make matrix symmetric, adding D to make it diagonally
         * dominant. Use parameter alpha>1 to control the weight of the diagonal
         * elements. The large alpha is, the larger the condition number of the
         * system is.
         */
        {   // A = ( A + A')/2.0
            double *AT = (double *) malloc(N * N * sizeof(double));
            for (i = 0; i < N; ++i) {           //AT = Trans A
                cblas_dcopy(N, A + i, N, AT + i * N, 1);
            }
            cblas_dscal(N * N, 0.5, A, 1);          //A <- 0.5*A
            cblas_daxpy(N * N, 0.5, AT, 1, A, 1);   //A <- 0.5*AT + A
            free(AT);
        }

        double *D = (double *) malloc(N * sizeof(double));
        memset(D, 0, N * sizeof(double));
        cblas_dcopy(N, D, 1, A, N + 1);             // A = tril(A,-1) + triu(A,1)

        alpha = pow(10.0, trans);
        for (i = 0; i < N; ++i) {               // diag(D + alpha*rand(n,1))
            D[i] = cblas_dasum(N, A + (i * N), 1) + alpha * Random(0.0, 1.0);
        }
        cblas_dcopy(N, D, 1, A, N + 1);
        free(D);
    }

    double *B = (double *) malloc(N * sizeof(double));
    double *U = (double *) malloc(N * sizeof(double));
    for (i = 0; i < N; ++i) {
        U[i] = 1.0;
    }
    cblas_dsymv(CblasRowMajor, CblasUpper, N, 1.0, A, N, U, 1, 0.0, B, 1);

    double condition_number = 0.0;

    //compute condition number using lapack
    char u = 'U';
    int info = 0;
    double inf_norm = norm_inf_mat_Y11(A, N);
    double *AF = (double *) malloc(N * N * sizeof(double));
    cblas_dcopy(N * N, A, 1, AF, 1);
    dpotrf_(&u, &N, AF, &N, &info);
    double rcond = 0.0;
    double *work = (double *) malloc(3 * N * sizeof(double));
    int *iwork = (int *) malloc(N * sizeof(int));
    dpocon_(&u, &N, AF, &N, &inf_norm, &rcond, work, iwork, &info);
    condition_number = 1.0 / rcond;
    //printf("condition number: %f\n", 1.0 / rcond);
    free(U);
    char output_name[256];
    if (method == 1) {
        sprintf(output_name, "%d_m1_%s_c%.1f.mat.bin", N, filename, condition_number);
    } else {
        sprintf(output_name, "%d_m2_a1e%d_%s_c%.1f.mat.bin", N, trans, filename, condition_number);
    }
    FILE *ptr_file = fopen(output_name, "wb");
    if (!ptr_file) {
        printf("Unable to open file!");
    } else {
        fwrite(&N, sizeof(int), 1, ptr_file);
        fwrite(A, sizeof(double), (size_t) (N * N), ptr_file);
        fwrite(B, sizeof(double), (size_t) N, ptr_file);
        fclose(ptr_file);
    }

    free(A);
    free(B);
    return 0;
}
