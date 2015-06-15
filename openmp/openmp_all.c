#include <stdio.h>
#include <getopt.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#ifdef __APPLE__

#include <cblas.h>

#define set_num_threads(x) openblas_set_num_threads(x)
#define get_num_threads() openblas_get_num_threads()
#else
#include <mkl.h>
#define set_num_threads(x) mkl_set_num_threads(x)
#define get_num_threads() mkl_get_num_threads()
#endif


double norm_inf_mat_Y11(double *X, int n, int extra);

int main(int argc, char **argv) {

    int n, i, j;
    char *filename = NULL;
    double tolerance = 1e-8;
    int option = 0;
    int threads_num = 1;
    while ((option = getopt(argc, argv, "t:f:")) != -1) {
        switch (option) {
            case 'f':
                filename = optarg;
                break;
            case 't':
                threads_num = atoi(optarg);
                break;
            default:
                printf("Usage: mexp -f string \n");
                return 0;
        }
    }
    if (filename == NULL) {
        printf("Usage: mexp -t num_threads -f string \n");
        return 0;
    }

    double *A = NULL, *R = NULL, *B = NULL, *W = NULL;
    double *x, *r, *p, *w;
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Unable to open file!");
        return 0;
    }

    fread(&n, sizeof(int), 1, file);
    A = (double *) malloc((n + 1) * (n + 1) * sizeof(double));
    R = (double *) malloc((n + 1) * (n + 1) * sizeof(double));
    B = (double *) malloc((n + 1) * sizeof(double));
    W = (double *) malloc(n * sizeof(double));
    x = (double *) malloc(n * sizeof(double));
    r = (double *) malloc(n * sizeof(double));
    p = (double *) malloc(n * sizeof(double));
    w = (double *) malloc(n * sizeof(double));

    printf("set_threads_num:%d\t", threads_num++);

    fseek(file, 0, SEEK_SET);
    fread(&n, sizeof(int), 1, file);

    //fseek(file, sizeof(int), SEEK_SET);
    memset(A, 0, sizeof(double) * (n + 1) * (n + 1));
    memset(R, 0, sizeof(double) * (n + 1) * (n + 1));
    for (i = 0; i < n; ++i) {
        fread(A + i * (n + 1), sizeof(double), (size_t) n, file);
    }
    memset(B, 0, sizeof(double) * (n + 1));
    fread(B, sizeof(double), (size_t) n, file);

    set_num_threads(threads_num);
    omp_set_num_threads(threads_num);

    cblas_dcopy(n, B, 1, A + n, n + 1);

    double omp_time = omp_get_wtime();
    double lambda = norm_inf_mat_Y11(A, n, 1);

    //init Y matrix
    for (i = 0; i < n; ++i) {
        double *A_T = A + i * (n + 1);
#pragma omp parallel for
        for (j = 0; j < n; ++j) {
            A_T[j] = -A_T[j] / lambda;
        }
    }
#pragma omp parallel for
    for (i = 0; i < (n + 1) * (n + 1); i += n + 2) {
        A[i] += 1;
    }
#pragma omp parallel for
    for (i = n; i < (n + 1) * n; i += n + 1) {
        A[i] /= lambda;
    }
#pragma omp parallel for
    for (i = 0; i < n; ++i) {
        double *A_T = A + n * (n + 1);
        A_T[i] = 0.0;
    }
    A[(n + 1) * (n + 1)] = 1.0;


    //loop
    double a, b, *T;
    size_t index;
    int itern = 0;
    while (1) {
        a = norm_inf_mat_Y11(A, n, 1);
        index = cblas_idamax(n, A + n, n + 1);
        b = A[(index + 1) * (n + 1) - 1];
        if (a / b < tolerance)
            break;
        else {
            //R = A*A
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n + 1, n + 1,
                        n + 1, 1.0, A, n + 1, A, n + 1, 0.0, R, n + 1);
            T = A;
            A = R;
            R = T;
            itern++;
        }
    }

    omp_time = omp_get_wtime() - omp_time;

    cblas_dcopy(n, A + n, n + 1, B, 1);

    /*
    * check result
    */
    for (i = 0; i < n; ++i) {
        if (fabs(B[i] - 1) > 1e-3) {
            printf("MEXP ERR %f\t", B[i]);
            break;
        }
    }
    printf("mexp_iter:%d\t", itern);
    printf("omp_time:%f\t", omp_time);


    fseek(file, 0, SEEK_SET);
    fread(&n, sizeof(int), 1, file);

    fread(A, sizeof(double), (size_t) (n * n), file);
    memset(B, 0, sizeof(double) * (n + 1));
    fread(B, sizeof(double), (size_t) n, file);
#pragma omp parallel for
    for (i = 0; i < n; ++i) {
        W[i] = 1.0;
        //B[i] /= A[i*(n+1)];
        //cblas_dscal(n,1/A[i*(n+1)],A+i*n,1);
    }

    omp_time = omp_get_wtime();
    int iteration = 0;

    double alpha, beta, rho_new = 0.0, rho_old;
    memset(x, 0, sizeof(double) * n);
    cblas_dcopy(n, B, 1, r, 1); //r <- B

    rho_new = cblas_ddot(n, r, 1, r, 1);
    double w_norm2 = sqrt(rho_new);
    while (1) {
        if (sqrt(rho_new) / w_norm2 < tolerance) {
            break;
        }
        iteration++;
        if (iteration == 1) {
            cblas_dcopy(n, r, 1, p, 1); //p <- r
        } else {
            beta = rho_new / rho_old;
            cblas_dscal(n, beta, p, 1); // p <- beta * p
            cblas_daxpy(n, 1.0, r, 1, p, 1); // p <- r + pcd
        }
        cblas_dsymv(CblasRowMajor, CblasUpper, n, 1.0, A, n, p, 1, 0.0, w, 1);
        alpha = rho_new / cblas_ddot(n, p, 1, w, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, w, 1, r, 1);  // r = r - alpha* w
        rho_old = rho_new;
        rho_new = cblas_ddot(n, r, 1, r, 1); //rho_new = r' dot r;
    }

    printf("cg_it: %d\t", iteration);
    printf("cg_omp_time: %f", omp_get_wtime() - omp_time);

    /*
    * check result
    */
    for (i = 0; i < n; ++i) {
        if (fabs(x[i] - 1) > 1e-3) {
            printf(" CG ERR %f\t", x[i]);
            break;
        }
    }
    printf("\n");

    free(A);
    free(B);
    free(W);
    free(R);
    free(x);
    free(r);
    free(p);
    free(w);
    fclose(file);
}

double norm_inf_mat_Y11(double *X, int n, int extra) {
    double lambda = 0.0, lambda_t;
    int i;
    for (i = 0; i < n; ++i) {
        lambda_t = cblas_dasum(n, X + (i * (n + extra)), 1);
        if (lambda_t > lambda)
            lambda = lambda_t;
    }
    return lambda;
}
