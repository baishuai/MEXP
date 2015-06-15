#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#ifdef __APPLE__

#include <cblas.h>
#include <libiomp/omp.h>

#define set_num_threads(x) openblas_set_num_threads(x)
#define get_num_threads() openblas_get_num_threads()
#else
#include <mkl.h>
#include <omp.h>
#define set_num_threads(x) mkl_set_num_threads(x)
#define get_num_threads() mkl_get_num_threads()
#endif

double norm_inf_mat_Y11(double *X, int n);

int main(int argc, char **argv) {

    char *filename = NULL;
    int option = 0, omp_threads = 1;
    while ((option = getopt(argc, argv, "f:t:")) != -1) {
        switch (option) {
            case 'f':
                filename = optarg;
                break;
            case 't':
                omp_threads = atoi(optarg);
                break;
            default:
                printf("Usage: mpi_mexp -f string \n");
                return 0;
        }
    }
    if (filename == NULL) {
        printf("Usage: mpi_mexp -f string \n");
        return 0;
    }

    set_num_threads(omp_threads);
    omp_set_num_threads(omp_threads);

    int numTasks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int qProcs = (int) sqrt(numTasks);
    if (qProcs - sqrt(numTasks) != 0) {
        if (rank == 0)
            printf("np must be a square number\n");
        MPI_Finalize();
        return 0;
    }

    int q_root = qProcs - 1;
    int world_root = numTasks - 1;
    int n, dims[2] = {qProcs, qProcs}, periods[2] = {0, 0}, reorder = 0, coords[2], rc, i;
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartComm);
    MPI_Comm_rank(cartComm, &rank);
    MPI_Cart_coords(cartComm, rank, 2, coords);

    MPI_File matFile;
    rc = MPI_File_open(cartComm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &matFile);
    if (rc && (rank == 0)) {
        printf("Unable to open file %s\n", filename);
        fflush(stdout);
    }

    MPI_Status status;
    MPI_File_read(matFile, &n, 1, MPI_INT, &status);
    int blockSize = (n + qProcs) / qProcs;  /* number of rows in _block_ */
    int lastBlock = n + 1 - (qProcs - 1) * blockSize;

    double *LocalA = (double *) malloc(blockSize * blockSize * sizeof(double));

    MPI_Datatype readFileType;
    MPI_Type_vector(blockSize, blockSize, n, MPI_DOUBLE, &readFileType);
    MPI_Type_commit(&readFileType);
    MPI_Offset offset = (MPI_Offset) (1 * sizeof(int) +
                                      sizeof(double) * (blockSize * coords[1] + blockSize * n * coords[0]));
    MPI_File_set_view(matFile, offset, MPI_DOUBLE, readFileType,
                      "native", MPI_INFO_NULL);

    MPI_File_read_at(matFile, 0, LocalA, blockSize * blockSize, MPI_DOUBLE, &status);
    MPI_Type_free(&readFileType);
    if (coords[0] == q_root) {
        memset(LocalA + (lastBlock - 1) * blockSize, 0, (blockSize + 1 - lastBlock) * blockSize * sizeof(double));
    }
    if (coords[1] == q_root) {
        int kk = 0;
        for (kk = 0; kk < blockSize; ++kk) {
            memset(LocalA + kk * blockSize + lastBlock - 1, 0, (blockSize + 1 - lastBlock) * sizeof(double));
        }
    }

    double loc_d_var;
    double *loc_vec_var = (double *) malloc(blockSize * sizeof(double));
    memset(loc_vec_var, 0, blockSize * sizeof(double));
    offset = (MPI_Offset) (1 * sizeof(int) + sizeof(double) * n * n);
    MPI_File_set_view(matFile, offset, MPI_DOUBLE, MPI_DOUBLE,
                      "native", MPI_INFO_NULL);
    offset = coords[0] * blockSize;
    MPI_File_read_at_all(matFile, offset, loc_vec_var, coords[0] == qProcs - 1 ? lastBlock : blockSize, MPI_DOUBLE,
                         &status);

    MPI_File_close(&matFile);


    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, coords[0], rank, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, coords[1], rank, &colComm);

    int row_rank, col_rank;
    MPI_Comm_rank(colComm, &col_rank);
    MPI_Comm_rank(rowComm, &row_rank);


    if (coords[1] == qProcs - 1) {
        cblas_dcopy(blockSize, loc_vec_var, 1, LocalA + lastBlock - 1, blockSize);
    }


    double mpi_start = MPI_Wtime();
    //compute inf norm of matA
    for (i = 0; i < (col_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
        loc_vec_var[i] = cblas_dasum((row_rank == q_root ? lastBlock - 1 : blockSize), LocalA + i * blockSize, 1);
    }

    if (row_rank == q_root) {
        MPI_Reduce(MPI_IN_PLACE, loc_vec_var, blockSize, MPI_DOUBLE, MPI_SUM, q_root, rowComm);
    } else {
        MPI_Reduce(loc_vec_var, loc_vec_var, blockSize, MPI_DOUBLE, MPI_SUM, q_root, rowComm);
    }

    if (row_rank == q_root) {
        loc_d_var = 0;
        for (i = 0; i < (col_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
            if (loc_vec_var[i] > loc_d_var)
                loc_d_var = loc_vec_var[i];
        }
        if (col_rank == q_root) {
            MPI_Reduce(MPI_IN_PLACE, &loc_d_var, 1, MPI_DOUBLE, MPI_MAX, q_root, colComm);
        } else {
            MPI_Reduce(&loc_d_var, &loc_d_var, 1, MPI_DOUBLE, MPI_MAX, q_root, colComm);
        }
    }

    MPI_Bcast(&loc_d_var, 1, MPI_DOUBLE, world_root, MPI_COMM_WORLD);

    //init Y martix

    for (i = 0; i < (col_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
        cblas_dscal((row_rank == q_root ? lastBlock - 1 : blockSize), -1 / loc_d_var, LocalA + i * blockSize, 1);
    }
    if (row_rank == q_root) {
        cblas_dscal((col_rank == q_root ? lastBlock - 1 : blockSize), 1 / loc_d_var, LocalA + lastBlock - 1, blockSize);

    }
    if (row_rank == col_rank) {
        for (i = 0; i < (row_rank == q_root ? lastBlock-1 : blockSize); ++i) {
            LocalA[i * blockSize + i] += 1;
        }
    }
    if (rank == world_root) {
        LocalA[(lastBlock - 1) * (blockSize + 1)] = 1;
    }

    //loop
    double tolerance = 1e-8;
    int loopBreak = 0, iteration = 0;
    double *col_mat = (double *) malloc(qProcs * blockSize * blockSize * sizeof(double));
    memset(col_mat, 0, qProcs * blockSize * blockSize * sizeof(double));

    double *row_mat = (double *) malloc(qProcs * blockSize * blockSize * sizeof(double));
    memset(row_mat, 0, qProcs * blockSize * blockSize * sizeof(double));

    while (true) {
        //norm mat
        //compute inf norm of matA
        for (i = 0; i < (col_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
            loc_vec_var[i] = cblas_dasum((row_rank == q_root ? lastBlock - 1 : blockSize), LocalA + i * blockSize, 1);
        }

        if (row_rank == q_root) {
            MPI_Reduce(MPI_IN_PLACE, loc_vec_var, blockSize, MPI_DOUBLE, MPI_SUM, q_root, rowComm);
        } else {
            MPI_Reduce(loc_vec_var, loc_vec_var, blockSize, MPI_DOUBLE, MPI_SUM, q_root, rowComm);
        }

        if (row_rank == q_root) {
            loc_d_var = 0;
            for (i = 0; i < (col_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
                if (loc_vec_var[i] > loc_d_var)
                    loc_d_var = loc_vec_var[i];
            }
            if (col_rank == q_root) {
                MPI_Reduce(MPI_IN_PLACE, &loc_d_var, 1, MPI_DOUBLE, MPI_MAX, q_root, colComm);
            } else {
                MPI_Reduce(&loc_d_var, &loc_d_var, 1, MPI_DOUBLE, MPI_MAX, q_root, colComm);
            }

            size_t index = cblas_idamax((col_rank == q_root ? lastBlock - 1 : blockSize), LocalA + lastBlock - 1,
                                        blockSize);
            double norm2b;
            MPI_Reduce(&LocalA[index * blockSize + lastBlock - 1], &norm2b, 1, MPI_DOUBLE, MPI_MAX, q_root, colComm);

            if (col_rank == q_root) {
                printf("normM:%.4f, normB:%.4f\n", loc_d_var, norm2b);
                loopBreak = (loc_d_var / norm2b < tolerance) ? 1 : 0;
            }

        }
        MPI_Bcast(&loopBreak, 1, MPI_INT, world_root, MPI_COMM_WORLD);
        if (loopBreak > 0)
            break;

        iteration++;

        //gather loc mat
        MPI_Allgather(LocalA, blockSize * blockSize, MPI_DOUBLE, col_mat, blockSize * blockSize, MPI_DOUBLE, colComm);
        MPI_Allgather(LocalA, blockSize * blockSize, MPI_DOUBLE, row_mat, blockSize * blockSize, MPI_DOUBLE, rowComm);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blockSize, blockSize, blockSize,
                    1, row_mat, blockSize, col_mat, blockSize, 0.0, LocalA, blockSize);
        for (i = 1; i < qProcs; ++i) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blockSize, blockSize, blockSize,
                        1, row_mat + i * (blockSize * blockSize), blockSize, col_mat + i * (blockSize * blockSize),
                        blockSize, 1, LocalA, blockSize);
        }
    }


    if (rank == world_root) {
        printf("mpi_mexp_iter:%d \n", iteration);
        printf("mpi_mexp_time: %f\n", MPI_Wtime() - mpi_start);

    }

    if (row_rank == q_root) {
        for (i = 0; i < (row_rank == qProcs - 1 ? lastBlock : blockSize); ++i) {
            if (fabs(LocalA[i * blockSize + lastBlock - 1] - 1) > 1e-3) {
                printf("MPI_MEXP_ERR %f \n", LocalA[i * blockSize + lastBlock - 1]);
                break;
            }
        }
    }

    MPI_Comm_free(&cartComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);

    MPI_Finalize();
}

