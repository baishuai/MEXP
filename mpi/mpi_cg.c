#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <libiomp/omp.h>
#include <string.h>
#include <stdbool.h>

#ifdef __APPLE__

#include <cblas.h>

#define set_num_threads(x) openblas_set_num_threads(x)
#define get_num_threads() openblas_get_num_threads()
#else
#include <mkl.h>
#define set_num_threads(x) mkl_set_num_threads(x)
#define get_num_threads() mkl_get_num_threads()
#endif


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
                printf("Usage: mpi_cg -f string -t num_threads \n");
                return 0;
        }
    }
    if (filename == NULL) {
        printf("Usage: mpi_cg -f string -t num_threads \n");
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

    int n, dims[2] = {qProcs, qProcs}, periods[2] = {0, 0}, reorder = 0, coords[2], rc;
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
    int blockSize = (n + qProcs - 1) / qProcs;  /* number of rows in _block_ */
    int lastBlock = n - (qProcs - 1) * blockSize;

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
    if (coords[0] == qProcs - 1) {
        memset(LocalA + lastBlock * blockSize, 0, (blockSize - lastBlock) * blockSize * sizeof(double));
    }
    if (coords[1] == qProcs - 1) {
        int kk = 0;
        for (kk = 0; kk < blockSize; ++kk) {
            memset(LocalA + kk * blockSize + lastBlock, 0, (blockSize - lastBlock) * sizeof(double));
        }
    }

    double *LocalB = (double *) malloc(blockSize * sizeof(double));
    memset(LocalB, 0, blockSize * sizeof(double));
    offset = (MPI_Offset) (1 * sizeof(int) + sizeof(double) * n * n);
    MPI_File_set_view(matFile, offset, MPI_DOUBLE, MPI_DOUBLE,
                      "native", MPI_INFO_NULL);
    offset = coords[1] * blockSize;
    MPI_File_read_at_all(matFile, offset, LocalB, coords[1] == qProcs - 1 ? lastBlock : blockSize, MPI_DOUBLE, &status);

    MPI_File_close(&matFile);
    double mpi_start = MPI_Wtime();
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, coords[0], rank, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, coords[1], rank, &colComm);

    double tolerance = 1e-8;
    double alpha, beta, rho_new = 0.0, rho_old = 0.0;
    int iteration = 0;
    int world_root = numTasks - 1;
    double *loc_Vr = (double *) malloc(blockSize * sizeof(double));
    double *loc_Vp = (double *) malloc(blockSize * sizeof(double));
    double *loc_Vw = (double *) malloc(blockSize * sizeof(double));
    double *loc_Vx = (double *) malloc(blockSize * sizeof(double));

    int row_rank, col_rank;
    MPI_Comm_rank(colComm, &col_rank);
    MPI_Comm_rank(rowComm, &row_rank);

    //localr <- localB
    memset(loc_Vx, 0, blockSize * sizeof(double));
    cblas_dcopy(blockSize, LocalB, 1, loc_Vr, 1);
    double b_norm2 = 0;
    if (col_rank == qProcs - 1) {
        double loc_b_dot = cblas_ddot((row_rank == qProcs - 1 ? lastBlock : blockSize), loc_Vr, 1, loc_Vr, 1);
        MPI_Reduce(&loc_b_dot, &rho_new, 1, MPI_DOUBLE, MPI_SUM, qProcs - 1, rowComm);
        b_norm2 = sqrt(rho_new);
    }

    int loopBreak = 0;
    while (true) {
        if (rank == world_root) {
            loopBreak = (sqrt(rho_new) / b_norm2 < tolerance) ? 1 : 0;
        }
        MPI_Bcast(&loopBreak, 1, MPI_INT, world_root, MPI_COMM_WORLD);
        if (loopBreak > 0)
            break;

        iteration++;
        if (iteration == 1) {
            cblas_dcopy(blockSize, loc_Vr, 1, loc_Vp, 1);
        } else {
            if (rank == world_root) {
                beta = rho_new / rho_old;
            }
            MPI_Bcast(&beta, 1, MPI_DOUBLE, world_root, MPI_COMM_WORLD);
            cblas_dscal((row_rank == qProcs - 1 ? lastBlock : blockSize), beta, loc_Vp, 1);
            cblas_daxpy((row_rank == qProcs - 1 ? lastBlock : blockSize), 1.0, loc_Vr, 1, loc_Vp, 1);
        }

        cblas_dgemv(CblasRowMajor, CblasNoTrans, (col_rank == qProcs - 1 ? lastBlock : blockSize),
                    (row_rank == qProcs - 1 ? lastBlock : blockSize),
                    1.0, LocalA, blockSize, loc_Vp, 1, 0.0, loc_Vw, 1);

        if (col_rank == row_rank) {
            MPI_Reduce(MPI_IN_PLACE, loc_Vw, blockSize, MPI_DOUBLE, MPI_SUM, col_rank, rowComm);
        } else {
            MPI_Reduce(loc_Vw, loc_Vw, blockSize, MPI_DOUBLE, MPI_SUM, col_rank, rowComm);
        }

        MPI_Bcast(loc_Vw, (row_rank == qProcs - 1 ? lastBlock : blockSize), MPI_DOUBLE, row_rank, colComm);

        if (col_rank == qProcs - 1) {
            double loc_pw_dot = cblas_ddot((row_rank == qProcs - 1 ? lastBlock : blockSize), loc_Vp, 1, loc_Vw,
                                           1), glo_pw_dot;
            MPI_Reduce(&loc_pw_dot, &glo_pw_dot, 1, MPI_DOUBLE, MPI_SUM, qProcs - 1, rowComm);
            if (rank == world_root) {
                alpha = rho_new / glo_pw_dot;
            }
        }
        MPI_Bcast(&alpha, 1, MPI_DOUBLE, world_root, MPI_COMM_WORLD);

        cblas_daxpy((row_rank == qProcs - 1 ? lastBlock : blockSize), -alpha, loc_Vw, 1, loc_Vr, 1);

        if (coords[0] == qProcs - 1) {
            rho_old = rho_new;
            cblas_daxpy((row_rank == qProcs - 1 ? lastBlock : blockSize), alpha, loc_Vp, 1, loc_Vx, 1);
            double loc_r_dot = cblas_ddot(blockSize, loc_Vr, 1, loc_Vr, 1);

            MPI_Reduce(&loc_r_dot, &rho_new, 1, MPI_DOUBLE, MPI_SUM, qProcs - 1, rowComm);
        }
    }

    if (rank == world_root) {
        printf("mpi_cg_it: %d\t", iteration);
        printf("mpi_cg_time: %f\n", MPI_Wtime() - mpi_start);

    }
    MPI_Comm_free(&cartComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);

    if (col_rank == qProcs - 1) {
        for (rc = 0; rc < (row_rank == qProcs - 1 ? lastBlock : blockSize); ++rc) {
            if (fabs(loc_Vx[rc] - 1) > 1e-3) {
                printf("MPI_CG_ERR %f \n", loc_Vx[rc]);
                break;
            }
        }
    }
    MPI_Finalize();
}
