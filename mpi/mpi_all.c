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

    char *files[] = {
            //"1200_m1_t2_hi_c363417000.7.mat.bin",
            //"1200_m1_t7_hi_c536234414.4.mat.bin",
            //"2340_m2_a1e15_to_c3291.0.mat.bin",
            //"5900_m1_t1_to_c3205771604.1.mat.bin",
            //"1200_m1_t3_hi_c70487569.2.mat.bin",
            //"2340_m1_t1_to_c1446918556.5.mat.bin",
            //"2340_m2_a1e5_to_c8.0.mat.bin",
            //"5900_m2_a1e15_to_c4198.3.mat.bin",
            "494_bus.mtx.bin",
            //"bcsstk13.mtx.bin",
            //"bcsstk27.mtx.bin",
            //"ex9.mtx.bin",
            //"msc01050.mtx.bin",
            //"bcsstk15.mtx.bin",
            //"cage9.mtx.bin",
            //"gyro.mtx.bin",
            "tomography.mtx.bin"
    };
    int num_of_files = sizeof(files) / sizeof(char *);

    char *file_dir = NULL;
    int option = 0, omp_threads = 1;
    while ((option = getopt(argc, argv, "f:t:")) != -1) {
        switch (option) {
            case 'f':
                file_dir = optarg;
                break;
            case 't':
                omp_threads = atoi(optarg);
                break;
            default:
                printf("Usage: mpi_mexp -f string -t num_threads \n");
                return 0;
        }
    }
    if (file_dir == NULL) {
        printf("Usage: mpi_mexp -f string -t num_threads \n");
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


    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, coords[0], rank, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, coords[1], rank, &colComm);

    int row_rank, col_rank;
    MPI_Comm_rank(colComm, &col_rank);
    MPI_Comm_rank(rowComm, &row_rank);


    char filename[1024];
    while (num_of_files-- > 0) {

        memset(filename, 0, strlen(filename));
        strcpy(filename, file_dir);
        strcat(filename, files[num_of_files]);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
            printf("\n%s\n", filename);
        MPI_File matFile;
        rc = MPI_File_open(cartComm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &matFile);
        if (rc && (rank == 0)) {
            printf("Unable to open file %s\n", filename);
            fflush(stdout);
        }

        MPI_Status status;
        MPI_File_read(matFile, &n, 1, MPI_INT, &status);

        int blockSize = 0, lastBlock = 0;
        blockSize = (n + qProcs) / qProcs;  /* number of rows in _block_ */
        lastBlock = n + 1 - (qProcs - 1) * blockSize;

        if (rank == 0) {
            printf("size: %d, %d ,%d\n", n, blockSize, lastBlock);
        }

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
            memset(LocalA + (lastBlock - 1) * blockSize, 0,
                   (blockSize + 1 - lastBlock) * blockSize * sizeof(double));
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

        if (coords[1] == qProcs - 1) {
            cblas_dcopy(blockSize, loc_vec_var, 1, LocalA + lastBlock - 1, blockSize);
        }


        double mpi_start = MPI_Wtime();
        //compute inf norm of matA
        for (i = 0; i < (col_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
            loc_vec_var[i] = cblas_dasum((row_rank == q_root ? lastBlock - 1 : blockSize), LocalA + i * blockSize,
                                         1);
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
            cblas_dscal((row_rank == q_root ? lastBlock - 1 : blockSize), -1 / loc_d_var, LocalA + i * blockSize,
                        1);
        }
        if (row_rank == q_root) {
            cblas_dscal((col_rank == q_root ? lastBlock - 1 : blockSize), 1 / loc_d_var, LocalA + lastBlock - 1,
                        blockSize);
        }
        if (row_rank == col_rank) {
            for (i = 0; i < (row_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
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

        double *row_mat = (double *) malloc(qProcs * blockSize * blockSize * sizeof(double));

        while (true) {
            //norm mat
            //compute inf norm of matA
            for (i = 0; i < (col_rank == q_root ? lastBlock - 1 : blockSize); ++i) {
                loc_vec_var[i] = cblas_dasum((row_rank == q_root ? lastBlock - 1 : blockSize),
                                             LocalA + i * blockSize, 1);
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

                size_t index = cblas_idamax((col_rank == q_root ? lastBlock - 1 : blockSize),
                                            LocalA + lastBlock - 1,
                                            blockSize);
                double norm2b = 0;
                MPI_Reduce(&LocalA[index * blockSize + lastBlock - 1], &norm2b, 1, MPI_DOUBLE, MPI_MAX, q_root,
                           colComm);

                if (col_rank == q_root) {
                    loopBreak = (loc_d_var / norm2b < tolerance) ? 1 : 0;
                }

            }
            MPI_Bcast(&loopBreak, 1, MPI_INT, world_root, MPI_COMM_WORLD);
            if (loopBreak > 0)
                break;

            iteration++;

            //gather loc mat
            MPI_Allgather(LocalA, blockSize * blockSize, MPI_DOUBLE, col_mat, blockSize * blockSize, MPI_DOUBLE,
                          colComm);
            MPI_Allgather(LocalA, blockSize * blockSize, MPI_DOUBLE, row_mat, blockSize * blockSize, MPI_DOUBLE,
                          rowComm);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blockSize, blockSize, blockSize,
                        1, row_mat, blockSize, col_mat, blockSize, 0.0, LocalA, blockSize);
            for (i = 1; i < qProcs; ++i) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blockSize, blockSize, blockSize,
                            1, row_mat + i * (blockSize * blockSize), blockSize,
                            col_mat + i * (blockSize * blockSize), blockSize, 1, LocalA, blockSize);
            }
        }

        free(row_mat);
        free(col_mat);
        free(loc_vec_var);


        if (rank == world_root) {
            printf("NumProc: %d\n", numTasks);
            printf("mpi_mexp_iter: %d\t", iteration);
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
        free(LocalA);

        /*
        * MPI_CG
        */
        rc = MPI_File_open(cartComm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &matFile);
        if (rc && (rank == 0)) {
            printf("Unable to open file %s\n", filename);
            fflush(stdout);
        }

        MPI_File_read(matFile, &n, 1, MPI_INT, &status);
        blockSize = (n + qProcs - 1) / qProcs;  /* number of rows in _block_ */
        lastBlock = n - (qProcs - 1) * blockSize;

        LocalA = (double *) malloc(blockSize * blockSize * sizeof(double));

        MPI_Type_vector(blockSize, blockSize, n, MPI_DOUBLE, &readFileType);
        MPI_Type_commit(&readFileType);
        offset = (MPI_Offset) (1 * sizeof(int) +
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
        MPI_File_read_at_all(matFile, offset, LocalB, coords[1] == qProcs - 1 ? lastBlock : blockSize, MPI_DOUBLE,
                             &status);

        MPI_File_close(&matFile);
        mpi_start = MPI_Wtime();

        double alpha, beta, rho_new = 0.0, rho_old = 0.0;
        iteration = 0;
        double *loc_Vr = LocalB; //(double *) malloc(blockSize * sizeof(double));
        double *loc_Vp = (double *) malloc(blockSize * sizeof(double));
        double *loc_Vw = (double *) malloc(blockSize * sizeof(double));
        double *loc_Vx = (double *) malloc(blockSize * sizeof(double));


        //localr <- localB
        memset(loc_Vx, 0, blockSize * sizeof(double));
        //cblas_dcopy(blockSize, LocalB, 1, loc_Vr, 1);
        double b_norm2 = 0;
        if (col_rank == qProcs - 1) {
            double loc_b_dot = cblas_ddot((row_rank == qProcs - 1 ? lastBlock : blockSize), loc_Vr, 1, loc_Vr, 1);
            MPI_Reduce(&loc_b_dot, &rho_new, 1, MPI_DOUBLE, MPI_SUM, qProcs - 1, rowComm);
            b_norm2 = sqrt(rho_new);
        }

        loopBreak = 0;
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
        free(LocalA);
        free(loc_Vp);
        free(loc_Vr);
        free(loc_Vw);

        if (col_rank == qProcs - 1) {
            for (rc = 0; rc < (row_rank == qProcs - 1 ? lastBlock : blockSize); ++rc) {
                if (fabs(loc_Vx[rc] - 1) > 1e-3) {
                    printf("MPI_CG_ERR %f \n", loc_Vx[rc]);
                    break;
                }
            }
        }
        free(loc_Vx);
    }

    MPI_Comm_free(&cartComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);

    MPI_Finalize();
}
