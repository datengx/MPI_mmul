#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> #include <math.h>
#include <string.h>


#define A(i, k) A[i*MATSIZE+k]
#define A_local(j,i) A_local[j*n+i]
#define B_local(j,i) B_local[j*n+i]
#define C_local(j,i) C_local[j*n+i]
#define B(k, j) B[k*n+j]
#define C(i, j) C[i*n+j]
#define Cans(i, j) Cans[i*n+j]

void mmul(float *A, float*B, float *C, int n) {
    int pnum, pid;

    int i, j, k, l;
    double elapsed_time2;
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int lA_row = n / pnum;
    int lC_row = n / pnum;
    float* B_local;
    float* A_local = (float*)malloc( sizeof(float) * n * lA_row );
    float* C_local = (float*)malloc( sizeof(float) * n * lC_row );
    
    if (pid == 0) {
        B_local = B; // If root
    } else {
        // If not root, allocate local buffer for broadcast
        B_local = (float*)malloc( sizeof(float) * n * n );
    }
    
    MPI_Bcast( B_local,
               n * n,
               MPI_FLOAT,
               0,
               MPI_COMM_WORLD
                );
    MPI_Scatter( A,
                 n * lA_row,
                 MPI_FLOAT,
                 A_local,
                 n * lA_row,
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD
                  );
    // MPI_Barrier(MPI_COMM_WORLD);
    
    /* Matrix Multiplication */
        for (i = 0; i < lC_row; i++) {
                for (j = 0; j < n; j++)
                    C_local(i,j) = 0;

                for (k = 0; k < n; k++) {
                    for (j = 0; j < n; j++) {
                        C_local(i,j) += A_local(i,k)*B_local(k,j);
                    }
                }
        }
    /* Send result back to the root. */
    MPI_Gather( C_local,
                n * lA_row,
                MPI_FLOAT,
                C,
                n * lA_row,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD );
    MPI_Barrier(MPI_COMM_WORLD);            


    free(A_local);
    if (pid != 0) {
        free(B_local);
    }
    free(C_local);
}

void mmul2(float *A, float *B, float *C, int n)
{
    int pnum, pid;
    
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    
    unsigned int MATSIZE = n;
    unsigned int P_ROW = 4;
    unsigned int P_COL = 4;
    unsigned int LMATSIZE_ROW = MATSIZE / P_ROW;
    unsigned int LMATSIZE_COL = MATSIZE / P_COL;
    
    
    int scounts_b[pnum];
    int disp_b[pnum];
    int scounts_a[pnum];
    int disp_a[pnum];
    int sc_b = MATSIZE/P_ROW; /* Send Count for matrix B */
    int sc_a = MATSIZE/P_COL; /* Send Count for matrix A */
    int stride_b = MATSIZE * (MATSIZE/P_COL);
    int stride_a = MATSIZE * (MATSIZE/P_ROW);
    int j, k, i, q;
    
    for (j = 0; j < P_COL; j++) {
        for (i = 0; i < P_ROW; i++) {
            scounts_b[i+j*P_ROW] = sc_b;
            disp_b[i+j*P_ROW] = sc_b*i + j*stride_b;
            // printf("(%d, %d, %d)\n", scounts[i+j*pnum_row], disp[i+j*pnum_row], i+j*pnum_row);
        }
    }
    for (j = 0; j < P_ROW; j++) {
        for (i = 0; i < P_COL; i++) {
            scounts_a[i+j*P_COL] = sc_a;
            disp_a[i*P_ROW+j] = sc_a*i + j*stride_a;
            // printf("(%d, %d, %d)\n", scounts[i+j*pnum_row], disp[i+j*pnum_row], i+j*pnum_row);
        }
    }
    
    MPI_Comm row_comm;
    MPI_Comm col0_comm;
    int color = pid / P_ROW;
    MPI_Comm_split(MPI_COMM_WORLD, color, pid, &row_comm);
    int row_rank, row_size;
    int col0_rank, col0_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    int col_color = pid % P_ROW;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, pid, &col0_comm);
    MPI_Comm_rank(col0_comm, &col0_rank);
    MPI_Comm_size(col0_comm, &col0_size);
    
    
    /* Create receive buffer */
    float* A_local = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL);
    float* B_local = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL);
    float* C_local = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW);
    float* C_local_reduced = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW);
    memset( C_local_reduced, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW );
    memset( A_local, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL );
    memset( B_local, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL );
    memset( C_local, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW );
//    memset( Cans, 0, sizeof(float) * MATSIZE * MATSIZE );
    /*
        *    Matrix A only needs to be send once
        */
    for (j = 0; j < LMATSIZE_ROW; j++) {
            MPI_Scatterv(   A+j*MATSIZE, /* Send buffer */
                            scounts_a,     /* Send counts */
                            disp_a,        /* Displacement for each process */
                            MPI_FLOAT,     /* Data type */
                            A_local+j*LMATSIZE_COL,/* Local receiver pointer */
                            LMATSIZE_COL,          /* Receive size */
                            MPI_FLOAT,     /* Receive Type */
                            0,           /* Root */
                            MPI_COMM_WORLD);
    }
    int row_scs[P_COL];
    int row_disp[P_COL];
    int c_local_scs[P_ROW];
    int c_local_disp[P_ROW];
    for (i = 0; i < P_COL; i++) {
        row_scs[i] = LMATSIZE_ROW;
        row_disp[i] = i*LMATSIZE_COL*MATSIZE;
    }
    for (q = 0; q < row_size; q++) {
        if (row_rank == 0) {
            for (j = 0; j < LMATSIZE_COL; j++) {
                MPI_Scatterv(   B + j*MATSIZE + q*LMATSIZE_ROW,   /* Send buffer */
                                row_scs,     /* Send counts */
                                row_disp,        /* Displacement for each process */
                                MPI_FLOAT,     /* Data type */
                                B_local+j*LMATSIZE_ROW,/* Local receiver pointer */
                                LMATSIZE_ROW,          /* Receive size */
                                MPI_FLOAT,     /* Receive Type */
                                0,             /* Root */
                                col0_comm);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast( B_local,
                   LMATSIZE_ROW * LMATSIZE_COL,
                   MPI_FLOAT,
                   0,
                   row_comm
                    );
                    
        MPI_Barrier(MPI_COMM_WORLD);
        
        /* Local submatrix multiplication */
        //#pragma omp parallel for private(j,k)
        for (i=0; i<LMATSIZE_ROW; i++) {
            for (j=0; j<LMATSIZE_ROW; j++)
                C_local(i,j) = 0;
            
            for (k=0; k<LMATSIZE_COL; k++) {
                for (j=0; j<LMATSIZE_ROW; j++) {
                    C_local(i,j) += A_local(i,k)*B_local(k,j);
                }
            }
        }

        MPI_Reduce( C_local,
                    C_local_reduced,
                    LMATSIZE_ROW * LMATSIZE_ROW,
                    MPI_FLOAT,
                    MPI_SUM,
                    0,
                    col0_comm
                     );

        MPI_Barrier(MPI_COMM_WORLD);

                    
        for (i = 0; i < P_ROW; i++) {
            c_local_scs[i] = LMATSIZE_ROW;
            c_local_disp[i] = i*LMATSIZE_ROW * MATSIZE;
        }             
//        if (q == 2 && pid == 3) {
//            for (unsigned int p = 0; p < LMATSIZE_ROW * LMATSIZE_ROW; p++) {
//                printf("%f\n", C_local_reduced[p]);
//            }
//            for (unsigned int j = 0; j < LMATSIZE_COL; j++) {
//                 for (unsigned int i = 0; i < LMATSIZE_ROW; i++) {
//                        printf("%.2f ", B_local(j,i));
//                    }
//                printf("\n");
//            }
//        }
        MPI_Barrier(MPI_COMM_WORLD);         
        if (col0_rank == 0) {
            for (j = 0; j < LMATSIZE_ROW; j++) {
                MPI_Gatherv( C_local_reduced + j*LMATSIZE_ROW,
                             LMATSIZE_ROW,
                             MPI_FLOAT,
                             C + j*MATSIZE + q*LMATSIZE_ROW,
                             c_local_scs,
                             c_local_disp,
                             MPI_FLOAT,
                             0,
                             row_comm
                              );
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
//        printf("Finish processing col %d\n", q);
    }
}

