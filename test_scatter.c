#include <stdio.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MATSIZE 4096
#define P_ROW 2
#define P_COL 8
#define LMATSIZE_ROW MATSIZE/P_ROW
#define LMATSIZE_COL MATSIZE/P_COL
#define A(i, k) A[i*MATSIZE+k]
#define A_local(j,i) A_local[j*LMATSIZE_COL+i]
#define B_local(j,i) B_local[j*LMATSIZE_ROW+i]
#define C_local(j,i) C_local[j*LMATSIZE_ROW+i]
#define B(k, j) B[k*MATSIZE+j]
#define C(i, j) C[i*MATSIZE+j]
#define Cans(i, j) Cans[i*n+j]

void mmul1(float *A, float *B, float *C, int n)
{
    int i, j, k;
//#pragma omp parallel for private(j,k)
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++)
            C(i,j) = 0;

        for (k=0; k<n; k++) {
            for (j=0; j<n; j++) {
                C(i,j) += A(i,k)*B(k,j);
            }
        }
    }
}

int compute_diff(float *C, float *Cans, int n)
{
    int cnt = 0;
    int i, j;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (fabs(C(i,j) - Cans(i,j)) > 10e-4)
                cnt++;
        }
    }
    return cnt;
}

int main(int argc, char *argv[]) {
    int pnum, pid;
    
    double elapsed_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int scounts_b[pnum];
    int disp_b[pnum];
    int scounts_a[pnum];
    int disp_a[pnum];
    int sc_b = MATSIZE/P_ROW; /* Send Count for matrix B */
    int sc_a = MATSIZE/P_COL; /* Send Count for matrix A */
    int stride_b = MATSIZE * (MATSIZE/P_COL);
    int stride_a = MATSIZE * (MATSIZE/P_ROW);
    unsigned short seed[3];
    seed[0] = 0; seed[1] = 1; seed[2] = 2;
    float* A;
    float* B;
    float* C = (float*)malloc(sizeof(float) * MATSIZE * MATSIZE);
    memset( C, 0, sizeof(float) * MATSIZE * MATSIZE );
    /* Testing scatter */
    if (pid == 0) {
        A = (float*)malloc(sizeof(float) * MATSIZE * MATSIZE);
        B = (float*)malloc(sizeof(float) * MATSIZE * MATSIZE);
        for (unsigned int i = 0; i < MATSIZE; i++) {
            for (unsigned int j = 0; j < MATSIZE; j++) {
                A(i,j) = (float)erand48(seed);
                B(i,j) = (float)erand48(seed);
            }
        }    
    }
    for (unsigned int j = 0; j < P_COL; j++) {
        for (unsigned int i = 0; i < P_ROW; i++) {
            scounts_b[i+j*P_ROW] = sc_b;
            disp_b[i+j*P_ROW] = sc_b*i + j*stride_b;
            // printf("(%d, %d, %d)\n", scounts[i+j*pnum_row], disp[i+j*pnum_row], i+j*pnum_row);
        }
    }
    for (unsigned int j = 0; j < P_ROW; j++) {
        for (unsigned int i = 0; i < P_COL; i++) {
            scounts_a[i+j*P_COL] = sc_a;
            disp_a[i*P_ROW+j] = sc_a*i + j*stride_a;
            // printf("(%d, %d, %d)\n", scounts[i+j*pnum_row], disp[i+j*pnum_row], i+j*pnum_row);
        }
    }
    /* Timing Start */
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -1*MPI_Wtime();
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
    if (row_rank == 0) {
        printf("WORLD RANK/SIZE: %d/%d \t COL0 RANK/SIZE: %d/%d\n",
                     pid, pnum, col0_rank, col0_size);
    }
    /* Create receive buffer */
    float* A_local = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL);
    float* B_local = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL);
    float* C_local = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW);
    float* Cans = (float*)malloc(sizeof(float) * MATSIZE * MATSIZE);
    float* C_local_reduced = (float*)malloc(sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW);
    memset( C_local_reduced, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW );
    memset( A_local, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL );
    memset( B_local, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_COL );
    memset( C_local, 0, sizeof(float) * LMATSIZE_ROW * LMATSIZE_ROW );
    memset( Cans, 0, sizeof(float) * MATSIZE * MATSIZE );
    

	
    /* Scatter out sub-matrices for A, which only needs to be done once,
     * since only matrix B will be redistribute.
     */

    /*
    *    Matrix A only needs to be send once
    */
    for (unsigned int j = 0; j < LMATSIZE_ROW; j++) {
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
    for (unsigned int i = 0; i < P_COL; i++) {
        row_scs[i] = LMATSIZE_ROW;
        row_disp[i] = i*LMATSIZE_COL*MATSIZE;
    }
    for (unsigned int q = 0; q < row_size; q++) {
        if (row_rank == 0) {
            for (unsigned int j = 0; j < LMATSIZE_COL; j++) {
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
        for (unsigned int i=0; i<LMATSIZE_ROW; i++) {
            for (unsigned int j=0; j<LMATSIZE_ROW; j++)
                C_local(i,j) = 0;
            
            for (unsigned int k=0; k<LMATSIZE_COL; k++) {
                for (unsigned int j=0; j<LMATSIZE_ROW; j++) {
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
        for (unsigned int i = 0; i < P_ROW; i++) {
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
            for (unsigned int j = 0; j < LMATSIZE_ROW; j++) {
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
    /* Timing End */
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();
    
    
    
//    if (row_rank == 7 && col0_rank == 1) {
//        for (unsigned int j = 0; j < LMATSIZE_COL; j++) {
//            for (unsigned int i = 0; i < LMATSIZE_ROW; i++) {
//                printf("%.2f ", B_local(j,i));
//            }
//            printf("\n");
//        }
//        printf("WORLD RANK/SIZE: %d/%d \t COL0 RANK/SIZE: %d/%d ROW RANK/SIZE %d/%d\n",
//                             pid, pnum, col0_rank, col0_size, row_rank, row_size);
//    }
    
    
    if (pid == 0) {
          mmul1( A, B, Cans, MATSIZE );
          printf("Diff: %d", compute_diff(C, Cans, MATSIZE));
//        for (unsigned int j = 0; j < LMATSIZE_COL; j++) {
//            for (unsigned int i = 0; i < LMATSIZE_ROW; i++) {
//                printf("%.2f ", B(j,i));
//            }
//            printf("\n");
//        }
//        for (unsigned int j = 0; j < MATSIZE; j++) {
//            for (unsigned int i = 0; i < MATSIZE; i++) {
//                printf("%.2f ", C(j,i));
//            }
//            printf("\n");
//        }
        printf("\nelapsed_time: %fs\n", elapsed_time);
    }
	
//    printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d\n",
//                pid, pnum, row_rank, row_size);
    
    free( A_local );
    free( B_local );
    free( C_local );
    free( Cans );
    free( C_local_reduced );
    
    MPI_Comm_free(&col0_comm);
    MPI_Comm_free(&row_comm);
    MPI_Finalize();
}