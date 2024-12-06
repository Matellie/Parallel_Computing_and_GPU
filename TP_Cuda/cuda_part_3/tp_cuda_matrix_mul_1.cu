/*
**  PROGRAM: Matrix Multiply
**
**  PURPOSE: This is a simple matrix multiply program. 
**           It will compute the product
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**
**  HISTORY: Written by Tim Mattson, Nov 1999.
*            Modified and extended by Jonathan Rouzaud-Cornabas, Oct 2022
*/


#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
//#include <sys/time.h>
#include "win-gettimeofday.h"

#define AVAL 3.14
#define BVAL 5.42
#define TOL  0.001

__global__ void computeMatMul(double* A, double* B, double* C, int N, int M, int P, int offset) {
  	int bid = blockIdx.x;

	for (int i=bid*offset; i<(bid+1)*offset; i++) {
		if (i >= N) {
			break;
		}
		for (int j=0; j<M; ++j) {
			double tmp = 0.0;
			for(int k=0; k<P; ++k) {
				/* C(i,j) = sum(over k) A(i,k) * B(k,j) */
				tmp += A[i*P+k]*B[k*M+j];
			}
			C[i*M+j] = tmp;
		}
	}
}

int main(int argc, char **argv) {
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
	int i,j;
	double *A, *B, *C, cval, err, errsq;
    double dN, dM, dP, mflops;

    // Read command line arguments.
	for ( int i = 0; i < argc; i++ ) {
		if ( ( strcmp( argv[ i ], "-N" ) == 0 )) {
			Ndim = atoi( argv[ ++i ] );
			printf( "  User N is %d\n", Ndim );
		} else if ( ( strcmp( argv[ i ], "-M" ) == 0 )) {
			Mdim = atoi( argv[ ++i ] );
			printf( "  User M is %d\n", Mdim );
		} else if ( ( strcmp( argv[ i ], "-P" ) == 0 )) {
			Pdim = atoi( argv[ ++i ] );
			printf( "  User P is %d\n", Pdim );
		} else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
			printf( "  Matrix multiplication Options:\n" );
			printf( "  -N <int>:              Size of the dimension N (by default 1000)\n" );
			printf( "  -M <int>:              Size of the dimension M (by default 1000)\n" );
			printf( "  -P <int>:              Size of the dimension P (by default 1000)\n" );
			printf( "  -help (-h):            print this message\n\n" );
			exit( 1 );
		}
	}


	A = (double *)malloc(Ndim*Pdim*sizeof(double));
	B = (double *)malloc(Pdim*Mdim*sizeof(double));
	C = (double *)malloc(Ndim*Mdim*sizeof(double));

	/* Initialize matrices */

	for (i=0; i<Ndim; i++) {
		for (j=0; j<Pdim; j++) {
			A[i*Pdim+j] = AVAL;
		}
	}

	for (i=0; i<Pdim; i++) {
		for (j=0; j<Mdim; j++) {
			B[i*Mdim+j] = BVAL;
		}
	}

	for (i=0; i<Ndim; i++) {
		for (j=0; j<Mdim; j++) {
			C[i*Mdim+j] = 0.0;
		}
	}


	double *gpu_A;
	cudaError_t err1 = cudaMalloc(&gpu_A, Ndim*Pdim*sizeof(double));
	if (err1 != cudaSuccess) {
		printf("Err:  %s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	double *gpu_B;
	cudaError_t err2 = cudaMalloc(&gpu_B, Pdim*Mdim*sizeof(double));
	if (err2 != cudaSuccess) {
		printf("Err:  %s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	double *gpu_C;
	cudaError_t err3 = cudaMalloc(&gpu_C, Ndim*Mdim*sizeof(double));
	if (err3 != cudaSuccess) {
		printf("Err:  %s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(gpu_A, A, Ndim*Pdim*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_B, B, Pdim*Mdim*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_C, C, Ndim*Mdim*sizeof(double), cudaMemcpyHostToDevice);

	int nbBlocks = 256;
	int offset = Ndim/nbBlocks + 1;

	/* Do the matrix product */

	// Timer products.
	struct timeval begin, end;

	gettimeofday( &begin, NULL );

	// computeMatMul(int* A, int* B, int* C, int N, int M, int P)
	computeMatMul<<<nbBlocks, 1>>>(gpu_A, gpu_B, gpu_C, Ndim, Mdim, Pdim, offset);
	cudaMemcpy(C, gpu_C, Ndim * Mdim * sizeof(double), cudaMemcpyDeviceToHost);

	/* Check the answer */

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );

	printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

	dN = (double)Ndim;
	dM = (double)Mdim;
	dP = (double)Pdim;
	mflops = 2.0 * dN * dM * dP/(1000000.0 * time);
 
	printf(" N %d M %d P %d multiplication at %lf mflops\n", Ndim, Mdim, Pdim, mflops);

	cval = Pdim * AVAL * BVAL;
	errsq = 0.0;
	for (i=0; i<Ndim; i++) {
		for (j=0; j<Mdim; j++) {
			err = C[i*Mdim+j] - cval;
		    errsq += err * err;
		}
	}

	if (errsq > TOL) {
		printf("\n Errors in multiplication: %f",errsq);
	} else {
		printf("\n Hey, it worked");
	}

	printf("\n all done \n");
}
