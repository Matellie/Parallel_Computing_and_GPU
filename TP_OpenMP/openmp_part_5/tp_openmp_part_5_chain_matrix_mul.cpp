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
#include <cmath>
#include <sys/time.h>

#define AVAL 3.14
#define BVAL 5.42
#define TOL  0.001

int main(int argc, char **argv)
{
    int Ndim = 1000, Cdim = 2, CdimStep=2;   /* A[N][P], B[P][M], C[N][M] */
	int i,j,k;
	double cval, tmp, err, errsq;
    double dN, dM, dP, mflops;


    
    // Read command line arguments.
	for ( int i = 0; i < argc; i++ ) {
	if ( ( strcmp( argv[ i ], "-N" ) == 0 )) {
		Ndim = atoi( argv[ ++i ] );
		printf( "  User N is %d\n", Ndim );
	} else if ( ( strcmp( argv[ i ], "-C" ) == 0 )) {
		CdimStep = atoi( argv[ ++i ] );
		Cdim = pow( 2.0, CdimStep );
		printf( "  User C is %d\n", Cdim );
	} else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
		printf( "  Matrix multiplication Options:\n" );
		printf( "  -N <int>:              Size of the dimension N (by default 1000)\n" );
		printf( "  -help (-h):            print this message\n\n" );
		exit( 1 );
	}
	}
	
	double** listMatrix = (double**)malloc((Cdim + Cdim - 1)*sizeof(double*));


	for (int i = 0; i < Cdim; i++) {
		listMatrix[i] = (double *)malloc(Ndim * Ndim * sizeof(double));
		for (int j = 0; j < Ndim; j++)
			for (int k = 0; k < Ndim; k++) {
				if (i % 2 == 0)
					listMatrix[i][j * Ndim + k] = AVAL;
				else
					listMatrix[i][j * Ndim + k] = BVAL;
			}
	}

	for(int i=Cdim; i<Cdim+Cdim-1;i++){
		listMatrix[i] = (double *)malloc(Ndim*Ndim*sizeof(double));
		for (j=0; j<Ndim; j++)
			for (k=0; k<Ndim; k++)
				listMatrix[i][j * Ndim + k] = 0.0;
	}

	/* Do the matrix product */
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

	int last = Cdim;

	int cI = 0, cJ = 0, cK = 0;

	for (int p=0; p<pow(2.0, CdimStep + 1)-2; p+=2){
		#pragma omp parallel for collapse(2)
		for (int i=0; i<Ndim; i++){
			cI++;
			for (int j=0; j<Ndim; j++){
				cJ++;
				tmp = 0.0;
				#pragma omp simd reduction(+:tmp)
				for(int k=0;k<Ndim;k++){
					cK++;
					/* C(i,j) = sum(over k) A(i,k) * B(k,j) */
					tmp += listMatrix[p][i*Ndim+k] * listMatrix[p+1][k*Ndim+j];
					printf("Ndim = %d, i = %d, j = %d, k = %d \n",Ndim, i, j, k);
					//printf("ci = %d, cj = %d, ck = %d \n", cI, cJ, cK);
				}
				listMatrix[last][i*Ndim+j] = tmp;
			}
		}
		
		last++;
	}
	
	
	/* Check the answer */


    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
	printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Ndim, Ndim, time);
	for(int i = 0; i<Ndim; i++){
		for(int j = 0; j<Ndim; j++){
			printf("%lf | ", listMatrix[last-1][i * Ndim + j]);
		}
		printf("\n");
	}

	/*
      dN = (double)Ndim;
      dM = (double)Ndim;
      dP = (double)Ndim;
      mflops = 2.0 * dN * dN * dN/(1000000.0* time);
 
	printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Ndim, Ndim, mflops);

	cval = Ndim * AVAL * BVAL;
	errsq = 0.0;
	for (i=0; i<Ndim; i++){
		for (j=0; j<Ndim; j++){
			err = *(C+i*Ndim+j) - cval;
		    errsq += err * err;
		}
	}

	if (errsq > TOL) 
		printf("\n Errors in multiplication: %f",errsq);
	else
		printf("\n Hey, it worked");

	printf("\n all done \n");*/
}
