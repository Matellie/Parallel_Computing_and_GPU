/*

This program will numerically compute the integral of

                  4/(1+x*x) 

from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
//#include "win-gettimeofday.h"

static long nbSteps = 1000000000;
static int nbBlocks = 1024;
double step;

__global__ void computePi(double step, long nbSteps, long offset, double *sum) {
  int bix = blockIdx.x;

  for (int i = bix*offset; i < (bix+1)*offset; i++) {
    if (i < nbSteps) {
      double x = (i - 0.5) * step;
      sum[bix] += 4.0 / (1.0 + x * x);
    } else {
      sum[bix] = 0.0;
    }
  }
}


int main (int argc, char** argv) {
  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
      nbSteps = atol( argv[ ++i ] );
      printf( "  User num_steps is %ld\n", nbSteps );
    } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  Pi Options:\n" );
      printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  double pi = 0.0;

  double *cpu_sum = (double *)malloc(nbBlocks * sizeof(double));

  step = 1.0/(double) nbSteps;

  // Timer products.
  struct timeval begin, end;

  gettimeofday( &begin, NULL );

  double *gpu_sum;
  cudaError_t err = cudaMalloc(&gpu_sum, nbBlocks * sizeof(double));
  if (err != cudaSuccess) {
    printf("Err:  %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  long offset = nbSteps/(nbBlocks-1) > 0 ? nbSteps/(nbBlocks-1) : 1;

  computePi<<<nbBlocks, 1>>>(step, nbSteps, offset, gpu_sum);

  cudaMemcpy(cpu_sum, gpu_sum, nbBlocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(gpu_sum);

  double sum = 0.0;
  for (int i=0; i<nbBlocks; i++) {
    sum += cpu_sum[i];
  }
  pi = step * sum;

  gettimeofday( &end, NULL );

  // Calculate time.
  double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
            1.0e-6 * ( end.tv_usec - begin.tv_usec );

  printf("\n pi with %ld steps is %lf in %lf seconds\n ",nbSteps, pi,time);
}
