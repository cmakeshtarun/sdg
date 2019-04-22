#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

extern double size;
//
//  benchmarking program
//

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{

}

__global__ void compute_forces_gpu(particle_t * particles, particle_t** d_bins, int* d_npbins, int n, double bin_dim, int n_bins, int bin_size)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

}

__global__ void create_bins_gpu (particle_t* particles, particle_t** d_bins, int* d_npbins, int n, double size, double bin_dim, int n_bins, int bin_size)
{
}



int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    if(!savename)                                                                                  
	savename = "out.txt";
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    double size = get_size();
    init_particles( n, particles );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    //Setting up variables for binning
    double bin_dim = cutoff*1;
    int n_bins = size/bin_dim +1;

    //Bin data structure - Hold pointer to particles in that bin - Efficient? 
    //Each bin can contain twice the size of max_particles - vary after profiling
    int bin_size = 2;
    particle_t* *d_bins;
    cudaMalloc((void **) &d_bins, n_bins*n_bins*bin_size*sizeof(particle_t*));

    //Holds the number of particles in each bin
    int* d_npbins;
    cudaMalloc((void **) &d_npbins, n_bins*n_bins*sizeof(int));


    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
	int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

	//
        //  Initialise the particle count array and do binning
	//
        cudaMemset(d_npbins, 0, n_bins*n_bins*sizeof(int));
	create_bins_gpu <<< blks, NUM_THREADS >>> (d_particles, d_bins, d_npbins, n, size, bin_dim, n_bins, bin_size);

        //
        //  compute forces
        //
	compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, d_bins, d_npbins, n, bin_dim, n_bins, bin_size);
        
        //
        //  move particles
        //
	move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        
        //  Moving this down so that the file dump time is not included in execution time
        //  save if necessary
        //
        //if( fsave && (step%SAVEFREQ) == 0 ) {
	//    // Copy the particles back to the CPU
        //    cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
        //    save( fsave, n, particles);
	//}
    }

    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
    save( fsave, n, particles);

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
