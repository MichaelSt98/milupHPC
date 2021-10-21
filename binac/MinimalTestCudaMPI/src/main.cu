#include <mpi.h>
#include <cuda_runtime.h>
#include <fstream>


int main( int argc, char *argv[] )
{
  int rank;
  float *ptr = NULL;
  const size_t elements = 12;
  MPI_Status status;

  float h_array[elements];

  MPI_Init( NULL, NULL );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  cudaSetDevice(rank);
  cudaMalloc( (void**)&ptr, elements * sizeof(float) );

  for (int i=0; i<elements; i++) {
    if (rank==0) {
      h_array[i] = i;
    }
    else {
      h_array[i] = 0.0;
    }
    printf("[rank %i] h_array[i] = %f\n", rank, i, h_array[i]);    
  }
  
  if (rank==0)
    cudaMemcpy(ptr, &h_array, elements*sizeof(float), cudaMemcpyHostToDevice);

  if( rank == 0 )
    MPI_Send( ptr, elements, MPI_FLOAT, 1, 0, MPI_COMM_WORLD );
  if( rank == 1 )
    MPI_Recv( ptr, elements, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status );

  if (rank == 1) {
    cudaMemcpy(&h_array, ptr, elements*sizeof(float), cudaMemcpyDeviceToHost);
  }

  if (rank==1)
    printf("Finished communication!\n");

  for (int i=0; i<elements; i++) {
    if (rank == 1) {
      printf("[rank [%i] h_array[i] = %f\n", rank, i, h_array[i]);
    } 
 }

  cudaFree( ptr );
  MPI_Finalize();

  return 0;
}
