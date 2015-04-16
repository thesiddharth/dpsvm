#include <stdio.h>
#include <mpi.h>
#include <cblas.h>

float m[] = {
  	3, 1, 3,
    1, 5, 9,
	2, 6, 5
	};

float x[] =  {
		-1, -1, 1
		};

float y[] = {
		 	0, 0, 0
			};

int main(int argc, char *argv[]) {
	int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Init(&argc, &argv);
 	
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);
    
	printf("Process %d on %s out of %d\n", rank, processor_name, numprocs);

	int i, j;

	for (i=0; i<3; ++i) {
		for (j=0;j<3; ++j) printf("%5.1f", m[i*3+j]);
		putchar('\n');
	}
	  
	cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, m, 3, x, 1, 0.0, y, 1);

	for (i=0; i<3; ++i)  printf("%5.1f\n", y[i]);

	MPI_Finalize();

	return 0;

}

