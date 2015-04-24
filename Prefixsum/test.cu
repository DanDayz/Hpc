# include <bits/stdc++.h>
# include <cuda.h>

#define SIZE 60// Global Size
#define BLOCK_SIZE 1024
using namespace std;

//::::::::::::::::::::::::::::::::::::::::::GPU::::::::::::::::::::::::::::::::

// :::: Kernel

__global__ void kernel_prefix_sum_inefficient(double *g_idata,double *g_odata,int l){ // Sequential Addressing technique

  __shared__ double sdata[BLOCK_SIZE];
  // each thread loads one element from global to shared mem

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i<l && tid !=0){ // bad thing -> severely punished performance.
    sdata[tid] = g_idata[i-1];
  }else{
    sdata[tid] = g_idata[0];
  }

  // do reduction in shared mem
  for(unsigned int s=1;s<=tid;s *=2){
    __syncthreads();
     sdata[tid]+=sdata[tid-s];
    }

  // write result for this block to global mem
  g_odata[i] = sdata[tid];
}

// :::: Calls
void d_VectorMult(double *Vec1,double *Total){
  double * d_Vec1;
  double * d_Total;
  double Blocksize=BLOCK_SIZE; // Block of 1Dim

  cudaMalloc((void**)&d_Vec1,SIZE*sizeof(double));
  cudaMalloc((void**)&d_Total,SIZE*sizeof(double));

  cudaMemcpy(d_Vec1,Vec1,SIZE*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Total,Total,SIZE*sizeof(double),cudaMemcpyHostToDevice);

  dim3 dimBlock(Blocksize,1,1);
  dim3 dimGrid(ceil(SIZE/Blocksize),1,1);
  kernel_prefix_sum_inefficient<<<dimGrid,dimBlock>>>(d_Vec1,d_Total,SIZE);

/*  int temp=SIZE;

    while(temp>1){
      dim3 dimBlock(Blocksize,1,1);
      int grid=ceil(temp/Blocksize);
      dim3 dimGrid(grid,1,1);

      KernelNormalVec
      cudaDeviceSynchronize();

      cudaMemcpy(d_Vec1,d_Total,SIZE*sizeof(double),cudaMemcpyDeviceToDevice);
      temp=ceil(temp/Blocksize);
    }
*/


    cudaMemcpy(Total,d_Total,SIZE*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(d_Vec1);
    cudaFree(d_Total);
  }

  //::::::::::::::::::::::::::::::::::::::::::CPU::::::::::::::::::::::::::::::::

  void h_prefix_sum(double *Vec1, double *all){
    all[0]=Vec1[0];
    for(int i=0;i<SIZE;i++) all[i]=all[i-1]+Vec1[i];
  }

  //:::::::::::::::::::::::::::: Rutinary Functions

  void Fill_vec(double *Vec,double Value){
    for(int i =0 ; i<SIZE ; i++) Vec[i]=Value;
  }

  void Show_vec(double *Vec){
    for (int i=0;i<SIZE;i++){
      if(i%10==0 && i!=0){
        cout<<endl;
      }
      cout<<"["<<Vec[i]<<"] ";
    }
    cout<<endl;
  }

  void Checksum(double *Answer1 , double  *Answer2){
    if(fabs(Answer1[0]-Answer2[0]) < 0.1) cout<<"Nice Work Guy"<<endl;
    else  cout<<"BAD Work Guy"<<endl;
  }


  // :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::
  double diffclock(clock_t clock1,clock_t clock2){
    double diffticks=clock2-clock1;
    double diffms=(diffticks)/(CLOCKS_PER_SEC/1); // /1000 mili
    return diffms;
  }

  // :::::::::::::::::::::::::::::::::::::::Main::::::::::::::::::::::::::::::::.

int main(){

    double T1,T2; // Time flags
    double *Vec1 = (double*)malloc((SIZE)*sizeof(double)); // Elements to compute. CPU way
    double *Total2 = (double*)malloc((SIZE)*sizeof(double)); // GPU
    double *Total1 = (double*)malloc(sizeof(double)*(SIZE)); // Total Variables.

    // Fill the containers vectors of data
    Fill_vec(Vec1,1.0);
    Fill_vec(Total2,0.0);

    // Register time to finish the algorithm
    // Secuential
    clock_t start = clock();
    h_prefix_sum(Vec1,Total1);
    clock_t end = clock();
    T1=diffclock(start,end);
    Show_vec(Total1);
    //cout<<"Serial Result: "<<*Total1<<" At "<<T1<<",Seconds"<<endl;
    // Parallel

    // releasing Memory

    free(Vec1);
    free(Total1);
    free(Total2);

    return 0;
}
