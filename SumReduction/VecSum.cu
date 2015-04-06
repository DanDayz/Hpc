# include <bits/stdc++.h>
# include <cuda.h>

#define SIZE 500000// Global Size
#define TILE_WIDTH 1024
using namespace std;

// ::::::::::::::::::::::::::::::::::::::::::GPU::::::::::::::::::::::::::::::::

// :::: Kernel

__global__ void KernelNormalVec(float *g_idata,float *g_odata,int l){

         __shared__ float sdata[TILE_WIDTH];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  	if(i<l){
    	sdata[tid] = g_idata[i];
    }else{
    	sdata[tid] = 0.0;
    }

    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// :::: Calls
void d_VectorMult(float *Vec1,float *Total){
  float * d_Vec1;
  float * d_Total;
  float Blocksize=TILE_WIDTH; // Block of 1Dim

  cudaMalloc((void**)&d_Vec1,SIZE*sizeof(float));
  cudaMalloc((void**)&d_Total,SIZE*sizeof(float));

  cudaMemcpy(d_Vec1, Vec1,SIZE*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Total,Total,SIZE*sizeof(float),cudaMemcpyHostToDevice);

  int temp=SIZE;

  while(temp>1){
     dim3 dimBlock(Blocksize,1,1);
     int grid=ceil(temp/Blocksize);
	 	 dim3 dimGrid(grid,1,1);
     KernelNormalVec<<<dimGrid,dimBlock>>>(d_Vec1,d_Total,temp);
     cudaDeviceSynchronize();
     cudaMemcpy(d_Vec1,d_Total,SIZE*sizeof(float),cudaMemcpyDeviceToDevice);
     temp=ceil(temp/Blocksize);
  }

  cudaMemcpy(Total,d_Total,SIZE*sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d_Vec1);
  cudaFree(d_Total);

}

//::::::::::::::::::::::::::::::::::::::::::CPU::::::::::::::::::::::::::::::::

float h_Mul_Mat(float *Vec1){
  float all=0;
  for(int i=0;i<SIZE;i++){all+=Vec1[i];}
  return all;
}

//:::::::::::::::::::::::::::: Rutinary Functions

void Fill_vec(float *Vec,float Value){
  for(int i =0 ; i<SIZE ; i++) Vec[i]=Value;
}

void Show_vec(float *Vec){
  for (int i=0;i<SIZE;i++){
    if(i%10==0 && i!=0){
      cout<<endl;
    }
    cout<<"["<<Vec[i]<<"] ";
  }
  cout<<endl;
}

void Checksum(float Answer1 , float  *Answer2){
  if(fabs(Answer1-Answer2[0]) < 0.1) cout<<"Nice Work Guy"<<endl;
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
  float *Vec1 = (float*)malloc((SIZE)*sizeof(float)); // Elements to compute. CPU way
  // GPU "Normal" way
  float *Total2 = (float*)malloc((SIZE)*sizeof(float));
  float Total1; // Total Variables.


  // Fill the containers vectors of data
  Fill_vec(Vec1,1.0);
  Fill_vec(Total2,0.0);


  // Register time to finish the algorithm
  // Secuential
  clock_t start = clock();
  Total1=h_Mul_Mat(Vec1);
  clock_t end = clock();
  T1=diffclock(start,end);
  cout<<"Serial Result: "<<Total1<<" At "<<T1<<",Seconds"<<endl;
  // Parallel
  start = clock();
  d_VectorMult(Vec1,Total2);
  end = clock();
  T2=diffclock(start,end);
  cout<<"Parallel Result: "<<Total2[0]<<" At "<<T2<<",Seconds"<<endl;
  cout<<"Total Acceleration: "<<T1/T2<<",X"<<endl;
	Checksum(Total1,Total2);
  // releasing Memory

  free(Vec1);
  free(Total2);

  return 0;
}
