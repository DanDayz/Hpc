
/* Daniel Diaz Giraldo

Restrictions
Mask = 5, Only works whit odd numbers and Mask size <= N _elements;
N_elements = defined by architecture from machine; (Femin-Maxwell....) in this case
i'm use a Kepler Arch; (the number of blocks that can support is around 2^31)

*/

#include <bits/stdc++.h>
#include <cuda.h>

#define N_elements 50000000
#define Mask_size  5
#define TILE_SIZE  32
#define BLOCK_SIZE 4

using namespace std;

__constant__ int Global_Mask[Mask_size];

//:::::::::::::::::::::::::::: Device Kernel Function ::::::::::::::::::::::::::::::

__global__ void convolution1d_tiles_constant_kernel(int *In, int *Out){
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; // Index 1d iterator.
  __shared__ int Tile[TILE_SIZE + Mask_size - 1];
  int n = Mask_size/2;
  int halo_left_index  = (blockIdx.x - 1 ) * blockDim.x + threadIdx.x;
  if (threadIdx.x  >= blockDim.x - n ){
     Tile[threadIdx.x - (blockDim.x - n )] = (halo_left_index < 0) ? 0 : In[halo_left_index];
  }
  Tile[n + threadIdx.x] = In[blockIdx.x * blockDim.x + threadIdx.x  ];
  int halo_right_index = (blockIdx.x + 1 ) * blockDim.x + threadIdx.x;
  if (threadIdx.x < n) {
    Tile[n + blockDim.x + threadIdx.x]=  (halo_right_index >= N_elements) ? 0 : In[halo_right_index];
  }
__syncthreads();
  int Value = 0;
  for (unsigned int j = 0; j  < Mask_size; j ++) {
    Value += Tile[threadIdx.x + j] * Global_Mask[j];
  }
  Out[index] = Value;
}

__global__ void convolution1d_notile_noconstant_kernel(int *In, int *Out){
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; // Index 1d iterator.
  int Value = 0;
  int N_start_point = index - (Mask_size/2);
  for ( int j = 0; j  < Mask_size; j ++) {
    if (N_start_point + j >= 0 && N_start_point + j < N_elements) {
      Value += In[N_start_point + j] * Global_Mask[j];
    }
  }
  Out[index] = Value;
}

//:: Invocation Function

void d_convolution1d(int *In,int *Out,int *h_Mask,int op){
  // Variables
  int Size_of_bytes = N_elements * sizeof(int);
  int *d_In, *d_Out;
  float Blocksize=BLOCK_SIZE;
  d_In = (int*)malloc(Size_of_bytes);
  d_Out = (int*)malloc(Size_of_bytes);
  // Memory Allocation in device
  cudaMalloc((void**)&d_In,Size_of_bytes);
  cudaMalloc((void**)&d_Out,Size_of_bytes);
  // Memcpy Host to device
  cudaMemcpy(d_In,In,Size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Out,Out,Size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Global_Mask,h_Mask,Mask_size*sizeof(int)); // avoid cache coherence
  // Thead logic and Kernel call
  dim3 dimGrid(ceil(N_elements/Blocksize),1,1);
  dim3 dimBlock(Blocksize,1,1);
  if(op==1){
    convolution1d_tiles_constant_kernel<<<dimGrid,dimBlock>>>(d_In,d_Out);
  }else{
    convolution1d_notile_noconstant_kernel<<<dimGrid,dimBlock>>>(d_In,d_Out);
  }

  cudaDeviceSynchronize();
  // save output result.
  cudaMemcpy (Out,d_Out,Size_of_bytes,cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_In);
  cudaFree(d_Out);
  //cudaFree(d_Mask);
}

//:::::::::::::::::::::::::::: Host Function ::::::::::::::::::::::::::::::

void h_Convolution_1d(int *In,int *Out, int *Mask){
  for(int i=0;i<N_elements;i++){
    int Gap=i-(Mask_size)/2; // asymmetric Gap (Left Right)
    int Value=0;
    for(int j=0;j<Mask_size;j++){
      if(Gap+j >= 0 && j+Gap<N_elements){
        Value+=In[Gap+j]*Mask[j];
      }// end if
    }// end for j
    Out[i]=Value;
  }// end for i
}
//:::::::::::::::::::::::::::: Rutinary Functions ::::::::::::::::::::::::::::::

void Fill_elements(int * VecIn1,int Value, int n){
    for (int i = 0; i < n; i++) {
          VecIn1[i]=Value;
    }
}

void Show_vec(int *Vec,int Elements,char * Msg ){
  cout<<Msg<<endl;
  for (int i=0;i<Elements;i++){
    if(i%10==0 && i!=0){
      cout<<endl;
    }
    cout<<"["<<Vec[i]<<"] ";
  }
  cout<<endl;
}

int Check_op(int *A , int *B){
    for(unsigned int i=0; i<N_elements;i++){
        if(A[i]!=B[i]){
          cout<<"Bad Work D:"<<endl;
          return 0;
        }
    }
    cout<<"Nice Work :D"<<endl;
    return 1;
}

// :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::
double diffclock(clock_t clock1,clock_t clock2){
  double diffticks=clock2-clock1;
  double diffms=(diffticks)/(CLOCKS_PER_SEC/1); // /1000 mili
  return diffms;
}
// :::::::::::::::::::::::::::::::::::::::Main::::::::::::::::::::::::::::::::::

int main(){

  double T1,T2; // Time flags
  clock_t start,end;// Time flags

  int *VecIn1=(int*)malloc(N_elements*sizeof(int)); // Sequential and Parallel Vector Input
  int *VecOut1=(int*)malloc(N_elements*sizeof(int)); // Sequential Vector Output
  int *VecOut2=(int*)malloc(N_elements*sizeof(int)); // Parallel Vector Output
  int *Mask=(int*)malloc(Mask_size*sizeof(int)); // Mask Vector;

  Fill_elements(VecIn1,1,N_elements);
  Fill_elements(Mask,1,Mask_size);

  //Show_vec(VecIn1,N_elements,(char *)"Vector In");
  //Show_vec(Mask,Mask_size,(char *)"Mask");
  start = clock();
	h_Convolution_1d(VecIn1,VecOut1,Mask);
  end = clock();
  T1=diffclock(start,end);
  cout<<"Serial Result"<<" At "<<T1<<",Seconds"<<endl;
  //Show_vec(VecOut1,N_elements,(char *)"Vector Out");

  start = clock();
  d_convolution1d(VecIn1,VecOut2,Mask,1);
  end = clock();
  T2=diffclock(start,end);
  cout<<"Parallel Result"<<" At "<<T2<<",Seconds"<<endl;
  //Show_vec(VecOut2,N_elements,(char *)"Vector Out");
  cout<<"Total acceleration "<<T1/T2<<"X"<<endl;
  Check_op(VecOut1,VecOut2);

  // Releasing Memory

  free(VecIn1);
  free(VecOut1);
  free(VecOut2);
  free(Mask);

  return 0;
}
