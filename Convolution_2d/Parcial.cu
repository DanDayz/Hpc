/* Daniel Diaz Giraldo

Restrictions
Mask = 5, Only works whit odd numbers and Mask size <= N _elements;
N_elements = defined by architecture from machine; (Femin-Maxwell....) in this case
i'm use a Kepler Arch; (the number of blocks that can support is around 2^31)

*/

#include <bits/stdc++.h>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

#define N_elements 32
#define Mask_size  3
#define TILE_SIZE  1024
#define BLOCK_SIZE 32

using namespace std;
using namespace cv;

__constant__ char Global_Mask[Mask_size*Mask_size];

__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}



__global__ void convolution2d_global_kernel(unsigned char *In,char *M, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg){

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++){
       for(int j = 0; j < Mask_Width; j++ ){
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)&&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg)){
               Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
           }
       }
   }

   Out[row*Rowimg+col] = clamp(Pvalue);

}

__global__ void convolution2d_constant_kernel(unsigned char *In,char *M, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg){

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++){
       for(int j = 0; j < Mask_Width; j++ ){
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)&&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg)){
               Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * Global_Mask[i*Mask_Width+j];
           }
       }
   }
   Out[row*Rowimg+col] = clamp(Pvalue);
}



__global__ void convolution2d_tiled_constant_kernel(unsigned char *In,char *M, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg){

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
   __shared__ int Tile[(TILE_SIZE + Mask_size - 1)*2];

   int

   int Pvalue = 0;
   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++){
       for(int j = 0; j < Mask_Width; j++ ){
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)&&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg)){
               Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * Global_Mask[i*Mask_Width+j];
           }
       }
   }
   Out[row*Rowimg+col] = clamp(Pvalue);
}

/*
__global__ void convolution1d_tiles_constant_kernel(int *In, int *Out){
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; // Index 1d iterator.
  __shared__ int Tile[TILE_SIZE + Mask_size - 1];
  int nx = Mask_size/2;
  int halo_left_index  = (blockIdx.x - 1 ) * blockDim.x + threadIdx.x;
  if (threadIdx.x  >= blockDim.x - n ){
     Tile[threadIdx.x - (blockDim.x - n )] = (halo_left_index < 0) ? 0 : In[halo_left_index];
  }

  if(index<N_elements){Tile[n + threadIdx.x] = In[index];
  }else{Tile[n + threadIdx.x] = 0;}
  int halo_right_index = (blockIdx.x + 1 ) * blockDim.x + threadIdx.x;
  if (threadIdx.x < n) {
    Tile[n + blockDim.x + threadIdx.x]=  (halo_right_index >= N_elements) ? 0 : In[halo_right_index];
  }

  int Value = 0;
__syncthreads();
  for (unsigned int j = 0; j  < Mask_size; j ++) {
    Value += Tile[threadIdx.x + j] * Global_Mask[j];
  }
  Out[index] = Value;
}

*/
//:: Invocation Function

void d_convolution1d(Mat image,unsigned char *In,unsigned char *Out,char *h_Mask,int Mask_Width,int Row,int Col,int op){
  // Variables
  int Size_of_bytes =  sizeof(unsigned char)*Row*Col*image.channels();
  int Mask_size_bytes =  sizeof(char)*9;
  unsigned char *d_In, *d_Out;
  char *d_Mask;
  float Blocksize=BLOCK_SIZE;


  // Memory Allocation in device
  cudaMalloc((void**)&d_In,Size_of_bytes);
  cudaMalloc((void**)&d_Out,Size_of_bytes);
  cudaMalloc((void**)&d_Mask,Mask_size_bytes);
  // Memcpy Host to device
  cudaMemcpy(d_In,In,Size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Out,Out,Size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Global_Mask,h_Mask,Mask_size_bytes); // avoid cache coherence
  // Thead logic and Kernel call
  dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  if(op==1){
    convolution2d_global_kernel<<<dimGrid,dimBlock>>>(d_In,d_Mask,d_Out,Mask_Width,Row,Col);
  }
  if(op==2){
    convolution2d_constant_kernel<<<dimGrid,dimBlock>>>(d_In,d_Mask,d_Out,Mask_Width,Row,Col);
  }
  if(op==3){

  }

  cudaDeviceSynchronize();
  // save output result.
  cudaMemcpy (Out,d_Out,Size_of_bytes,cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_In);
  cudaFree(d_Out);
  cudaFree(d_Mask);
}



// :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::
double diffclock(clock_t clock1,clock_t clock2){
  double diffticks=clock2-clock1;
  double diffms=(diffticks)/(CLOCKS_PER_SEC/1); // /1000 mili
  return diffms;
}
// :::::::::::::::::::::::::::::::::::::::Main::::::::::::::::::::::::::::::::::

int main(){

  int Mask_Width =  Mask_size;
  Mat image;
  image = imread("inputs/img1.jpg",0);   // Read the file
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;
  //char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1};
  //char h_Mask[] = {0,-1,0,-1,5,-1,0,-1,0}; Sharpen
  //char h_Mask[] = {-1,-1,-1,-1,8,-1,-1,-1,-1}; edge detection 3
  //char h_Mask[] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11};
  //char h_Mask[] = {-2,-2,0,-2,6,0,0,0,0};
  //char h_Mask[] = {1,2,1,2,4,2,1,2,1}; gaussian blur
  char h_Mask[] = {-1,-1,-1,0,0,0,1,1,1}; // A kernel for edge detection
  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());
  unsigned char *imgOut = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  img = image.data;

   //::::::::::::::::::::::::::::::::::::::::: Secuential filter ::::::::::::::::::::::::::::::::::::

  /// Generate grad_x and grad_y
  //Mat grad_x, grad_y;

  /// Gradient X
  //   ( src  , grad_x, ddepth,dx,dy,scale,delta, BORDER_DEFAULT );
  //Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

  /// Gradient Y
  //Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  //::::::::::::::::::::::::::::::::::::::::: Parallel filter ::::::::::::::::::::::::::::::::::::

  d_convolution1d(image,img,imgOut,h_Mask,Mask_Width,Row,Col,2);
  Mat gray_image;
  gray_image.create(Row,Col,CV_8UC1);
  gray_image.data = imgOut;
  imwrite("./outputs/1088015148.png",gray_image);

  //free(img);
  //free(imgOut);

  return 0;
}
/*
1 - convolution2d tile constant
2 - convolution2d notile noconstant
3 - convolution2d constant tile simple
*/
