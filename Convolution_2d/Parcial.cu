#include <bits/stdc++.h>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

#define Mask_size 3
//#define TILE_size_of_rgb  1024
#define BLOCKSIZE 32

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

__global__ void sobelFilter(unsigned char *In, int Row, int Col, unsigned int Mask_Width,char *Mask,unsigned char *Out){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue = 0;
    int N_start_point_row = row - (Mask_Width/2);
    int N_start_point_col = col - (Mask_Width/2);

    for(int i = 0; i < Mask_Width; i++){
        for(int j = 0; j < Mask_Width; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < Row)&&(N_start_point_row + i >=0 && N_start_point_row + i < Col)){
                Pvalue += In[(N_start_point_row + i)*Row+(N_start_point_col + j)] * Mask[i*Mask_Width+j];
            }
        }
    }
    Out[row*Row+col] = clamp(Pvalue);
}


__global__ void sobelFilterConstant(unsigned char *In, int Row, int Col, unsigned int Mask_Width,char *Mask,unsigned char *Out){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue = 0;
    int N_start_point_row = row - (Mask_Width/2);
    int N_start_point_col = col - (Mask_Width/2);

    for(int i = 0; i < Mask_Width; i++){
        for(int j = 0; j < Mask_Width; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < Row)&&(N_start_point_row + i >=0 && N_start_point_row + i < Col)){
                Pvalue += In[(N_start_point_row + i)*Row+(N_start_point_col + j)] * Mask[i*Mask_Width+j];
            }
        }
    }
    Out[row*Row+col] = clamp(Pvalue);
}

__global__ void gray(unsigned char *In, unsigned char *Out,int Row, int Col){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < Col) && (col < Row)){
        Out[row*Row+col] = In[(row*Row+col)*3+2]*0.299 + In[(row*Row+col)*3+1]*0.587+ In[(row*Row+col)*3+0]*0.114;
    }
}


void d_convolution2d(Mat image,unsigned char *In,unsigned char *h_Out,char *h_Mask,int Mask_Width,int Row,int Col,int op){
  // Variables
  int size_of_rgb = sizeof(unsigned char)*Row*Col*image.channels();
  int size_of_Gray = sizeof(unsigned char)*Row*Col; // sin canales alternativos
  int Mask_size_of_bytes =  sizeof(char)*(Mask_size*Mask_size);
  unsigned char *d_In,*d_Out,*d_sobelOut;
  char *d_Mask;
  float Blocksize=BLOCKSIZE;

  // Memory Allocation in device
  cudaMalloc((void**)&d_In,size_of_rgb);
  cudaMalloc((void**)&d_Out,size_of_Gray);
  cudaMalloc((void**)&d_Mask,Mask_size_of_bytes);
  cudaMalloc((void**)&d_sobelOut,size_of_Gray);

  // Memcpy Host to device
  cudaMemcpy(d_In,In,size_of_rgb, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask,h_Mask,Mask_size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Global_Mask,h_Mask,Mask_size_of_bytes); // avoid cache coherence
  // Thread logic and Kernel call
  dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  gray<<<dimGrid,dimBlock>>>(d_In,d_Out,Row,Col); // pasando a escala de grices.
  cudaDeviceSynchronize();
  if(op==1){
    sobelFilter<<<dimGrid,dimBlock>>>(d_Out,Row,Col,Mask_size,d_Mask,d_sobelOut);
  }
  if(op==2){
    sobelFilterConstant<<<dimGrid,dimBlock>>>(d_Out,Row,Col,Mask_size,d_Mask,d_sobelOut);
  }
  if(op==3){

  }
  // save output result.
  cudaMemcpy (h_Out,d_sobelOut,size_of_Gray,cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_In);
  cudaFree(d_Out);
  cudaFree(d_Mask);
  cudaFree(d_sobelOut);
}


int main(){

    int Mask_Width = Mask_size;
    char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1};
    Mat image,result_image;
    image = imread("inputs/img2.jpg",1);
    Size s = image.size();
    int Row = s.width;
    int Col = s.height;
    unsigned char * In = (unsigned char*)malloc( sizeof(unsigned char)*Row*Col*image.channels());
    unsigned char * h_Out = (unsigned char *)malloc( sizeof(unsigned char)*Row*Col);

    In = image.data;
    d_convolution2d(image,In,h_Out,h_Mask,Mask_Width,Row,Col,1);

    result_image.create(Col,Row,CV_8UC1);
    result_image.data = h_Out;
    imwrite("./outputs/1088015148.png",result_image);


    return 0;
}
