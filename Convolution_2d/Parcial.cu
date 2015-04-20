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


__global__ void sobelFilterShared(unsigned char *data, unsigned char *result, int width, int height){
  // Data cache: threadIdx.x , threadIdx.y
  const int n = (Mask_size*Mask_size) / 2;
  __shared__ int s_data[BLOCKSIZE + Mask_size  ][BLOCKSIZE + Mask_size ];

  // global mem address of the current thread in the whole grid
  const int pos = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * width + blockIdx.y * blockDim.y * width;

  // load cache (32x32 shared memory, 16x16 threads blocks)
  // each threads loads four values from global memory into shared mem
  // if in image area, get value in global mem, else 0
  int x, y; // image based coordinate

  // original image based coordinate
  const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  const int y0 = threadIdx.y + blockIdx.y * blockDim.y;

  // case1: upper left
  x = x0 - n;
  y = y0 - n;
  if ( x < 0 || y < 0 )
    s_data[threadIdx.y][threadIdx.x] = 0;
  else
    s_data[threadIdx.y][threadIdx.x] = *(data + pos - n - (width * n));

  // case2: upper right
  x = x0 + n;
  y = y0 - n;
  if ( x > (width - 1) || y < 0 )
    s_data[threadIdx.y][threadIdx.x + blockDim.x] = 0;
  else
    s_data[threadIdx.y][threadIdx.x + blockDim.x] = *(data + pos + n - (width * n));

  // case3: lower left
  x = x0 - n;
  y = y0 + n;
  if (x < 0 || y > (height - 1))
    s_data[threadIdx.y + blockDim.y][threadIdx.x] = 0;
  else
    s_data[threadIdx.y + blockDim.y][threadIdx.x] = *(data + pos - n + (width * n));

  // case4: lower right
  x = x0 + n;
  y = y0 + n;
  if ( x > (width - 1) || y > (height - 1))
    s_data[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = 0;
  else
    s_data[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = *(data + pos + n + (width * n));

  __syncthreads();

  // convolution
  int sum = 0;
  x = n + threadIdx.x;
  y = n + threadIdx.y;
  for (int i = - n; i <= n; i++)
    for (int j = - n; j <= n; j++)
      sum += s_data[y + i][x + j] * Global_Mask[n + i] * Global_Mask[n + j];

  result[pos] = sum;
}


__global__ void gray(unsigned char *In, unsigned char *Out,int Row, int Col){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < Col) && (col < Row)){
        Out[row*Row+col] = In[(row*Row+col)*3+2]*0.299 + In[(row*Row+col)*3+1]*0.587+ In[(row*Row+col)*3]*0.114;
    }
}

// :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::
double diffclock(clock_t clock1,clock_t clock2){
  double diffticks=clock2-clock1;
  double diffms=(diffticks)/(CLOCKS_PER_SEC/1); // /1000 mili
  return diffms;
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
    sobelFilterShared<<<dimGrid,dimBlock>>>(d_Out,d_sobelOut,Row,Col);
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

    double T1,T2; // Time flags
    clock_t start,end;// Time flags

    int Mask_Width = Mask_size;
    char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1};
    Mat image,result_image;
    image = imread("inputs/img1.jpg",1);
    Size s = image.size();
    int Row = s.width;
    int Col = s.height;
    unsigned char * In = (unsigned char*)malloc( sizeof(unsigned char)*Row*Col*image.channels());
    unsigned char * h_Out = (unsigned char *)malloc( sizeof(unsigned char)*Row*Col);

    In = image.data;
    start = clock();
    d_convolution2d(image,In,h_Out,h_Mask,Mask_Width,Row,Col,1);
    end = clock();
    T1=diffclock(start,end);
    cout<<" Result Parallel"<<" At "<<T1<<",Seconds"<<endl;

    Mat gray_image_opencv, grad_x, abs_grad_x;
    start = clock();
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    end = clock();
    T2=diffclock(start,end);
    cout<<" Result secuential"<<" At "<<T2<<",Seconds"<<endl;
    cout<<"Total acceleration "<<T2/T1<<"X"<<endl;

    result_image.create(Col,Row,CV_8UC1);
    result_image.data = h_Out;
    imwrite("./outputs/1088015148.png",grad_x);

    return 0;
}
