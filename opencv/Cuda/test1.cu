#include <bits/stdc++.h>

#define BLOCKSIZE 32

using namespace std;


void steganography(){

}

step_one(int height, int width , Mat image , uchar *image_data , uchar *image_example_data,string mensaje,int *lc){
   // Variables
  int size_of_rgb = sizeof(unsigned char)*Row*Col*image.channels();
  int size_of_msg = sizeof(uchar *)*mensaje.lengh();
  uchar *d_image_data,*d_image_example_data,*d_msg;
  float Blocksize=BLOCKSIZE;

  // Memory Allocation in device
  cudaMalloc((void**)&d_image_data,size_of_rgb);
  cudaMalloc((void**)&d_image_example_data,size_of_rgb);
  cudaMalloc((void**)&d_msg,Size_of_msg);


  // Memcpy Host to device
  cudaMemcpy(d_image_data,image_data,size_of_rgb, cudaMemcpyHostToDevice);
  cudaMemcpy(d_msg,mensaje,size_of_msg,cudaMemcpyHostToDevice);
  // Thread logic and Kernel call
  dim3 dimGrid(ceil(height/Blocksize),ceil(width/Blocksize),1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  gray<<<dimGrid,dimBlock>>>(d_In,d_Out,Row,Col); // pasando a escala de grices.
  cudaDeviceSynchronize();
   // save output result.
  cudaMemcpy (h_Out,d_sobelOut,size_of_Gray,cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_In);
  cudaFree(d_Out);
  cudaFree(d_Mask);
  cudaFree(d_sobelOut);

}


int main(){


  string foto="inputs/img3.jpg";
  string mensaje="hola a todos";

  cout<<"Imagen a codificar: "<<foto<<endl;

  Mat image= imread(foto,CV_LOAD_IMAGE_COLOR);
  uchar *image_data;
  Size s = image.size();
  int *lc = (unsigned int *)malloc(sizeof(unsigned int)*1);
  int width = s.width;
  int height= s.height;
  uchar *image_example_data = (unsigned char *)malloc(sizeof(unsigned char)*(width*height)*3); // 3 channels
  image_data = image.data;

  cout<<"tamaño de la imágen : "<<width <<" x "<<height<<endl;

  step_one(height,width,lc);
  //int nc=step_two(lc);
  //step_three(height,width,nc);
  //step_three(height,width,file,step_two(step_one(height,width)));
  string a="nano "+file;
  system(a.c_str());



  return 0;
}
