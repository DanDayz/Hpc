#include <bits/stdc++.h>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

#define Mask_size  3
using namespace cv;
using namespace std;

int clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}

unsigned char filter(int a,int b){
  int out = round(sqrt(a*a + b*b));
  if(out > 255){
    out = 255;
  }
  return out;
}

void grayfilter(unsigned char *In, unsigned char *Out, int Width , int Height ){
  for(int i=0 ; i<Height; i++){
    for(int j=0;j<Width;j++){
      Out[i*Width+j]=In[(i*Width+j)*3+2]*0.299+In[(i*Width+j)*3+1]*0.587+In[(i*Width+j)*3+0]*0.114;
    }
  }
}

void Sobel_filter(unsigned char * In,unsigned char *Out,char * Kernel,int Mask_width,int width,int height){

  for(int i=0 ; i< height; i++){
      for(int j=0 ;j<width;j++){
        int Gap_x=i-(Mask_width)/2; // asymmetric Gap (Left Right)
    		int Value1=0;
    		int Value2=0;
        int Gap_y=j-(Mask_width)/2; // asymmetric Gap (upper lower)
        for(int kernel_i = 0; kernel_i < Mask_width; kernel_i++){
          for(int kernel_j = 0; kernel_j < Mask_width; kernel_j++){
            if(Gap_x + kernel_i >= 0 and Gap_x+ kernel_i < height and Gap_y + kernel_j >= 0 and Gap_y+kernel_j < width){
              Value1+= (int)In[(Gap_x+kernel_i)*width+(Gap_y+kernel_j)] *(int)Kernel[kernel_i*Mask_width+kernel_j];
              Value2+= (int)In[Gap_x+kernel_i*width+Gap_y+kernel_j] *(int) Kernel[kernel_j*Mask_width+kernel_i];
          }// end if
        }// end for kernel_i
    	}// end for kernel_j
      Out[i*width+j]=filter(clamp(Value1),clamp(Value2));
 		}// end for j

  }// end for i
}


int main(){

  Mat image,image_final;
  image=imread("inputs/img3.jpg",1);
  Size s = image.size();
  int width = s.width;
  int height= s.height;
  int channels = image.channels();
  unsigned char *image_data;
  unsigned char *image_gray_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);
  unsigned char *image_sobel_out = (unsigned char *)malloc(sizeof(unsigned char)*width*height);

  //int Kernel[] = {-1,-1,-1,0,0,0,1,1,1};
  char Kernel[] = {-1,0,1,-2,0,2,-1,0,1};
  image_data=image.data;
  grayfilter(image_data,image_gray_out,width,height);
  Sobel_filter(image_gray_out,image_sobel_out,Kernel,Mask_size,width,height);
  image_final.create(height,width,CV_8UC1);
  image_final.data = image_sobel_out;
  imwrite("./outputs/1088015148.png",image_final);

  return 0;
}
