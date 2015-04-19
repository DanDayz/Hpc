#include <bits/stdc++.h>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

#define Mask_size  9
using namespace cv;
using namespace std;

unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}



void filter(unsigned char * In,unsigned char *Out,char *Kernel,int Mask_width,int Rowimg,int Colimg){

  for(int i=0 ; i< Rowimg *Colimg ; i++){
    int Gap=i-(Mask_width)/2; // asymmetric Gap (Left Right)
    int Value=0;
      for(int j=0 ;j<Mask_width;j++){
        if(Gap+j >= 0 && j+Gap<(Rowimg *Colimg)){
          Value+=In[Gap+j]*Kernel[j];
        }// end if
      }
      Out[i]=clamp(Value);
  }

}


int main(){

  Mat image,image_final;
  image=imread("inputs/img1.jpg",0);
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;
  int channels = image.channels();
  unsigned char *image_data = (unsigned char *)malloc(sizeof(unsigned char)*Row*Col*channels);
  unsigned char *image_out = (unsigned char *)malloc(sizeof(unsigned char)*Row*Col*channels);
  char Kernel[] = {-1,-1,-1,0,0,0,1,1,1};
  image_data=image.data;
  filter(image_data,image_out,Kernel,Mask_size*,Row,Col);
  image_final.create(Row,Col,CV_8UC1);
  image_final.data = image_out;
  imwrite("./outputs/1088015148.png",image_final);

  return 0;
}
