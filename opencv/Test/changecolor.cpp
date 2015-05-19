#include <highgui.h>
#include <bits/stdc++.h>
#include <cv.h>

using namespace cv;
using namespace std;

int main(){

  Mat image,image_example;
  uchar *image_data;
  image = imread("red.jpg",CV_LOAD_IMAGE_COLOR);   // Read the file, BGR format
  Size s = image.size();
  int width = s.width;
  int height= s.height;
  uchar *image_example_data = (unsigned char *)malloc(sizeof(unsigned char)*(width*height)*3); // 3 channels
  image_data = image.data;

  for (int i=0 ; i < height; i++) {
      for (int j = 0; j < width; j++) {
        unsigned int B=(int)image_data[(i*width+j)*3+0];
        unsigned int G=(int)image_data[(i*width+j)*3+1];
        unsigned int R=(int)image_data[(i*width+j)*3+2];
        //cout<<"Pixel: "<<j<<"x"<<i<<" B: "<<B<<" G: "<<G<<" R: "<<R<<endl;
        image_example_data[(i*width+j)*3]=B+50;
        image_example_data[(i*width+j)*3+1]=G+50;
        image_example_data[(i*width+j)*3+2]=R-150;

      }
}


  image_example.create(height,width,CV_8UC3);  // 8 bits 3 Channels
  //image_example.create(1,2,CV_8UC3);  // 8 bits 3 Channels
  image_example.data = image_example_data;

  namedWindow("image", CV_WINDOW_AUTOSIZE);
  imshow("image", image);
  waitKey();

  namedWindow("image2", CV_WINDOW_AUTOSIZE);
  imshow("image2", image_example);
  waitKey();

  imwrite("test1.jpg",image_example);



  return 0;
}
