#include <highgui.h>
#include <bits/stdc++.h>
#include <cv.h>

using namespace cv;
using namespace std;

int main(){

    Mat image;
    unsigned char *image_data;
    image = imread("inputs/Color-blue.JPG",CV_LOAD_IMAGE_COLOR);   // Read the file
    Size s = image.size();
    int width = s.width;
    int height= s.height;
    image_data = image.data;

    // BGR
    int a=(int)image_data[0];
    int b=(int)image_data[1];
    int c=(int)image_data[2];
    int d=(int)image_data[3];
    int e=(int)image_data[3];
    int f=(int)image_data[3];
    cout<<a<<endl;
    cout<<b<<endl;
    cout<<c<<endl;
    cout<<d<<endl;
    cout<<e<<endl;
    cout<<f<<endl;
  /*
    for (int i=0 ; i < height; i++) {
        for (int j = 0; j < width; j++) {
          int a=(int)image_data[(i*width+j)*3];
          cout<<a<<endl;
        }
  }*/

    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", image);
    waitKey();

		//imwrite("./outputs/test1.png",image);

    return 0;
}

/*
string binary = bitset<8>(128).to_string(); //to binary
cout<<binary<<"\n";

unsigned long decimal = bitset<8>(binary).to_ulong();
cout<<decimal<<"\n";

*/
