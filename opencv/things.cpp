#include <highgui.h>
#include <bits/stdc++.h>
#include <cv.h>

using namespace cv;
using namespace std;

int main(){

    Mat image,image_example;
    unsigned char *image_data;
    image = imread("inputs/Color-blue.JPG",CV_LOAD_IMAGE_COLOR);   // Read the file
    Size s = image.size();
    int width = s.width;
    int height= s.height;
    unsigned char *image_example_data = (unsigned char *)malloc(sizeof(unsigned char)*((width*height)*3)); // 3 channels
    image_data = image.data;

    cout<<"Ancho de la imágen: "<<width<<endl;
    cout<<"Alto de la imágen: "<<height<<endl;

    // BGR
    int a=(int)image_data[0]; //B
    int b=(int)image_data[1]; //G
    int c=(int)image_data[2]; //R

    //cout<<"B: "<<a<<endl;
    //cout<<"G: "<<b<<endl;
    //cout<<"R: "<<c<<endl;


    for (int i=0 ; i < height; i++) {
        for (int j = 0; j < width; j++) {
          /*unsigned int B=(int)image_data[(i*width+j)*3+0];
          unsigned int G=(int)image_data[(i*width+j)*3+1];
          unsigned int R=(int)image_data[(i*width+j)*3+2];
          //cout<<"Pixel: "<<j<<"x"<<i<<" B: "<<B<<" G: "<<G<<" R: "<<R<<endl;
          image_example_data[(i*width+j)*3]=B;
          image_example_data[(i*width+j)*3+1]=G;
          image_example_data[(i*width+j)*3+2]=R;
        */

        }
  }

    image_example.create(height,width,CV_8UC3);  // 8 bits 3 Channels
    image_example.data = image_example_data;

    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", image);
    waitKey();

    namedWindow("image2", CV_WINDOW_AUTOSIZE);
    imshow("image2", image_example);
    waitKey();

    Vec3f intensity = image.at<Vec3b>(150, 100);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];
    cout<<"Pixel "<<j<<"x"<<i<<" B: "<<(int)blue<<" G: "<<(int)green<<" R: "<<(int)red<<endl;

		//imwrite("./outputs/test1.png",image);

    return 0;
}

/*
string binary = bitset<8>(128).to_string(); //to binary
cout<<binary<<"\n";

unsigned long decimal = bitset<8>(binary).to_ulong();
cout<<decimal<<"\n";

*/
#include <highgui.h>
#include <bits/stdc++.h>
#include <cv.h>

using namespace cv;
using namespace std;

int main(){

    ifstream file;
    //file.open("a.txt", ios::in);
    file.open("big.txt");
    char my_character;
    double number_of_character=0;
    int number_of_lines = 0;

    while (!file.eof() ) {

    file.get(my_character);
    //cout << my_character;
    ++number_of_character;
      if (my_character == '\n'){
        ++number_of_lines;
      }
    }
    cout << "Número de líneas del archivo: " <<number_of_lines<<endl;
    cout << "Número de caracteres: "<<number_of_character<< endl;

    Mat image,image_example;
    uchar *image_data;
    image = imread("inputs/Color-blue.JPG",CV_LOAD_IMAGE_COLOR);   // Read the file, BGR format
    //image = imread("inputs/tpr.jpg",CV_LOAD_IMAGE_COLOR);
    Size s = image.size();
    int width = s.width;
    int height= s.height;
    uchar *image_example_data = (unsigned char *)malloc(sizeof(unsigned char)*(width*height)*3); // 3 channels
    image_data = image.data;

    cout<<"Ancho de la imágen: "<<width<<endl;
    cout<<"Alto de la imágen: "<<height<<endl;

    if((( width * height ) / 8) < number_of_character ){
      cout<<"No hay suficientes píxeles para representar su mensaje"<<endl;
      cout<<"Elimine: "<<number_of_character - (( width * height ) / 8)<<" Caracteres"<<endl;
      double total_pixels=(( width * height ) * number_of_character) ;
      double div1 = (( width * height )/8) ;
      cout<<"Se necesita una imagen de "<<total_pixels/div1<<" total píxeles para representar su mensaje"<<endl;

      return 0;
    }

    for (int i=0 ; i < height; i++) { // y
        for (int j = 0; j < width; j++) { // x
          Vec3f intensity = image.at<Vec3b>(i, j); //(y,x)
          int blue = intensity.val[0];
          int green = intensity.val[1];
          int red = intensity.val[2];
          // change values from message values.
            //pass char to int (ascii value)
            //pass int (ascii value) to 1 byte
            //
          //

          image_example_data[(i*width+j)*3]=blue;
          image_example_data[(i*width+j)*3+1]=green;
          image_example_data[(i*width+j)*3+2]=red;

          //string binary_b = bitset<8>(blue).to_string();
          //string binary_g = bitset<8>(green).to_string();
          //string binary_r = bitset<8>(red).to_string();

          //cout<<"Pixel "<<j<<"x"<<i<<" B: "<<blue<<" G: "<<green<<" R: "<<red<<endl;

        }
  }
    image_example.create(height,width,CV_8UC3);  // 8 bits 3 Channels
    image_example.data = image_example_data;

    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", image);
    waitKey();

    namedWindow("image2", CV_WINDOW_AUTOSIZE);
    imshow("image2", image_example);
    waitKey();

		imwrite("./outputs/test1.png",image);

    return 0;
}

/*
string binary = bitset<8>(128).to_string(); //to binary
cout<<binary<<"\n";

unsigned long decimal = bitset<8>(binary).to_ulong();
cout<<decimal<<"\n";

*/
int file_statistics(){
  char my_character;
  string cad;
  ifstream file;
  int i=0;

  system("wc -c big.txt > stats.txt");
  file.open("stats.txt");

  while (!file.eof()){
    if (isspace(my_character)){
        break;
    }
    cad[i]=(my_character);
    i++;
  }

  return atoi(cad.c_str());


}
