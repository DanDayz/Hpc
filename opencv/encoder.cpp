#include <highgui.h>
#include <bits/stdc++.h>
#include <cv.h>

using namespace cv;
using namespace std;

double file_statistics(){
  char my_character;
  string cad;
  ifstream file;

  system("wc -c big.txt > stats.txt");
  file.open("stats.txt");

  while (!file.eof()){
    file.get(my_character);
    if (isspace(my_character)){
        break;
    }else{
      cad.push_back(my_character);
    }
  }
  return atoi(cad.c_str());
}


int main(){

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

    // :::::::::::::::::::::::: Coding

    ifstream file;
    file.open("a.txt");
    //file.open("big.txt");
    char my_character;
    double number_of_character = file_statistics();
    int number_of_lines = 0,i=0,j=0,k=0;

    while (!file.eof()){

      file.get(my_character);
      string binary_char = bitset<8>((int)my_character).to_string();
        cout<<my_character<<endl;
        for(int k = 0 ;k < 3 ; k++){ // D: ugly form to solve the general problem, maybe i will fix in a future.

           if(j%width==0 && j!=0){
                i+=1;
                j=0;
            }

            Vec3f intensity = image.at<Vec3b>(i,j); //(y,x)
            int blue = intensity.val[0];
            int green = intensity.val[1];
            int red = intensity.val[2];

            string binary_blue = bitset<8>(blue).to_string();
            string binary_green = bitset<8>(green).to_string();
            string binary_red = bitset<8>(red).to_string();

            binary_blue[7]=binary_char[k*3];
            binary_green[7]=binary_char[k*3+1];

            //cout<<"blue: "<<binary_blue.c_str()<<endl;
            //cout<<"green: "<<binary_green.c_str()<<endl;

            image_example_data[(i*width+j)*3]=bitset<8>(binary_blue).to_ulong();
            image_example_data[(i*width+j)*3+1]=bitset<8>(binary_green).to_ulong();

            if(k!=2){
            binary_red[7]=binary_char[k*3+2];
            image_example_data[(i*width+j)*3+2]=bitset<8>(binary_red).to_ulong();
            //cout<<"red: "<<bitset<8>(binary_red).to_ulong()<<endl;
          }else{image_example_data[(i*width+j)*3+2]=red;//cout<<"red: "<<red<<endl;
          }

            j++;
      }

        if (my_character == '\n'){
          ++number_of_lines;
        }
    }

    cout<<"i: "<<i<<" j: "<<j<<endl;

    // rest of data
    if(i<height || j < width){
      for (i;i < height; i++) { // y
          while(j < width) { // x
            Vec3f intensity = image.at<Vec3b>(i, j); //(y,x)
            int blue = intensity.val[0];
            int green = intensity.val[1];
            int red = intensity.val[2];
            image_example_data[(i*width+j)*3]=blue;
            image_example_data[(i*width+j)*3+1]=green;
            image_example_data[(i*width+j)*3+2]=red;
            //cout<<"Pixel "<<j<<"x"<<i<<" B: "<<blue<<" G: "<<green<<" R: "<<red<<endl;
            j++;
          }
          j=0;
    }
  }

    cout << "Número de líneas del archivo: " <<number_of_lines<<endl;
    cout << "Número de caracteres: "<<number_of_character<< endl;

    if((( width * height ) / 8) < number_of_character ){
      cout<<"No hay suficientes píxeles para representar su mensaje completo"<<endl;
      cout<<"Elimine: "<<number_of_character - (( width * height ) / 8)<<" Caracteres"<<endl;
      double total_pixels=(( width * height ) * number_of_character) ;
      double div1 = (( width * height )/8) ;
      cout<<"Se necesita una imagen de "<<total_pixels/div1<<" total píxeles para representar su mensaje completamente"<<endl;

      //return 0;
    }


    image_example.create(height,width,CV_8UC3);  // 8 bits 3 Channels
    image_example.data = image_example_data;

    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", image);
    waitKey();

    namedWindow("image2", CV_WINDOW_AUTOSIZE);
    imshow("image2", image_example);
    waitKey();

		imwrite("./outputs/test1.png",image_example);

    return 0;
}
