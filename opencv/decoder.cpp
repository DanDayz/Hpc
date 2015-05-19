#include <highgui.h>
#include <bits/stdc++.h>
#include <cv.h>

using namespace cv;
using namespace std;

int i=0,j=0;
Mat image;  // Read the file, BGR format

int binaryToBase10(int n){
    int output = 0;

    for(int i=0; n > 0; i++) {

        if(n % 10 == 1) {
            output += (1 << i);
        }
        n /= 10;
    }

    return output;
}

//sacamos la longitud de la cadena a leer
int step_one(int height ,int  width){

    string binary;

    for(j;j<2;j++){
      Vec3f intensity = image.at<Vec3b>(0,j); //(y,x)
      short int blue = intensity.val[0];
      short int green = intensity.val[1];
      short int red = intensity.val[2];

      string binary_blue = bitset<8>(blue).to_string();
      string binary_green = bitset<8>(green).to_string();
      string binary_red = bitset<8>(red).to_string();

      binary.push_back(binary_blue[7]);
      binary.push_back(binary_green[7]);
      binary.push_back(binary_red[7]);

      /*cout<<"binary "<<binary<<endl;
      cout<<"B "<<binary_blue[7]<<endl;
      cout<<"G "<<binary_green[7]<<endl;
      cout<<"R "<<binary_red[7]<<endl;
      */

    }
    //cout<<"binary "<<binary<<endl;
    int lc = bitset<6>(binary).to_ulong();
    //cout<<"lc: "<<lc<<endl;
    return lc;
}

int step_two(int lc){

  string binary2;
  Size s = image.size();
  int width = s.width;
  int height= s.height;

  for(int k=0;k<ceil((float)lc/3.0);k++){

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

    if(k*3 < lc){
      binary2.push_back(binary_blue[7]);
      //cout<<"blue: "<<binary_blue.c_str()<<endl;
    }

    if(k*3+1 < lc){
      binary2.push_back(binary_green[7]);
      //cout<<"green: "<<binary_green.c_str()<<endl;
    }

    if(k*3+2 < lc){
      binary2.push_back(binary_red[7]);
      //cout<<"red: "<<bitset<8>(binary_red).to_ulong()<<endl;
    }
    j++;
  }
  //cout<<"binary "<<binary2<<endl;
  int nc=bitset<31>(binary2).to_ulong();
  cout<<"numero de caracteres a leer "<<nc<<endl;
  return nc;
}

void step_three(int height ,int  width,int nc){

  system("touch outputs/decode.txt");
  ofstream file("outputs/decode.txt");

  while(nc>0){
    string binary_char;
    for(int k=0;k<3;k++ ){

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

       binary_char.push_back(binary_blue[7]);
       binary_char.push_back(binary_green[7]);


       if(k<2){
       binary_char.push_back(binary_red[7]);
      }
       j++;
    }
      //cout<<"binary char "<<binary_char<<endl;
      char c = strtol( binary_char.c_str(), (char **)NULL, 2);
      //cout<<"char "<<c<<endl;
      file << c;
      nc--;
      if(i==height){break;}
  }

  file.close();

}

int main(int argc, char** argv){

  if (argc != 2) {
    cerr << "Wrong call\n";
    return 1;
  }

  string foto=argv[1];
  cout<<"foto a decodificar: "<<foto<<endl;
  image= imread(foto,CV_LOAD_IMAGE_COLOR);
  uchar *image_data;
  Size s = image.size();
  int width = s.width;
  int height= s.height;
  uchar *image_example_data = (unsigned char *)malloc(sizeof(unsigned char)*(width*height)*3); // 3 channels
  image_data = image.data;

  cout<<"tamaño de la imágen : "<<width <<" x "<<height<<endl;

  //int lc=step_one(height,width);
  //int nc=step_two(lc);
  //step_three(height,width,nc);

   step_three(height,width,step_two(step_one(height,width)));
   system("nano outputs/decode.txt");  

    return 0;
}
