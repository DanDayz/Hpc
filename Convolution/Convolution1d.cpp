
/* Daniel Diaz Giraldo

Restrictions
Mask = 5, Only works whit odd numbers and Mask size <= N _elements;
N_elements = defined by architecture from machine; (Femin-Maxwell....) in this case
use a Kepler Arch; (Because the number of blocks that can support)

*/

#include <bits/stdc++.h>

#define N_elements 7
using namespace std;

//:::::::::::::::::::::::::::: Host Function ::::::::::::::::::::::::::::::

void h_Convolution_1d(int *In,int *Out, int *Mask, int Mask_size){
  //int Mask_Center = Gap + 1;
  for(int i=0;i<N_elements;i++){
    int Gap=i-(Mask_size)/2; // asymmetric Gap (Left Right)
    int Value=0;
    for(int j=0;j<Mask_size;j++){
      if(Gap+j >= 0 && j+Gap<N_elements){
        Value+=In[Gap+j]*Mask[j];
        //cout<<"i "<<i<<" "<<In[Gap+j]<<" * "<<Mask[j]<<endl;
        //cout<<"j "<<j<<endl;
      }// end if
    }// end for j
    Out[i]=Value;
  }// end for i
}
//:::::::::::::::::::::::::::: Rutinary Functions ::::::::::::::::::::::::::::::

void Fill_elements(int * VecIn1,int Value){
    for (int i = 0; i < N_elements; i++) {
          VecIn1[i]=Value;
    }
}

void Show_vec(int *Vec,int Elements,char * Msg ){
  cout<<Msg<<endl;
  for (int i=0;i<Elements;i++){
    if(i%10==0 && i!=0){
      cout<<endl;
    }
    cout<<"["<<Vec[i]<<"] ";
  }
  cout<<endl;
}

// :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::
double diffclock(clock_t clock1,clock_t clock2){
  double diffticks=clock2-clock1;
  double diffms=(diffticks)/(CLOCKS_PER_SEC/1); // /1000 mili
  return diffms;
}
// :::::::::::::::::::::::::::::::::::::::Main::::::::::::::::::::::::::::::::::

int main(){

  double T1,T2; // Time flags

  int *VecIn1=(int*)malloc(N_elements*sizeof(int)); // Sequential Vector input
  int *VecOut1=(int*)malloc(N_elements*sizeof(int)); // Sequential Vector Output
  int *VecIn2=(int*)malloc(N_elements*sizeof(int)); // Parallel Basic Vector
  int *VecOut2=(int*)malloc(N_elements*sizeof(int)); // Parallel Vector Output
  int Mask_size = 5;
  int *Mask=(int*)malloc(Mask_size*sizeof(int)); // Mask Vector;

  Fill_elements(VecIn1,1);
  Fill_elements(Mask,1);

  Show_vec(VecIn1,N_elements,(char *)"Vector In");
  Show_vec(Mask,Mask_size,(char *)"Mask");
  clock_t start = clock();
	h_Convolution_1d(VecIn1,VecOut1,Mask,Mask_size);
  clock_t end = clock();
  T1=diffclock(start,end);
  cout<<"Serial Result"<<" At "<<T1<<",Seconds"<<endl;
  Show_vec(VecOut1,N_elements,(char *)"Vector Out");

  return 0;
}

/*

Book Test Values  int Mask_size = 5;
#define N_elements 7
VecIn1[0]=1;
VecIn1[1]=2;
VecIn1[2]=3;
VecIn1[3]=4;
VecIn1[4]=5;
VecIn1[5]=6;
VecIn1[6]=7;


Mask[0]=3;
Mask[1]=4;
Mask[2]=5;
Mask[3]=4;
Mask[4]=3;

*/
