# include <bits/stdc++.h>
# include <cuda.h>

#define TILE_WIDTH 32 //(TITLE_WIDTH = BLOCKSIZE)

using namespace std;
// ::::::::::::::::::::::::::::::::::::::::::GPU::::::::::::::::::::::::::::::::

__global__ void KernelNormalMul(int *Mat1,int *Mat2,int *Mat3,int m,int n,int p){
  int j = threadIdx.y + blockDim.y * blockIdx.y; // row
  int i = threadIdx.x + blockDim.x * blockIdx.x; // col

  if((j<m) && (i<p)){
    int value=0;
    for(int k=0;k<n;++k){
      value+=Mat1[n*j+k]*Mat2[p*k+i];
    }
    Mat3[p*j+i]=value;
  }
}


__global__ void  KernelTilesMul(int *Mat1,int *Mat2,int *Mat3,int rowM1,int colM1,int colM2){

  __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  int Pvalue = 0;


  for(int k = 0; k < (colM1+TILE_WIDTH-1)/(TILE_WIDTH); ++k){

    if(k*TILE_WIDTH + tx < colM1 && row < rowM1){
      Mds[ty][tx] = Mat1[row*colM1 + k*TILE_WIDTH + tx];
    }else{
      Mds[ty][tx] = 0;
    }
    if(k*TILE_WIDTH + ty < colM1 && col < colM2){
      Nds[ty][tx] = Mat2[(k*TILE_WIDTH + ty) * colM2 + col];
    }else{
      Nds[ty][tx] =0;
    }

    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; ++k){
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  if (row < rowM1 && col < colM2){
    Mat3[row*colM2+col] = Pvalue;
  }

}



void d_MatrixMult(int *Mat1,int *Mat2,int *Mat3,int rowM1,int colM1,int colM2, int op ){
  int * d_Mat1;
  int * d_Mat2;
  int * d_Mat3;
  float Blocksize=TILE_WIDTH; // Bloque de 2 dimensiones 32*32=256  número de blokes= 1024 (1024/256=4)
  int size1=rowM1*colM1;
  int size2=colM1*colM2;
  int size3=rowM1*colM2;

  // 1. Separamos memoria en el device
  cudaMalloc(&d_Mat1,size1*sizeof(int));
  cudaMalloc(&d_Mat2,size2*sizeof(int));
  cudaMalloc(&d_Mat3,size3*sizeof(int));

  // 2. Copiamos el valor de las variables de host a las variables del device.
  cudaMemcpy(d_Mat1, Mat1,size1*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mat2, Mat2,size2*sizeof(int),cudaMemcpyHostToDevice);
  // 3. Lógica de bloques e hilos, elementos para realizar la parelelización.
  dim3 dimGrid(ceil(colM2/Blocksize),ceil(rowM1/Blocksize),1);
  //dim3 dimGrid((m+Blocksize-1)/Blocksize,(p+Blocksize-1)/Blocksize,1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  // 4. Invocación del kernel (invoción del host, ejecutadas en el device), <<<<#dimGrid,#dimBlock>>>
  if(op==1){KernelNormalMul<<<dimGrid,dimBlock>>>(d_Mat1,d_Mat2,d_Mat3,rowM1,colM1,colM2);}else{
    KernelTilesMul<<<dimGrid,dimBlock>>>(d_Mat1,d_Mat2,d_Mat3,rowM1,colM1,colM2);
  }
  // 5. Copiamos el resultado para mostrar en el I/O del host.
  cudaMemcpy (Mat3,d_Mat3,size3*sizeof(int),cudaMemcpyDeviceToHost);
  // 6. Liberamos memoria.
  cudaFree(d_Mat3);
}

// :::::::::::::::::::::::::::::::::::::::Normal::::::::::::::::::::::::::::::::


void h_Mul_Mat(int *Mat1,int *Mat2, int *Mat3,int m,int n,int p){

  for(int i=0;i<m;i++){
    for(int j=0;j<p;j++){
      int value=0;
      for(int k=0;k<n;k++){
        value+=Mat1[n*i+k]*Mat2[p*k+j];
      }
      Mat3[p*i+j]=value;
    }
  }
}

void llena_mat(int *Mat, int Value,int m,int n){// ver matriz como vector serial.
  int size=n*m; // matriz lineal
  for(int i =0 ; i<size ; i++){
    Mat[i]=Value;
  }
}

void mostrar_mat(int *Mat,int m,int n){//
  int size=n*m; // matriz lineal
  for (int i=0;i<size;i++) {
    if(i%n==0 && n!=0){
      cout<<endl;
    }
    cout<<"["<<Mat[i]<<"] ";
  }
  cout<<endl;
}


int check_mat(int *Mat1,int *Mat2,int m,int p){
  for(int i=0; i<(m*p);++i){
    if(Mat1[i]!=Mat2[i]){
      cout<<"Error, Las matrices no son iguales"<<endl;
      return 0;
    }
  }
  cout<<"Las Matrices son iguales"<<endl;
  return 0;
}



// :::::::::::::::::::::::::::::::::::Clock Function::::::::::::::::::::::::::::

double diffclock(clock_t clock1,clock_t clock2){
  double diffticks=clock2-clock1;
  double diffms=(diffticks)/(CLOCKS_PER_SEC/1); //  /1000 mili
  return diffms;
}

// :::::::::::::::::::::::::::::::::::::::Main::::::::::::::::::::::::::::::::.

int main(){
  double T1,T2,T3; // variables de tiempo

  int rowM1=2;
  int colM1=4;
  int colM2=4;
  int *Mat1 = (int*)malloc((rowM1*colM1)*sizeof(int));
  int *Mat2 = (int*)malloc((colM1*colM2)*sizeof(int));
  int *Mat3 = (int*)malloc((rowM1*colM2)*sizeof(int));
  int *Mat4 = (int*)malloc((rowM1*colM2)*sizeof(int));
  int *Mat5 = (int*)malloc((rowM1*colM2)*sizeof(int));

  llena_mat(Mat1,1,rowM1,colM1);
  llena_mat(Mat2,1,colM1,colM2);

  clock_t start = clock();
  h_Mul_Mat(Mat1,Mat2,Mat3,rowM1,colM1,colM2);
  clock_t end = clock();
  T1=diffclock(start,end);
  cout <<"Tiempo secuencial: "<<T1<<endl;
  mostrar_mat(Mat3,rowM1,colM2);
  clock_t start2 = clock();
  d_MatrixMult(Mat1,Mat2,Mat4,rowM1,colM1,colM2,1); // paralelo
  clock_t end2 = clock();
  mostrar_mat(Mat4,rowM1,colM2);
  T2=diffclock(start2,end2);
  cout <<"Tiempo Paralelo: "<<T2<<endl;
  cout<<"Aceleración lograda: "<<T1/T2<<endl;

  check_mat(Mat3,Mat4,rowM1,colM2);


  clock_t start3 = clock();
  d_MatrixMult(Mat1,Mat2,Mat5,rowM1,colM1,colM2,2); // tiles
  mostrar_mat(Mat5,rowM1,colM2);
  clock_t end3 = clock();
  T3=diffclock(start3,end3);

  cout <<"Tiempo Paralelo con Tiles: "<<T3<<endl;
  cout<<"Aceleración lograda Respecto a el tiempo paralelo: "<<T2/T3<<endl;

  check_mat(Mat4,Mat5,rowM1,colM2);

  free(M1);
  free(M2);
  free(M3);
  free(M4);
  free(M5);

  return 0;
}

// http://www.techdarting.com/2014/03/matrix-multiplication-in-cuda-using.html
